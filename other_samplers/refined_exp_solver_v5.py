import math
from typing import NamedTuple, Optional, Dict, Any, Protocol, TypedDict

import comfy.model_sampling
import torch
from tqdm.auto import tqdm


class DenoiserModel(Protocol):
    def __call__(self, x: torch.Tensor, t: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class RefinedExpCallbackPayload(TypedDict):
    x: torch.Tensor
    i: int
    sigma: torch.Tensor
    sigma_hat: torch.Tensor
    denoised: torch.Tensor
    denoised2: torch.Tensor


class RefinedExpCallback(Protocol):
    def __call__(self, payload: RefinedExpCallbackPayload) -> None: ...


class StepOutput(NamedTuple):
    x_next: torch.Tensor
    denoised: torch.Tensor
    denoised2: torch.Tensor
    vel: Optional[torch.Tensor]
    vel_2: Optional[torch.Tensor]


def _phi_1(neg_h: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)


def _phi_2(neg_h: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h.square(), nan=0.5)


def _clamp_c2(c2: float) -> float:
    return max(1.0e-3, min(float(c2), 1.0))


def _de_second_order(h: torch.Tensor, c2: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c2_t = torch.tensor(_clamp_c2(c2), device=h.device, dtype=h.dtype)
    neg_h = -h
    neg_c2_h = -c2_t * h

    phi1 = _phi_1(neg_h)
    phi2 = _phi_2(neg_h)
    phi_1_c2 = _phi_1(neg_c2_h)

    a2_1 = c2_t * phi_1_c2
    phi2_div_c2 = phi2 / c2_t

    b1 = phi1 - phi2_div_c2
    b2 = phi2_div_c2
    return a2_1, b1, b2


class MomentumTracker:
    """
    Schedule-aware momentum with an adaptive mode that softens near the tail
    and damps sudden update spikes.
    """

    def __init__(self, base_momentum: float, strategy: str = "adaptive"):
        self.base_momentum = max(0.0, min(float(base_momentum), 0.999))
        self.strategy = strategy
        self.velocity: Optional[torch.Tensor] = None
        self.prev_update_norm: Optional[float] = None

    def _schedule_progress(
        self,
        sigma: float,
        sigma_min: float,
        sigma_max: float,
        step_index: int,
        total_steps: int,
    ) -> float:
        eps = 1.0e-12
        sigma = max(sigma, sigma_min, eps)
        sigma_min = max(sigma_min, eps)
        sigma_max = max(sigma_max, sigma, sigma_min + eps)

        denom = math.log(sigma_max) - math.log(sigma_min)
        if abs(denom) <= eps:
            sigma_progress = 1.0
        else:
            sigma_progress = (math.log(sigma) - math.log(sigma_min)) / denom
            sigma_progress = max(0.0, min(1.0, sigma_progress))

        if total_steps <= 1:
            step_progress = sigma_progress
        else:
            step_progress = 1.0 - (step_index / max(total_steps - 1, 1))

        return (0.65 * sigma_progress) + (0.35 * step_progress)

    def _anneal(self, progress: float) -> float:
        if self.strategy == "static":
            return self.base_momentum
        if self.strategy == "linear":
            return self.base_momentum * progress
        if self.strategy == "cosine":
            return self.base_momentum * 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))

        # adaptive: cosine-like early behavior with stronger late-step damping
        cosine = self.base_momentum * 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
        tail_soften = 0.15 + (0.85 * (progress ** 1.35))
        return cosine * tail_soften

    def apply(
        self,
        current_update: torch.Tensor,
        sigma: float,
        sigma_min: float,
        sigma_max: float,
        step_index: int,
        total_steps: int,
        disagreement: float = 0.0,
        stage_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.base_momentum <= 0:
            return current_update

        update_norm = float(torch.linalg.vector_norm(current_update.detach().float()).item())
        if self.velocity is None:
            self.velocity = current_update
            self.prev_update_norm = update_norm
            return current_update

        eff_momentum = self._anneal(
            self._schedule_progress(sigma, sigma_min, sigma_max, step_index, total_steps)
        )
        eff_momentum *= max(0.0, min(stage_scale, 1.0))

        if self.prev_update_norm is not None and self.prev_update_norm > 0:
            norm_ratio = update_norm / self.prev_update_norm
            if norm_ratio > 1.25:
                eff_momentum /= min(norm_ratio, 2.5)
            elif self.strategy == "adaptive" and norm_ratio < 0.75:
                eff_momentum = min(self.base_momentum, eff_momentum * 1.05)

        if self.strategy == "adaptive" and disagreement > 0:
            # When stage predictions disagree, trust fresh information more than history.
            disagreement_dampen = 1.0 / (1.0 + min(disagreement, 3.0) * 2.0)
            eff_momentum *= disagreement_dampen

        self.velocity = (eff_momentum * self.velocity) + ((1.0 - eff_momentum) * current_update)
        self.prev_update_norm = update_norm
        return self.velocity


def _relative_update_disagreement(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1.0e-12) -> float:
    ref_norm = float(torch.linalg.vector_norm(reference.detach().float()).item())
    diff_norm = float(torch.linalg.vector_norm((candidate - reference).detach().float()).item())
    return diff_norm / max(ref_norm, eps)


def get_model_sampling(model):
    for path in (
        ("inner_model", "inner_model", "model_sampling"),
        ("inner_model", "model_sampling"),
        ("model_sampling",),
    ):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _rk2_rf_coeffs(c2: float):
    c2 = _clamp_c2(c2)
    b2 = 1.0 / (2.0 * c2)
    b1 = 1.0 - b2
    a21 = c2
    return a21, b1, b2


def _rf_drift(x, denoised, sigma, eps=1e-12):
    return (x - denoised) / sigma.clamp_min(eps)


def _resolve_sigma_bounds(sigmas: torch.Tensor) -> tuple[float, float]:
    positive_sigmas = sigmas[sigmas > 0]
    if positive_sigmas.numel() == 0:
        sigma_max = float(sigmas.max().item()) if sigmas.numel() > 0 else 1.0
        sigma_min = max(1.0e-3, sigma_max)
        return sigma_min, sigma_max
    return float(positive_sigmas.min().item()), float(sigmas.max().item())


def _noise_sample(
    noise_sampler,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    fallback_active: bool,
    label: str,
) -> tuple[torch.Tensor, bool]:
    if fallback_active:
        return torch.randn_like(x), True
    try:
        return noise_sampler(sigma, sigma_next), False
    except RecursionError:
        print(f"{label}: Recursion limit hit, using Gaussian.")
        return torch.randn_like(x), True


def _refined_exp_sosu_step(
    model: DenoiserModel,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    momentum_1: MomentumTracker,
    momentum_2: MomentumTracker,
    sigma_min: float,
    sigma_max: float,
    step_index: int,
    total_steps: int,
    s_in: torch.Tensor,
    c2: float = 0.5,
    extra_args: Optional[Dict[str, Any]] = None,
    pbar: Optional[tqdm] = None,
) -> StepOutput:
    extra_args = {} if extra_args is None else extra_args
    ms = get_model_sampling(model)
    is_flow = (ms is not None) and isinstance(ms, comfy.model_sampling.CONST)

    if is_flow:
        h = sigma_next - sigma
        a21, b1, b2 = _rk2_rf_coeffs(c2)

        denoised = model(x, sigma * s_in, **extra_args)
        if pbar:
            pbar.update(0.5)

        k1 = _rf_drift(x, denoised, sigma)
        raw_update_2 = (a21 * h) * k1
        sig_scalar = float(sigma.item())
        vel_2 = momentum_2.apply(
            raw_update_2,
            sig_scalar,
            sigma_min,
            sigma_max,
            step_index,
            total_steps,
            stage_scale=0.85,
        )

        x_2 = x + vel_2
        sigma_2 = sigma + (a21 * h)

        denoised2 = model(x_2, sigma_2 * s_in, **extra_args)
        if pbar:
            pbar.update(0.5)

        k2 = _rf_drift(x_2, denoised2, sigma_2)
        disagreement = _relative_update_disagreement(k1, k2)
        raw_update_final = h * (b1 * k1 + b2 * k2)
        vel_1 = momentum_1.apply(
            raw_update_final,
            sig_scalar,
            sigma_min,
            sigma_max,
            step_index,
            total_steps,
            disagreement=disagreement,
        )

        return StepOutput(
            x_next=x + vel_1,
            denoised=denoised,
            denoised2=denoised2,
            vel=vel_1,
            vel_2=vel_2,
        )

    lam = sigma.log().neg()
    lam_next = sigma_next.log().neg()
    h = lam_next - lam

    a2_1, b1, b2 = _de_second_order(h, c2)

    denoised = model(x, sigma * s_in, **extra_args)
    if pbar:
        pbar.update(0.5)

    raw_update_2 = a2_1 * h * denoised
    sig_scalar = float(sigma.item())
    vel_2 = momentum_2.apply(
        raw_update_2,
        sig_scalar,
        sigma_min,
        sigma_max,
        step_index,
        total_steps,
        stage_scale=0.85,
    )

    c2_h = _clamp_c2(c2) * h
    x_2 = (-c2_h).exp() * x + vel_2
    lam_2 = lam + c2_h
    sigma_2 = (-lam_2).exp()

    denoised2 = model(x_2, sigma_2 * s_in, **extra_args)
    if pbar:
        pbar.update(0.5)

    disagreement = _relative_update_disagreement(denoised, denoised2)
    raw_update_final = h * (b1 * denoised + b2 * denoised2)
    vel_1 = momentum_1.apply(
        raw_update_final,
        sig_scalar,
        sigma_min,
        sigma_max,
        step_index,
        total_steps,
        disagreement=disagreement,
    )

    return StepOutput(
        x_next=(-h).exp() * x + vel_1,
        denoised=denoised,
        denoised2=denoised2,
        vel=vel_1,
        vel_2=vel_2,
    )


@torch.no_grad()
def sample_refined_exp_s_v5(
    model: DenoiserModel,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    denoise_to_zero: bool = True,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[RefinedExpCallback] = None,
    disable: Optional[bool] = None,
    ita: float = 0.0,
    c2: float = 0.5,
    noise_sampler=torch.randn_like,
    momentum: float = 0.4,
    momentum_strategy: str = "adaptive",
):
    sigmas = sigmas.to(x.device, dtype=x.dtype)
    sigma_min, sigma_max = _resolve_sigma_bounds(sigmas)
    c2 = _clamp_c2(c2)
    ita = max(0.0, float(ita))
    momentum = max(0.0, min(float(momentum), 0.999))
    extra_args = {} if extra_args is None else extra_args

    if noise_sampler is torch.randn_like:
        _noise_sampler = lambda sigma, sigma_next: noise_sampler(x)
    else:
        _noise_sampler = noise_sampler

    m_tracker_1 = MomentumTracker(momentum, momentum_strategy)
    m_tracker_2 = MomentumTracker(momentum, momentum_strategy)
    brownian_fallback = False

    ms = get_model_sampling(model)
    is_flow = (ms is not None) and isinstance(ms, comfy.model_sampling.CONST)

    s_in = x.new_ones([x.shape[0]])
    has_terminal_zero = bool(sigmas.numel() > 0 and float(sigmas[-1].item()) == 0.0)
    if denoise_to_zero and has_terminal_zero:
        main_steps = len(sigmas) - 2
    else:
        main_steps = len(sigmas) - 1
    main_steps = max(0, main_steps)

    total_steps = main_steps + (1 if denoise_to_zero else 0)

    with tqdm(disable=disable, total=total_steps) as pbar:
        for i in range(main_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_hat = sigma
            step_sigma_next = sigma_next
            rf_ancestral = False

            if is_flow and ita > 0 and sigma_next.item() > 0:
                rf_ancestral = True
                downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * ita
                step_sigma_next = sigma_next * downstep_ratio

            if ita > 0 and not is_flow:
                eps, fallback_used = _noise_sample(
                    _noise_sampler, x, sigma, sigma_next, brownian_fallback, "RESv5"
                )
                brownian_fallback = brownian_fallback or fallback_used
                sigma_hat = sigma * (1 + ita)
                noise_scale = (sigma_hat.square() - sigma.square()).sqrt()
                x = x + (noise_scale * eps)

            outs = _refined_exp_sosu_step(
                model=model,
                x=x,
                sigma=sigma_hat,
                sigma_next=step_sigma_next,
                momentum_1=m_tracker_1,
                momentum_2=m_tracker_2,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                step_index=i,
                total_steps=max(main_steps, 1),
                s_in=s_in,
                c2=c2,
                extra_args=extra_args,
                pbar=pbar,
            )
            x = outs.x_next

            if rf_ancestral:
                alpha_ip1 = 1.0 - sigma_next
                alpha_down = 1.0 - step_sigma_next
                eps_val = torch.finfo(x.dtype).eps
                alpha_ratio_sq = (alpha_ip1 ** 2) / (alpha_down ** 2 + eps_val)
                inner_term = sigma_next**2 - (step_sigma_next**2 * alpha_ratio_sq)
                renoise_coeff = torch.clamp(inner_term, min=0.0).sqrt()

                eps_noise, fallback_used = _noise_sample(
                    _noise_sampler, x, sigmas[i], sigmas[i + 1], brownian_fallback, "RESv5"
                )
                brownian_fallback = brownian_fallback or fallback_used
                x = (alpha_ip1 / (alpha_down + eps_val)) * x + eps_noise * renoise_coeff

            if callback is not None:
                payload = RefinedExpCallbackPayload(
                    x=x,
                    i=i,
                    sigma=sigma,
                    sigma_hat=sigma_hat,
                    denoised=outs.denoised,
                    denoised2=outs.denoised2,
                )
                callback(payload)

        if denoise_to_zero:
            last_sigma = sigmas[-2] if has_terminal_zero else sigmas[-1]

            if ita > 0 and not is_flow:
                eps, fallback_used = _noise_sample(
                    _noise_sampler, x, last_sigma, sigmas[-1], brownian_fallback, "RESv5"
                )
                brownian_fallback = brownian_fallback or fallback_used
                sigma_hat = last_sigma * (1 + ita)
                noise_scale = (sigma_hat.square() - last_sigma.square()).sqrt()
                x = x + (noise_scale * eps)
                last_sigma = sigma_hat

            x = model(x, last_sigma * s_in, **extra_args)
            pbar.update(1)

    return x
