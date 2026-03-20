import torch
import math
from typing import NamedTuple, Optional, Dict, Any, Protocol, TypedDict
from tqdm.auto import tqdm
import comfy.model_sampling

# ==========================================
# 1. Protocols & Types
# ==========================================

class DenoiserModel(Protocol):
    """Protocol for k-diffusion/ComfyUI model wrappers."""
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

# ==========================================
# 2. Optimized Math Helpers (Tensor-Native)
# ==========================================

def _phi_1(neg_h: torch.Tensor) -> torch.Tensor:
    """Computes phi_1(x) = (exp(x) - 1) / x numerically stably."""
    return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

def _phi_2(neg_h: torch.Tensor) -> torch.Tensor:
    """Computes phi_2(x) = (exp(x) - x - 1) / x^2 numerically stably."""
    return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h.square(), nan=0.5)

def _de_second_order(h: torch.Tensor, c2: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates RES coefficients using Crowson's formulation.
    All inputs/outputs are tensors on the correct device.
    """
    # Create c2 tensor on device once
    c2_t = torch.tensor(c2, device=h.device, dtype=h.dtype)
    
    # Pre-calculate common terms
    neg_h = -h
    neg_c2_h = -c2_t * h
    
    phi1 = _phi_1(neg_h)
    phi2 = _phi_2(neg_h)
    phi_1_c2 = _phi_1(neg_c2_h)
    
    # Calculate coefficients (Eq. Table 3 in paper)
    a2_1 = c2_t * phi_1_c2
    phi2_div_c2 = phi2 / c2_t
    
    b1 = phi1 - phi2_div_c2
    b2 = phi2_div_c2
    
    return a2_1, b1, b2

# ==========================================
# 3. Adaptive Momentum Logic
# ==========================================

class MomentumTracker:
    """
    Manages momentum state with annealing strategies.
    strategies: 'static', 'linear', 'cosine'
    """
    def __init__(self, base_momentum: float, strategy: str = "cosine"):
        self.base_momentum = base_momentum
        self.strategy = strategy
        self.velocity: Optional[torch.Tensor] = None
    
    def __repr__(self):
        return f"MomentumTracker(base={self.base_momentum}, strategy='{self.strategy}')"
        
    def apply(self, current_update: torch.Tensor, sigma: float, sigma_max: float) -> torch.Tensor:
        """
        Applies momentum: v_t = m * v_{t-1} + (1-m) * g_t
        """
        if self.base_momentum <= 0:
            return current_update

        # Initialize
        if self.velocity is None:
            self.velocity = current_update
            return current_update

        # Annealing Calculation
        # Progress goes from 1.0 (High Noise) -> 0.0 (Low Noise)
        progress = max(0.0, min(1.0, sigma / sigma_max))
        
        if self.strategy == "linear":
            eff_momentum = self.base_momentum * progress
        elif self.strategy == "cosine":
            eff_momentum = self.base_momentum * 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
        else: # static
            eff_momentum = self.base_momentum

        # Apply Momentum Update
        # Note: We use in-place add if possible to save a micro-alloc, but standard ops are safer for autograd
        self.velocity = (eff_momentum * self.velocity) + ((1.0 - eff_momentum) * current_update)
        
        return self.velocity

# ==========================================
# 4. Core Step Function
# ==========================================

def get_model_sampling(model):
    for path in (
        ("inner_model", "inner_model", "model_sampling"),  # what comfy/k_diffusion/sampling.py uses [web:7]
        ("inner_model", "model_sampling"),                 # some wrappers expose it here (seen in custom nodes) [web:70]
        ("model_sampling",),                               # fallback
    ):
        obj = model
        ok = True
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    return None

def _rk2_rf_coeffs(c2: float):
    # 2nd-order 2-stage RK with free c2 in (0,1]
    b2 = 1.0 / (2.0 * c2)
    b1 = 1.0 - b2
    a21 = c2
    return a21, b1, b2

def _rf_drift(x, denoised, sigma, eps=1e-12):
    # dx/dsigma = (x - denoised)/sigma
    return (x - denoised) / sigma.clamp_min(eps)

def _refined_exp_sosu_step(
    model: DenoiserModel,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    momentum_1: MomentumTracker,
    momentum_2: MomentumTracker,
    sigma_max: float,
    s_in: torch.Tensor,
    c2: float = 0.5,
    extra_args: Optional[Dict[str, Any]] = None,
    pbar: Optional[tqdm] = None,
) -> StepOutput:
    extra_args = {} if extra_args is None else extra_args
    
    # 0. RF/Flow Matching Branch
    # --------------------------
    ms = get_model_sampling(model)
    is_flow = (ms is not None) and isinstance(ms, comfy.model_sampling.CONST)

    if is_flow:
        # RF path: integrate in sigma directly (ODE: dx/dsigma = (x - denoised)/sigma)
        # h is purely sigma step, not log-sigma
        h = sigma_next - sigma
        
        # 2-stage RK2 coefficients
        a21, b1, b2 = _rk2_rf_coeffs(c2)

        # Stage 1
        denoised = model(x, sigma * s_in, **extra_args)
        if pbar: pbar.update(0.5)
        
        k1 = _rf_drift(x, denoised, sigma)
        
        # Intermediate update (x2)
        raw_update_2 = (a21 * h) * k1
        
        # Use existing momentum tracker (momentum_2)
        sig_scalar = sigma.item()
        vel_2 = momentum_2.apply(raw_update_2, sig_scalar, sigma_max)
        
        x_2 = x + vel_2
        sigma_2 = sigma + (a21 * h)

        # Stage 2
        denoised2 = model(x_2, sigma_2 * s_in, **extra_args)
        if pbar: pbar.update(0.5)
        
        k2 = _rf_drift(x_2, denoised2, sigma_2)


        # Final update
        raw_update_final = h * (b1 * k1 + b2 * k2)
        vel_1 = momentum_1.apply(raw_update_final, sig_scalar, sigma_max)
        
        x_next = x + vel_1

        return StepOutput(
            x_next=x_next,
            denoised=denoised,
            denoised2=denoised2,
            vel=vel_1,
            vel_2=vel_2
        )

    # 1. Standard Diffusion (Log-space) Path
    # --------------------------------------
    lam = sigma.log().neg()
    lam_next = sigma_next.log().neg()
    h = lam_next - lam
    
    # 2. Coefficients
    a2_1, b1, b2 = _de_second_order(h, c2)
    
    # 3. Stage 1: Denoise at sigma
    denoised = model(x, sigma * s_in, **extra_args)
    if pbar: pbar.update(0.5)

    # 4. Stage 1: Update to intermediate x_2
    # Calculate RES update vector
    raw_update_2 = a2_1 * h * denoised
    
    # Apply Momentum (Note: passing scalar sigma via .item())
    sig_scalar = sigma.item()
    vel_2 = momentum_2.apply(raw_update_2, sig_scalar, sigma_max)
    
    # x_2 calculation
    c2_h = c2 * h
    x_2 = (-c2_h).exp() * x + vel_2
    
    # 5. Stage 2: Denoise at intermediate sigma_2
    lam_2 = lam + c2_h
    sigma_2 = (-lam_2).exp()
    
    denoised2 = model(x_2, sigma_2 * s_in, **extra_args)
    if pbar: pbar.update(0.5)

    # 6. Stage 2: Final Update to x_next
    raw_update_final = h * (b1 * denoised + b2 * denoised2)
    vel_1 = momentum_1.apply(raw_update_final, sig_scalar, sigma_max)
    
    x_next = (-h).exp() * x + vel_1
    
    return StepOutput(
        x_next=x_next,
        denoised=denoised,
        denoised2=denoised2,
        vel=vel_1,
        vel_2=vel_2
    )

# ==========================================
# 5. Sampler Entry Point
# ==========================================

@torch.no_grad()
def sample_refined_exp_s_v4(
    model: DenoiserModel,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    denoise_to_zero: bool = True,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[RefinedExpCallback] = None,
    disable: Optional[bool] = None,
    ita: float = 0.0, 
    c2: float = 0.5,
    noise_sampler = torch.randn_like,
    momentum: float = 0.5,
    momentum_strategy: str = "cosine" 
):
    """
    Refined Exponential Solver (S) with Adaptive Momentum (Shiro-Optimized V4).
    """
    sigmas = sigmas.to(x.device, dtype=x.dtype)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    sigma_max_val = sigma_max.item()
    extra_args = {} if extra_args is None else extra_args

    if noise_sampler is torch.randn_like:
        _noise_sampler = lambda sigma, sigma_next: noise_sampler(x)
    else:
        _noise_sampler = noise_sampler

    m_tracker_1 = MomentumTracker(momentum, momentum_strategy)
    m_tracker_2 = MomentumTracker(momentum, momentum_strategy)
    brownian_fallback = False

    # 1a. RF/Flow safeguards
    ms = get_model_sampling(model)
    is_flow = (ms is not None) and isinstance(ms, comfy.model_sampling.CONST)
    # (ita force-zero removed to support RF ancestral)

    # 1b. Batch helper (avoid alloc inside loop)
    s_in = x.new_ones([x.shape[0]])

    # 1. Determine main loop steps (avoiding the final 0.0)
    has_terminal_zero = float(sigmas[-1].item()) == 0.0
    if denoise_to_zero and has_terminal_zero:
        main_steps = len(sigmas) - 2
    else:
        main_steps = len(sigmas) - 1
    
    if main_steps < 0: main_steps = 0

    # 2. Calculate TOTAL steps for the bar (Main + Cleanup)
    total_steps = main_steps
    if denoise_to_zero:
        total_steps += 1

    # 3. Start the progress bar context
    with tqdm(disable=disable, total=total_steps) as pbar:
        # Run the RES ODE steps
        for i in range(main_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            step_sigma_next = sigma_next
            rf_ancestral = False

            # RF Ancestral Logic
            # Fix: .item() check for tensor boolean safety
            if is_flow and ita > 0 and sigma_next.item() > 0:
                rf_ancestral = True
                # Math from comfy.k_diffusion.sampling.sample_euler_ancestral_RF
                downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * ita
                sigma_down = sigma_next * downstep_ratio
                step_sigma_next = sigma_down

            # Stochastic Injection (Standard Diffusion only)
            if ita > 0 and not is_flow:
                if not brownian_fallback:
                    try:
                        eps = _noise_sampler(sigma, sigma_next)
                    except RecursionError:
                        print("RESv4: Recursion limit hit, using Gaussian.")
                        eps = torch.randn_like(x)
                        brownian_fallback = True
                else:
                    eps = torch.randn_like(x)
                sigma_hat = sigma * (1 + ita)
                noise_scale = (sigma_hat.square() - sigma.square()).sqrt()
                x = x + noise_scale * eps
                sigma = sigma_hat
            
            # RES Step (updates pbar by 0.5 + 0.5 internally)
            outs = _refined_exp_sosu_step(
                model, x, sigma, step_sigma_next,
                m_tracker_1, m_tracker_2, sigma_max_val,
                s_in=s_in,
                c2=c2, extra_args=extra_args, pbar=pbar
            )
            x = outs.x_next

            # RF Renoise (Post-Step)
            if rf_ancestral:
                alpha_ip1 = 1.0 - sigma_next
                alpha_down = 1.0 - step_sigma_next
                
                # Numerical stability: clamp inner term >= 0 before sqrt
                # renoise_coeff = sqrt(sig_next^2 - sig_down^2 * (alpha_ip1/alpha_down)^2)
                eps_val = torch.finfo(x.dtype).eps
                alpha_ratio_sq = (alpha_ip1 ** 2) / (alpha_down ** 2 + eps_val)
                inner_term = sigma_next**2 - (step_sigma_next**2 * alpha_ratio_sq)
                renoise_coeff = torch.clamp(inner_term, min=0.0).sqrt()
                
                # Retrieve noise (using original endpoints matches ComfyUI convention)
                if not brownian_fallback:
                    try:
                        eps_noise = _noise_sampler(sigmas[i], sigmas[i+1])
                    except RecursionError:
                        eps_noise = torch.randn_like(x)
                        brownian_fallback = True
                else:
                    eps_noise = torch.randn_like(x)
                
                x = (alpha_ip1 / (alpha_down + eps_val)) * x + eps_noise * renoise_coeff
            
            if callback is not None:
                payload = RefinedExpCallbackPayload(
                    x=x, i=i, sigma=sigma, sigma_hat=sigma, 
                    denoised=outs.denoised, denoised2=outs.denoised2
                )
                callback(payload)

        # 4. Final Cleanup Step (Now INSIDE the 'with' block)
        if denoise_to_zero:
            last_sigma = sigmas[-2] if sigmas[-1] == 0 else sigmas[-1]
            
            if ita > 0 and not is_flow:
                # Same noise logic for consistency
                if not brownian_fallback:
                    try:
                        eps = _noise_sampler(last_sigma, sigmas[-1])
                    except RecursionError:
                         eps = torch.randn_like(x)
                else:
                    eps = torch.randn_like(x)
                sigma_hat = last_sigma * (1 + ita)
                noise_scale = (sigma_hat.square() - last_sigma.square()).sqrt()
                x = x + noise_scale * eps
                last_sigma = sigma_hat

            # Final Model Call
            x = model(x, last_sigma * s_in, **extra_args)
            
            # Update the final tick
            pbar.update(1)
            
    return x
