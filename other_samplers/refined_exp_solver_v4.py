import torch
import math
from typing import NamedTuple, Optional, Dict, Any, Protocol, TypedDict
from tqdm.auto import tqdm

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

def _refined_exp_sosu_step(
    model: DenoiserModel,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    momentum_1: MomentumTracker,
    momentum_2: MomentumTracker,
    sigma_max: float,
    c2: float = 0.5,
    extra_args: Dict[str, Any] = {},
    pbar: Optional[tqdm] = None,
) -> StepOutput:
    
    # 1. Log-space Timesteps & Step Size (h)
    lam = sigma.log().neg()
    lam_next = sigma_next.log().neg()
    h = lam_next - lam
    
    # 2. Coefficients
    a2_1, b1, b2 = _de_second_order(h, c2)
    
    # 3. Stage 1: Denoise at sigma
    denoised = model(x, sigma, **extra_args)
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
    
    denoised2 = model(x_2, sigma_2, **extra_args)
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
    extra_args: Dict[str, Any] = {},
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
    # Align devices
    sigmas = sigmas.to(x.device, dtype=x.dtype)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    sigma_max_val = sigma_max.item()

    # Trackers for the two RES stages
    m_tracker_1 = MomentumTracker(momentum, momentum_strategy)
    m_tracker_2 = MomentumTracker(momentum, momentum_strategy)

    # Pbar calculation: (N-1) steps * 1.0 (split into two 0.5s) + 1.0 if final denoise
    total_steps = len(sigmas) - (1 if denoise_to_zero else 2)
    
    with tqdm(disable=disable, total=total_steps) as pbar:
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Stochastic Injection (Euler-Ancestral style)
            if ita > 0:
                eps = noise_sampler(x)
                sigma_hat = sigma * (1 + ita)
                noise_scale = (sigma_hat.square() - sigma.square()).sqrt()
                x = x + noise_scale * eps
                sigma = sigma_hat
            
            # Perform RES Step
            outs = _refined_exp_sosu_step(
                model, x, sigma, sigma_next,
                m_tracker_1, m_tracker_2, sigma_max_val,
                c2=c2, extra_args=extra_args, pbar=pbar
            )
            
            x = outs.x_next
            
            # Callback
            if callback is not None:
                payload = RefinedExpCallbackPayload(
                    x=x, i=i, sigma=sigma, sigma_hat=sigma, 
                    denoised=outs.denoised, denoised2=outs.denoised2
                )
                callback(payload)

    # Final "Cleanup" Step to 0
    if denoise_to_zero:
        # We take the last used sigma (before 0) and run the model one last time.
        # Since D(x, sigma) predicts x0, this jumps straight to the clean image.
        last_sigma = sigmas[-2] if sigmas[-1] == 0 else sigmas[-1]
        
        # Consistent stochasticity check
        if ita > 0:
            eps = noise_sampler(x)
            sigma_hat = last_sigma * (1 + ita)
            noise_scale = (sigma_hat.square() - last_sigma.square()).sqrt()
            x = x + noise_scale * eps
            last_sigma = sigma_hat

        x = model(x, last_sigma, **extra_args)
        
        if pbar: pbar.update(1)
            
    return x
