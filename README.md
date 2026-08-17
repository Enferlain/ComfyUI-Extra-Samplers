# ComfyUI Extra Samplers

samplerres_exp_v4 noise_sampler practical take:

- `gaussian`: safest general-purpose choice. Best baseline if you want predictable behavior.
- `brownian`: often the best choice for true ancestral/SDE-style sampling, because it preserves noise continuity across sigma steps instead of drawing unrelated noise each time.
- `uniform`: can feel a little flatter or broader; worth testing if gaussian feels too conventional.
- `perlin`, `pyramid`, `highres-pyramid`: more “structured” noise. These can help texture, grain, or large-scale patterning, but they are easier to overdo and can bias the image.
- `laplacian`: heavier-tailed, so it can make things punchier or rougher, but also less stable.

So if you want a rule of thumb:

- For quality/stability tests: start with `gaussian`.
- For SDE/ancestral behavior tests: try `brownian` first.
- For stylized texture experiments: try `perlin` or `highres-pyramid`.
- For low CFG or fragile models: stay with `gaussian` or `brownian`.

Rest of the settings:

- `v-prediction` / `ztsnr`:
  Start conservative. `momentum=0.35-0.55`, `momentum_strategy=cosine`, `c2=0.5`, `ita=0.0-0.08`, `denoise_to_zero=true`.
  Reason: ztsnr already changes end-of-schedule behavior, so piling on too much extra stochasticity often makes the tail less stable than it needs to be.
- `RF` / flow-matching:
  Start even more conservative. `momentum=0.15-0.35`, `momentum_strategy=linear` or `static`, `c2=0.5`, `ita=0.0` first.
  If you want more texture, raise `ita` only a little, like `0.02-0.05`. so I’d treat that as experimental and not push it hard.

A good default mental model is:

- `momentum` controls how much the sampler “leans into” its previous update.
- `c2` controls the midpoint of the 2-stage step. `0.5` is the safe default.
- `ita` controls stochasticity. If `ita=0`, the run is deterministic and `noise_sampler_type` won’t matter.
- `noise_sampler_type` only matters once `ita > 0`.

My practical presets would be:

- For `v-pred + ztsnr`: `momentum 0.45`, `cosine`, `c2 0.5`, `ita 0.00` to start, then `0.03-0.06` if you want a little more liveliness.
- For `RF`: `momentum 0.25`, `linear`, `c2 0.5`, `ita 0.00` to start, then only tiny `ita` increases if the model tolerates it.

Symptoms help a lot too:

- If you get ringing, double edges, or overshoot: lower `momentum`.
- If the image feels too locked-in or sterile: raise `ita` a bit.
- If the end of the sample gets unstable or crunchy: lower `ita` first, especially on ztsnr and RF.
- If behavior changes wildly with small tweaks: go back to `c2=0.5`.

---


### Currently included extra samplers: 
* RES (+ Somewhat naively momentumized and modified, thanks to Kat and Birch-San for the source implementation!)
* DPMPP Dual SDE (+ Somewhat naively momentumized, tis simply DPMPP SDE with an added SDE akin to how 3M samples)
* Clyb 4M SDE (A modified DPMPP 3M SDE, with an added SDE, egotisticalized by yours truly)
* TTM (Thanks to Kat and Birch-San for the source implementation!)
* LCM Custom Noise (Supports different types of noise other than generic gaussian)
* DPMPP 3M SDE with Dynamic ETA (Anneals down towards a minimum eta via a cosine curve)
* Supreme (Many extra functionalities and step methods available)

### Currently included extra K-Sampling nodes:
* SamplerCustomNoise (Supports custom noises other than gaussian noise for init noise)
* SamplerCustomNoiseDuo (Same as above, but with an added High-res fix for simplicity.)
* SamplerCustomModelMixtureDuo (Samples with custom noises, and switches between model1 and model2 every step. If you encounter vram errors, try adding/removing `--disable-smart-memory` when launching ComfyUI)

### Currently included extra Guider nodes:
* GeometricCFGGuider: Samples the two conditionings, then blends between them using a user-chosen alpha.
* ScaledCFGGuider: Samples the two conditionings, then adds it using a method similar to "Add Trained Difference" from merging models.
* ImageAssistedCFGGuider: Samples the conditioning, then adds in the latent image using vector projection onto the CFG. Image latent ought to be of the same size as the diffusion latent.

#### Supreme Sampler features:
* step_method: You have the ability to choose your own step method with this sampler! Optionally, there's a dynamic step method, which chooses the appropriate order based on the calculated error between steps, allowing you to obtain higher quality when it matters in the sampling process. Defaults to **(Euler)**.
* centralization: Subtracts mean from the denoised latent. This can lead to perceptually sharper results, though may change the perceivable brightness of the image. Conservatively defaults to **(0.05)**.
* normalization: Divides the denoised latent by the standard deviation. Can increase contrast in the image, though may hurt fidelity and coherency at high strengths. Conservatively defaults to **(0.05)**.
* edge_enhancement: Sharpens the latent, and then applies a bilateral blur, leaving the edges sharpened. Defaults to **(0.25)**.
* perphist: Adds previous denoised variable to the current denoised using perpendicular vector projection. Default of **(0.5)**, where higher values add the old denoised variable, and negative values subtract the old denoised variable.
* substeps: Amount of times to iterate over each step and average the results. Can be useful for obtaining higher quality at a given step count. Inspired by [ReNoise](https://arxiv.org/pdf/2403.14602v1.pdf). Default of **(2)**.
* noise_modulation: Modulates the noise based on the denoising step or other functionality. Current options with a default of **("intensity")**: ("none", "intensity", "frequency")
* modulation_strength: Strength of the modulation, utilizing a weighted sum between the modulated noise and normal noise. Default of **(2.0)**. Only has an effect when noise_modulation != "none".
* modulation_dims: Chooses where to apply noise modulation. (1) is in the channels only, perceptually the strongest effect. (2) is in the height and width of the noisy tensor, and has the weakest effect (but that doesn't mean worse). (3) is both channels and height n' width. 
* reversible_dampen: Multiplier of the reversible correction in the `reversible_` samplers. Increase for less reversibility, and decrease for more. There be dragons lower than .5 with low steps.

#### Noise Modulation Functionality:
* modulation_dims: Choose between C, HW, or CHW. (In regards to the latent's channels, height + width, or channels + height + width)

* none: No change from normal ancestralness behavior.
* intensity: Scales noise based on the standard deviation of the current noisy tensor, and the intensity value.
* frequency: Scales the high-frequency components of the noise based on the given noisy tensor's standard deviation, and intensity.

#### Supreme Sampler step methods:
* Euler: A simple 1st-order step method.
* DPM_1/2/3S: 1st, 2nd, and 3rd-order samplers of the DPM family of solvers.
* Bogacki Shampine: Denoise using the Bogacki-Shampine method of ODE solvers. 3rd-order sampler.
* RK4/RK45/Adaptive_RK: High-order Runge-Kutta samplers, where RK4 is a 4th-order sampler, RK45 is 6th-order, and Adaptive RK will choose between 4th/3rd/2nd/1st order based on error between previous and current denoised variables.
* Reversible Variants: Utilizes an implementation of the reversible correction heun step method, similar to the one found in [torchsde](https://github.com/google-research/torchsde), and as implemented in [this paper](https://arxiv.org/abs/2105.13493). Reversible_Heun is a 2nd-order sampler, with an experimental Reversible_Heun_1S for a 1st-order alternative method. There is also Reversible_Bogacki_Shampine, which produces even more colorful results. 
* Trapezoidal: A [method to solve ODEs](https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)) derived from the [Trapezoidal Rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) for computing integrals.
* Dynamic: Chooses a sampler based on error. Error is the same methodization as in adaptive_rk. The choices between samplers are as follows, in order from high error to low error: RKF45, RK4, Bogacki_Shampine, Trapezoidal, Euler. May change in the future.

Utilizing custom sampling within ComfyUI is encouraged for these samplers!