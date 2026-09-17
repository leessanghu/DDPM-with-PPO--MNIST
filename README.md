# DDPM with PPO on MNIST

> From implementing a DDPM from scratch to exploring how reinforcement learning changes its denoising trajectory.

## Overview

This project began as a simple implementation exercise to understand **Denoising Diffusion Probabilistic Models (DDPMs)** on MNIST.

After building and training the baseline DDPM, I became interested in a new question:

> **What happens if reinforcement learning is applied to the reverse diffusion process?**

Diffusion generation consists of a sequence of stochastic transitions:

$$x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0$$

This makes it possible to view the reverse diffusion process as a sequential stochastic policy.

Based on this idea, I fine-tuned the pretrained DDPM using **Proximal Policy Optimization (PPO)**.

The project then evolved beyond simply asking whether PPO increases a reward.

I wanted to examine the result from several perspectives:

- Does PPO improve the chosen reward?
- Does PPO change the distribution of generated samples?
- If the final image changes, when do the denoising trajectories begin to diverge?
- How does that difference develop throughout reverse diffusion?
- Can the divergence be observed both visually and quantitatively?

---

# 1. DDPM from Scratch

The first stage of the project was simply to implement and understand a DDPM.

## 1.1 Forward Diffusion

The forward diffusion process gradually adds Gaussian noise to a clean image.

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$

where

$$\epsilon\sim\mathcal{N}(0,I)$$

and $\bar{\alpha}_t$ controls how much of the original image remains at timestep $t$.

As $t$ increases, the image becomes progressively noisier until it approaches Gaussian noise.

---

## 1.2 Noise Prediction

A small U-Net was trained to predict the noise contained in a noisy image.

The model predicts

$$\epsilon_\theta(x_t,t)$$

from the noisy state $x_t$ and timestep $t$.

The training objective is the standard DDPM noise-prediction MSE:

$$L_{\mathrm{DDPM}}=\mathbb{E}\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2\right]$$

The implementation includes:

- linear noise schedule
- forward diffusion
- sinusoidal timestep embedding
- residual blocks
- U-Net noise predictor
- DDPM reverse sampling

The model was intentionally kept small and trained for only **10 epochs**.

The initial purpose was to understand the mechanism rather than maximize MNIST generation performance.

---

## 1.3 DDPM Training

![DDPM Training Loss](assets/ddpm_training_loss.png)

The noise-prediction MSE decreased rapidly during the first few epochs and then gradually stabilized.

The final loss remained around the low `0.02` range.

For this small experiment, the model was sufficiently trained to produce recognizable MNIST-like structures.

---

## 1.4 Baseline Generation

![Baseline DDPM Samples](assets/ddpm_baseline_samples_1.png)

The baseline DDPM was able to generate recognizable digits from Gaussian noise.

Some samples remain ambiguous or distorted, which is expected given the small U-Net and short training schedule.

Another set of generated samples is shown below.

![Additional Baseline DDPM Samples](assets/ddpm_baseline_samples_2.png)

At this point, the project was still simply a **DDPM implementation exercise**.

---

## 1.5 Watching the Denoising Process

One interesting property of diffusion models is that generation can be inspected as a trajectory rather than only through the final image.

![DDPM Denoising Process](assets/ddpm_denoising_process.png)

A sample gradually evolves from noise into a digit:

```text
Gaussian Noise
     │
     ▼
  t = 999
     │
     ▼
  t = 750
     │
     ▼
  t = 500
     │
     ▼
  t = 250
     │
     ▼
  t = 100
     │
     ▼
  t = 0
     │
     ▼
Generated Digit
```

This sequential structure led to the next question:

> **If generation itself is a stochastic trajectory, can reinforcement learning be applied to it?**

---

# 2. Viewing Reverse Diffusion as a Policy

The DDPM reverse process is

$$x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0$$

Each reverse transition can be described by a Gaussian distribution:

$$p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta(x_t,t),\sigma_t^2I)$$

This provides a natural RL interpretation.

```text
State
 x_t
  │
  ▼
DDPM Policy
  │
  ▼
pθ(x_{t-1} | x_t)
  │
  ▼
Next State
x_{t-1}
  │
  ▼
 ...
  │
  ▼
Final Image
 x_0
  │
  ▼
Reward
```

The components can be interpreted as:

```text
State       : x_t
Action      : x_{t-1}
Policy      : pθ(x_{t-1} | x_t)
Trajectory  : x_T → ... → x_0
Reward      : R(x_0)
```

Instead of viewing the DDPM only as a noise predictor, I treated its reverse transition distribution as a **stochastic policy**.

PPO was then used to fine-tune this pretrained policy.

---

# 3. Reward Model

A separate CNN classifier was trained on MNIST.

The classifier reached approximately:

```text
Test Accuracy = 98.96%
```

For a generated image $x_0$, the reward was defined using the maximum predicted class probability.

$$R(x_0)=\max_k P(y=k\mid x_0)$$

In other words, the diffusion model receives a high reward when the classifier is confident that the generated image belongs to one of the MNIST classes.

The reward was intentionally simple.

The goal was not to design an optimal perceptual reward model.

Instead, I wanted to observe:

> **What happens to the diffusion process when a terminal reward is introduced through PPO?**

---

# 4. PPO Formulation

For each reverse transition, the Gaussian transition probability is interpreted as the policy probability.

The PPO probability ratio compares the probability of the same transition under the updated policy and the old policy.

The log-probability under the updated policy is

$$\log p_\theta(x_{t-1}\mid x_t)$$

The log-probability under the old policy is

$$\log p_{\theta_{\mathrm{old}}}(x_{t-1}\mid x_t)$$

The difference between the two log-probabilities is

$$\log p_\theta(x_{t-1}\mid x_t)-\log p_{\theta_{\mathrm{old}}}(x_{t-1}\mid x_t)$$

The PPO probability ratio is

$$r_t(\theta)=\exp\left(\log p_\theta(x_{t-1}\mid x_t)-\log p_{\theta_{\mathrm{old}}}(x_{t-1}\mid x_t)\right)$$

The unclipped policy objective is

$$r_t(\theta)A_t$$

The clipped probability ratio is

$$\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)$$

The clipped policy objective is

$$\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t$$

Finally, the PPO objective is

$$L_{\mathrm{PPO}}=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]$$

---

## 4.1 Value Network

A separate value network estimates the expected terminal reward from an intermediate diffusion state.

$$V_\phi(x_t,t)$$

The target can be interpreted as

$$V_\phi(x_t,t)\approx\mathbb{E}[R(x_0)\mid x_t]$$

For this experiment, I used the simplified advantage

$$A_t=R(x_0)-V_\phi(x_t,t)$$

This is a lightweight terminal-reward formulation rather than a full GAE-based PPO implementation.

---

# 5. PPO Experiment Setup

The pretrained DDPM was used as the initial PPO policy.

The overall experiment can be summarized as:

```text
Pretrained DDPM
      │
      ▼
Sample Reverse Trajectory
      │
      ▼
x_T → x_{T-1} → ... → x_0
      │
      ▼
CNN Reward Model
      │
      ▼
R(x_0)
      │
      ├───────────────┐
      ▼               ▼
  Advantage       Value Loss
      │
      ▼
PPO Policy Update
      │
      ▼
Updated DDPM
```

During trajectory collection, selected reverse-process transitions were stored together with their old policy log-probabilities.

The transition probability was then recomputed under the updated model to construct the PPO probability ratio.

The purpose of this setup was not to build a highly optimized diffusion-RL system, but to create a simple environment in which changes to the reverse diffusion trajectory could be observed directly.

---

# 6. PPO Training

![PPO Training](assets/ppo_training.png)

The blue curve represents the reward from newly sampled training trajectories.

It fluctuates considerably because each PPO iteration samples new stochastic reverse trajectories from a relatively small batch.

The orange curve represents evaluation using fixed initial noise.

This curve is substantially more stable and frequently lies above the baseline reward.

However, the reward does **not** continuously increase with PPO iterations.

Therefore, this experiment should not be interpreted simply as:

> **PPO continuously improves DDPM generation quality.**

Instead, PPO changed the diffusion policy under the chosen classifier-confidence objective, and some improvement was observed under that specific reward.

This raised another question:

> **Did PPO only improve the reward, or did it change the generative behavior itself?**

---

# 7. Evaluation on 1,000 Samples

To obtain a broader comparison, I generated **1,000 samples** from both the baseline and PPO models.

The classifier-confidence rewards were:

| Model | Mean Reward | Std. Reward |
|---|---:|---:|
| Baseline DDPM | 0.9154 | 0.1475 |
| PPO-DDPM | **0.9203** | **0.1461** |

The reward difference is

$$\Delta R=0.9203-0.9154$$

which gives approximately

$$\Delta R\approx0.0049$$

The optimized reward therefore improved, but only modestly.

The next step was to examine whether the distribution of generated classes had also changed.

---

# 8. Generated Class Distribution

![Generated Class Distribution](assets/class_distribution.png)

The predicted class counts for the 1,000 generated samples were:

| Digit | Baseline | PPO |
|---:|---:|---:|
| 0 | 50 | 17 |
| 1 | 73 | **211** |
| 2 | 148 | 93 |
| 3 | 125 | 73 |
| 4 | 103 | 127 |
| 5 | 135 | 110 |
| 6 | 113 | 89 |
| 7 | 99 | 123 |
| 8 | 76 | 61 |
| 9 | 78 | 96 |

The largest change occurred for digit `1`.

$$73\rightarrow211$$

Its proportion increased from

$$7.3\%\rightarrow21.1\%$$

This distributional change is much larger than the change in average reward.

The observed behavior is therefore closer to:

```text
                    PPO
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
Mean classifier          Generated class
reward ↑ slightly        distribution shifts
```

The reward does not contain an explicit objective for:

- class balance
- sample diversity
- preservation of the original DDPM distribution

Therefore, optimizing classifier confidence can alter the generated distribution.

One possible hypothesis is that PPO found image structures that provide an easier route toward high classifier confidence.

However, this experiment does **not** establish why digit `1` specifically became more frequent.

---

# 9. Looking Beyond the Final Image

At this point, the project became less about the final reward and more about the diffusion trajectory itself.

A diffusion model does not generate an image in a single step.

Its output is produced through

$$x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0$$

If PPO can change the final generated class, then a natural question is:

> **When and how do the baseline and PPO trajectories begin to diverge?**

Simply comparing final images cannot answer this.

So I constructed a controlled trajectory experiment.

---

# 10. Why Fixing Only the Initial Noise Is Not Enough

A DDPM reverse step is stochastic.

$$x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz_t$$

where

$$z_t\sim\mathcal{N}(0,I)$$

Suppose the baseline and PPO models start from exactly the same initial noise.

$$x_T^{\mathrm{Base}}=x_T^{\mathrm{PPO}}$$

This alone is not enough for a controlled comparison.

If the two models receive different reverse noise,

$$z_t^{\mathrm{Base}}\neq z_t^{\mathrm{PPO}}$$

then their trajectories may diverge simply because of stochastic sampling.

Therefore, both sources of randomness need to be controlled.

---

# 11. Controlled Trajectory Experiment

For the trajectory comparison, I fixed:

1. the initial Gaussian noise
2. the reverse-process noise at every timestep

The initial state satisfies

$$x_T^{\mathrm{Base}}=x_T^{\mathrm{PPO}}$$

and every reverse noise satisfies

$$z_t^{\mathrm{Base}}=z_t^{\mathrm{PPO}}$$

for all timesteps.

Conceptually:

```text
                  SAME x_T
                     │
          ┌──────────┴──────────┐
          │                     │
      Baseline                PPO
          │                     │
          └──── SAME z_t ───────┘
          │                     │
          ▼                     ▼
       x_{t-1}^B             x_{t-1}^P
          │                     │
          └─── SAME z_{t-1} ────┘
          │                     │
          ▼                     ▼
         ...                   ...
          │                     │
          ▼                     ▼
        x_0^B                 x_0^P
```

The fixed random variables used for this experiment are stored in:

```text
initial_noise.pt
trajectory_reverse_noises.pt
```

Under these matched stochastic conditions, the difference between the trajectories is no longer caused by independently sampled reverse noise.

Instead, the comparison focuses on differences introduced by the changed denoising model and their subsequent propagation.

---

# 12. Same Noise, Different Final Generation

One controlled sample produced a particularly interesting result.

![Full Denoising Trajectory](assets/full_trajectory.png)

Both models begin from the same initial noise and receive the same reverse-process noise at every timestep.

Yet their final outputs are different:

```text
Baseline DDPM  →  5
PPO-DDPM       →  9
```

At high-noise timesteps, the trajectories are visually difficult to distinguish.

As denoising progresses, different structures gradually emerge.

This sample was selected for more detailed trajectory analysis.

Importantly, this is a **single selected example** and should not be interpreted as evidence that every PPO trajectory behaves in the same way.

---

# 13. Detailed Denoising Trajectory

![Detailed Denoising Trajectory](assets/detailed_trajectory.png)

The later part of the trajectory was saved at finer intervals.

$$t=400,375,350,\ldots,25,0$$

At approximately $t=250$ to $t=200$, the states remain noisy, but structural differences become increasingly visible.

By approximately $t=150$ to $t=100$, the baseline and PPO trajectories visually resemble different digit structures much more clearly.

However, visual inspection alone cannot identify an exact semantic divergence timestep.

So I examined an earlier interval more closely.

---

# 14. Zooming Into the Noisy Region

![Trajectory Zoom](assets/trajectory_zoom.png)

The interval from

$$t=400$$

to

$$t=250$$

was inspected at smaller timestep intervals.

Directly comparing the raw $x_t$ states remained difficult.

Both trajectories still contain substantial noise, which can hide relatively small model-dependent differences.

This motivated a different question:

> Instead of asking what each noisy state looks like, **where are the two trajectories actually different?**

---

# 15. Difference Maps

For each selected timestep, I calculated the absolute pixel-space difference.

$$\Delta_t=\left|x_t^{\mathrm{PPO}}-x_t^{\mathrm{Baseline}}\right|$$

![Trajectory Difference Auto Scale](assets/trajectory_difference_autoscale.png)

The difference maps reveal spatial structure that is difficult to identify in the raw noisy trajectories.

However, the first visualization introduced an important visualization issue.

Each subplot was automatically normalized using its own value range.

This means that a small difference at an early timestep can appear visually as bright as a much larger difference at a later timestep.

Therefore, the auto-scaled visualization is useful for identifying **where** differences occur, but not for comparing their absolute magnitude across time.

---

# 16. Shared-Scale Difference Map

To compare difference magnitude across timesteps, I replotted every heatmap using a single shared color scale.

![Trajectory Difference Shared Scale](assets/trajectory_difference_shared.png)

The temporal pattern becomes much clearer.

```text
t = 400
weak but spatially structured difference
              │
              ▼
t = 350–300
difference gradually strengthens
              │
              ▼
t = 275–250
structured difference becomes clearer
              │
              ▼
t = 225–175
difference strongly amplifies
              │
              ▼
t = 150–100
large stroke-shaped differences
```

The difference does not suddenly appear at one obvious timestep.

Instead, the pixel-space difference appears to be progressively amplified throughout later denoising.

Importantly, a spatial difference at $t=400$ does **not** mean that the semantic `5` versus `9` decision has already been made.

The heatmap measures pixel-space differences, not semantic identity.

---

# 17. Quantifying Trajectory Divergence

Visualizations suggest that the difference grows during denoising.

To quantify this observation, I calculated the pixel-space MSE between the two trajectory states.

$$D_t=\mathrm{MSE}(x_t^{\mathrm{PPO}},x_t^{\mathrm{Baseline}})$$

![Trajectory Divergence](assets/trajectory_mse.png)

For the selected trajectory:

| Timestep | Baseline–PPO MSE |
|---:|---:|
| 400 | ~0.011 |
| 375 | ~0.014 |
| 350 | ~0.020 |
| 325 | ~0.026 |
| 300 | ~0.036 |
| 275 | ~0.048 |
| 250 | ~0.066 |
| 225 | ~0.085 |
| 200 | ~0.107 |
| 175 | ~0.135 |
| 150 | ~0.162 |
| 125 | ~0.191 |
| 100 | ~0.220 |

The curve increases smoothly rather than exhibiting one obvious discontinuity.

For this sample, the observed behavior is therefore closer to:

```text
Small Difference
       │
       ▼
Accumulation
       │
       ▼
Amplification
       │
       ▼
Different Final Structure
```

Pixel-space MSE alone does not tell us when the semantic identity of the sample changes.

It only measures how far apart the two intermediate states become.

---

# 18. Why Can a Small Difference Grow?

The reverse-process mean depends on the model's predicted noise.

Suppose PPO introduces a difference in the predicted noise:

$$\Delta\epsilon_t=\epsilon_{\theta_{\mathrm{PPO}}}(x_t,t)-\epsilon_{\theta_{\mathrm{Base}}}(x_t,t)$$

This changes the next reverse state:

$$\Delta\epsilon_t\rightarrow\Delta x_{t-1}$$

At the next timestep, the two models are now operating on different states.

That can produce another difference in their noise predictions:

$$\Delta x_{t-1}\rightarrow\Delta\epsilon_{t-1}$$

The process can continue recursively:

```text
Δε_t
 │
 ▼
Δx_{t-1}
 │
 ▼
Δε_{t-1}
 │
 ▼
Δx_{t-2}
 │
 ▼
...
 │
 ▼
Δx_0
```

This provides one possible interpretation of the smooth increase observed in the trajectory MSE.

A small difference in the denoising policy affects one reverse step.

That changes the state used by the following step, allowing the difference to propagate and accumulate through the trajectory.

---

# 19. What Did PPO Actually Change?

The experiment begins with a terminal reward:

$$R(x_0)$$

But PPO does not directly modify only the final image.

It updates the transition policy:

$$p_\theta(x_{t-1}\mid x_t)$$

throughout the reverse process.

Conceptually:

```text
Final Reward
     │
     ▼
 PPO Update
     │
     ▼
Reverse Transition Policy Changes
     │
     ▼
Trajectory Changes
     │
     ▼
Future States Change
     │
     ▼
Future Noise Predictions Change
     │
     ▼
Final Generation Changes
```

This helps explain why a relatively small change in average reward can coexist with a much larger change in generative behavior.

---

# 20. Main Observations

This exploratory experiment produced several observations.

## 20.1 Reward

The mean classifier-confidence reward changed from

$$0.9154\rightarrow0.9203$$

This is a modest improvement.

---

## 20.2 Generated Distribution

The generated frequency of digit `1` changed from

$$73\rightarrow211$$

or

$$7.3\%\rightarrow21.1\%$$

The generated distribution therefore changed substantially despite the relatively small reward increase.

---

## 20.3 Controlled Trajectory

Under matched initial noise and matched reverse-process noise, one selected sample produced:

```text
Baseline DDPM  →  5
PPO-DDPM       →  9
```

This shows that the PPO-updated model can follow a different reverse trajectory even when the stochastic sampling conditions are matched.

---

## 20.4 Progressive Pixel-Space Divergence

For the selected trajectory, the baseline–PPO MSE increased from approximately

$$0.011$$

at $t=400$ to approximately

$$0.220$$

at $t=100$.

The divergence increased smoothly rather than appearing as one obvious abrupt transition.

---

## 20.5 Overall Observation

The experiment suggests an important distinction:

```text
Reward Change
     ≠
Complete Description
of Model Behavior
```

A small improvement in the optimized scalar reward can coexist with:

- a substantial distribution shift
- altered reverse trajectories
- progressively amplified intermediate-state differences

Therefore, examining only the final reward may hide important changes in generative behavior.

---

# 21. Limitations

This is an exploratory MNIST-scale experiment rather than a general result about reinforcement learning for diffusion models.

## 21.1 Simple Reward

The reward is classifier confidence:

$$R(x_0)=\max_kP(y=k\mid x_0)$$

It does not directly measure:

- perceptual quality
- diversity
- class balance
- similarity to the baseline distribution

A higher reward therefore does not necessarily mean that the generated images are globally better.

---

## 21.2 Reward-Model Bias

Digit `1` became substantially more frequent after PPO.

A possible explanation is that certain structures are easier for the classifier to recognize with high confidence.

However, this was **not directly tested**.

The current experiment cannot determine whether the shift is caused by:

- classifier calibration
- digit complexity
- the original DDPM distribution
- PPO optimization dynamics
- interactions between these factors

---

## 21.3 Simplified PPO

The PPO implementation is intentionally lightweight.

The value network estimates the expected terminal reward, and the advantage is defined as

$$A_t=R(x_0)-V_\phi(x_t,t)$$

rather than using a full GAE-based formulation.

The experiment should therefore be interpreted as an exploration of PPO-style fine-tuning rather than a complete diffusion-specific RL framework.

---

## 21.4 MNIST-Scale Experiment

MNIST and the small U-Net make the denoising process easy to visualize.

However, observations from this setting cannot automatically be generalized to large text-to-image diffusion models or more complex diffusion architectures.

---

## 21.5 Pixel Distance Is Not Semantic Distance

The difference maps and MSE measure pixel-space divergence.

$$D_t=\mathrm{MSE}(x_t^{\mathrm{PPO}},x_t^{\mathrm{Baseline}})$$

They do not determine exactly when the semantic interpretation changes from `5` to `9`.

---

## 21.6 Selected Trajectory

The detailed trajectory analysis focuses on one interesting controlled example.

A single trajectory cannot establish whether progressive divergence is a general behavior of PPO-fine-tuned diffusion models.

A larger controlled trajectory analysis is required.

---

# 22. Next Steps

The trajectory experiment suggests several directions for further analysis.

## 22.1 Predicted Clean Image at Each Timestep

Instead of inspecting only the noisy state $x_t$, the model's predicted clean image can be estimated at each timestep.

$$\hat{x}_0^{(t)}=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$$

This changes the question from:

> When do the noisy states become different?

to:

> **When do the models begin predicting different final clean structures?**

Conceptually:

```text
t        Baseline x̂₀        PPO x̂₀

400           ?                 ?
350           ?                 ?
300           ?                 ?
250        5-like?           9-like?
200           5                 9
150           5                 9
100           5                 9
...
0             5                 9
```

This could help separate **pixel-space divergence** from **semantic divergence**.

---

## 22.2 Analyze Many Controlled Trajectories

The controlled experiment can be repeated across many initial noises.

Possible measurements include:

- trajectory MSE
- final class-change frequency
- semantic divergence timestep
- reward difference
- feature-space trajectory distance
- relationship between trajectory divergence and final class change

This would help determine whether the pattern observed in the selected sample also appears across many generations.

---

## 22.3 Feature-Space Trajectory Analysis

Pixel-space MSE may not capture semantic changes well.

Intermediate states could instead be projected into a learned feature space.

For example:

```text
x_t
 │
 ▼
Feature Encoder
 │
 ▼
h_t
```

Then the baseline and PPO trajectories could be compared using feature-space distance rather than only raw pixels.

This may provide a better view of when semantic structures begin to diverge.

---

## 22.4 Better Reward Design

The current reward optimizes only classifier confidence.

A more constrained reward could include additional objectives:

$$R=R_{\mathrm{confidence}}+\lambda_1R_{\mathrm{diversity}}+\lambda_2R_{\mathrm{distribution}}$$

This could test whether PPO can improve a desired reward while preserving more of the original generative distribution.

---

## 22.5 Beyond MNIST

The same trajectory-oriented analysis could eventually be applied to more complex datasets and architectures.

With richer images, trajectory differences could be studied through:

- local structures
- object parts
- visual attributes
- semantic features
- feature-space representations

This would make it possible to ask not only whether two trajectories diverge, but also **which visual concepts diverge first**.

---

# 23. Repository Structure

```text
DDPM-with-PPO--MNIST/
│
├── DDPM.ipynb
├── README.md
│
├── assets/
│   ├── ddpm_training_loss.png
│   ├── ddpm_baseline_samples_1.png
│   ├── ddpm_baseline_samples_2.png
│   ├── ddpm_denoising_process.png
│   ├── ppo_training.png
│   ├── class_distribution.png
│   ├── full_trajectory.png
│   ├── detailed_trajectory.png
│   ├── trajectory_zoom.png
│   ├── trajectory_difference_autoscale.png
│   ├── trajectory_difference_shared.png
│   └── trajectory_mse.png
│
├── ddpm_baseline.pth
├── mnist_reward_model.pth
├── baseline_samples.pt
├── initial_noise.pt
├── trajectory_reverse_noises.pt
├── baseline_trajectory.pt
├── ppo_trajectory.pt
├── baseline_detailed_trajectory.pt
└── ppo_detailed_trajectory.pt
```

---

# 24. Project Progression

The project evolved through the following sequence:

```text
Implement DDPM from Scratch
            │
            ▼
Understand Reverse Diffusion
            │
            ▼
View Denoising as a Trajectory
            │
            ▼
Introduce PPO
            │
            ▼
Measure Reward
            │
            ▼
Observe Distribution Shift
            │
            ▼
Control Sampling Randomness
            │
            ▼
Compare Baseline vs PPO Trajectories
            │
            ▼
Visualize Difference Maps
            │
            ▼
Measure Trajectory Divergence
```

What began as a simple DDPM implementation therefore developed into an exploration of how reinforcement learning can affect the internal generation process of a diffusion model.

---

# Final Takeaway

This project started with a simple goal:

> **Implement DDPM from scratch and understand how diffusion generation works.**

After observing the sequential nature of reverse diffusion, I explored whether the process could be treated as an RL trajectory and fine-tuned with PPO.

The PPO experiment produced a modest increase in classifier-confidence reward:

```text
Baseline Mean Reward : 0.9154
PPO Mean Reward      : 0.9203
```

But the reward alone did not describe the full change.

The generated class distribution shifted substantially, particularly for digit `1`.

This motivated a controlled comparison of the denoising trajectories.

By fixing both the initial noise and the reverse-process noise, I compared the baseline and PPO models under matched stochastic conditions.

For one selected sample:

```text
Same Initial Noise
        +
Same Reverse Noise
        │
        ▼
┌───────────────────────┐
│                       │
▼                       ▼
Baseline               PPO
│                       │
▼                       ▼
5                       9
```

Difference maps and trajectory MSE showed that the two trajectories did not separate at one obvious timestep.

Instead, a relatively small early difference progressively increased during later denoising.

This changed the main question of the project from:

> **Does PPO increase the reward?**

to:

> **How does a reward-driven policy update reshape the denoising trajectory itself?**

The current experiment does not provide a general answer to that question, but it provides a controlled setup for investigating it.

The next step is to move beyond pixel-space divergence and examine **when semantic predictions along the trajectory begin to change**.
