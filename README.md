# DDPM with PPO on MNIST

> From a simple DDPM implementation to exploring how reinforcement learning changes the denoising trajectory.

This project started as a small practice project to understand **DDPM (Denoising Diffusion Probabilistic Models)** by implementing one from scratch on MNIST.

After training the baseline model, I became interested in a different question:

> **What happens if reinforcement learning is applied to the reverse diffusion process?**

Since diffusion generation can be viewed as a sequence of stochastic transitions

\[
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0,
\]

I experimented with treating this denoising process as a policy trajectory and fine-tuning it using **Proximal Policy Optimization (PPO)**.

The goal eventually became broader than simply checking whether PPO increased a reward.

I wanted to examine the result from several perspectives:

- Does PPO actually change the reward?
- Does it change the distribution of generated samples?
- If the final image changes, **when and how does the denoising trajectory begin to diverge?**
- Can that divergence be observed both visually and quantitatively?

---

# 1. Starting Point — DDPM from Scratch

The first stage was simply to implement and understand a DDPM.

For the forward diffusion process,

\[
x_t =
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
\]

A small U-Net was trained to predict the added noise,

\[
\epsilon_\theta(x_t,t),
\]

using the standard noise-prediction objective

\[
\mathcal{L}_{DDPM}
=
\mathbb{E}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right].
\]

The implementation includes:

- linear noise schedule
- forward diffusion
- sinusoidal timestep embedding
- residual blocks
- U-Net noise predictor
- DDPM reverse sampling

The model was intentionally kept small and trained for only **10 epochs**, since the initial purpose was to understand the mechanism rather than maximize MNIST generation performance.

---

## DDPM Training

![DDPM Training Loss](assets/ddpm_training_loss.png)

The noise-prediction MSE decreased rapidly during the first few epochs and then gradually stabilized.

The final loss remained around the low `0.02` range.

This was sufficient for the model to generate recognizable MNIST-like structures, although some samples were still ambiguous or distorted.

---

## Baseline Generation

![Baseline DDPM Samples](assets/ddpm_baseline_samples.png)

After training, the DDPM was able to generate recognizable digits from Gaussian noise.

The samples were not perfectly clean, which was expected given the small network and short training schedule.

At this point, the project was still simply a **DDPM implementation exercise**.

---

## Watching the Denoising Process

One reason diffusion models are interesting is that generation can be inspected as a trajectory rather than only as a final output.

![DDPM Denoising Process](assets/ddpm_denoising_process.png)

For example,

```text
t = 999
   ↓
t = 750
   ↓
t = 500
   ↓
t = 250
   ↓
t = 100
   ↓
t = 0
```

shows noise gradually developing into a digit structure.

This observation led to the next question:

> If generation is a sequential trajectory, could reinforcement learning be applied to that trajectory?

---

# 2. Adding Reinforcement Learning

The DDPM reverse process is

\[
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_0.
\]

Each reverse transition can be written as

\[
p_\theta(x_{t-1}|x_t)
=
\mathcal{N}
\left(
\mu_\theta(x_t,t),
\sigma_t^2I
\right).
\]

This suggests the following interpretation:

```text
State
 x_t
  │
  ▼
DDPM / Policy
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

Instead of viewing the DDPM only as a noise predictor, I treated the reverse diffusion model as a **stochastic policy**.

PPO was then used to fine-tune this policy.

---

# 3. Reward Model

A separate CNN classifier was trained on MNIST and reached approximately

```text
Test Accuracy = 98.96%
```

For a generated image \(x_0\), I defined the reward as

\[
R(x_0)
=
\max_k P(y=k|x_0).
\]

In other words, the diffusion model receives a high reward when the classifier is confident that the generated image belongs to one of the MNIST classes.

This reward was deliberately simple.

The experiment was not designed to build an optimal reward model, but to see **what happens to the diffusion process when a terminal reward is introduced through PPO**.

---

# 4. PPO Setup

For each reverse transition, the Gaussian transition probability was used as the policy probability.

The PPO ratio is

\[
r_t(\theta)
=
\exp
\left[
\log p_\theta(x_{t-1}|x_t)
-
\log p_{\theta_{\text{old}}}(x_{t-1}|x_t)
\right].
\]

The clipped objective is

\[
\mathcal{L}_{PPO}
=
\mathbb{E}
\left[
\min
\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

A small value network was also trained to estimate

\[
V_\phi(x_t,t)
\approx
\mathbb{E}[R(x_0)|x_t],
\]

with a simplified advantage

\[
A_t = R(x_0)-V_\phi(x_t,t).
\]

The PPO experiment used the pretrained DDPM as the starting policy rather than training a diffusion model from scratch with RL.

---

# 5. PPO Training

![PPO Training](assets/ppo_training.png)

The blue curve represents the reward from newly sampled training trajectories.

It fluctuates considerably because each PPO iteration samples new stochastic reverse trajectories from a relatively small batch.

The orange curve evaluates the current model using fixed initial noise.

This evaluation is considerably more stable and frequently lies above the baseline reward.

However, the reward does **not** continuously increase with PPO iterations.

Therefore, I do not interpret this experiment as simply:

> "PPO continuously improves DDPM generation."

Instead, PPO clearly changed the policy, and some improvement was observed under the chosen classifier-confidence reward.

The next question was whether this improvement represented the **whole story**.

---

# 6. Evaluating 1,000 Generated Samples

To look beyond the small fixed evaluation set, I generated **1,000 samples** from the baseline and PPO models.

The average rewards were:

| Model | Mean Reward | Std. Reward |
|---|---:|---:|
| Baseline DDPM | 0.9154 | 0.1475 |
| PPO-DDPM | **0.9203** | **0.1461** |

The mean reward increased by approximately

\[
+0.0049.
\]

So the optimized reward improved, but only modestly.

Then I examined the predicted classes of the generated images.

---

# 7. PPO Changed the Generated Distribution

![Generated Class Distribution](assets/class_distribution.png)

The class counts were:

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

The most striking change occurred for digit `1`:

\[
73 \rightarrow 211.
\]

Its proportion increased from

\[
7.3\% \rightarrow 21.1\%.
\]

This was much larger than the change in average reward.

So the result was not simply:

```text
PPO
 ↓
slightly better images
```

Instead, what I observed was closer to:

```text
PPO
 ├── Mean classifier reward ↑ slightly
 │
 └── Generated class distribution changes substantially
```

The reward contains no explicit objective for maintaining class balance or generation diversity.

Therefore, maximizing classifier confidence can alter the generative distribution.

One possible explanation is that PPO found particular structures that provide an easier path toward high classifier confidence.

However, this experiment does **not** establish why digit `1` specifically became more frequent.

---

# 8. Looking Beyond the Final Image

At this point, I became more interested in the diffusion process itself.

If PPO can change the final generated class, then:

> **How does that difference develop through the denoising trajectory?**

A naive comparison would be to give both models the same initial noise \(x_T\).

But DDPM sampling is stochastic:

\[
x_{t-1}
=
\mu_\theta(x_t,t)
+
\sigma_tz_t,
\qquad
z_t\sim\mathcal{N}(0,I).
\]

Therefore,

\[
x_T^{Base}=x_T^{PPO}
\]

alone does **not** guarantee a controlled comparison.

Different \(z_t\) values could create different trajectories even without PPO.

---

# 9. Controlled Trajectory Experiment

To isolate the model difference as much as possible, I fixed both:

### 1. Initial noise

\[
x_T^{Base}=x_T^{PPO}
\]

### 2. Reverse-process noise at every timestep

\[
z_t^{Base}=z_t^{PPO}
\qquad
\forall t.
\]

The comparison therefore becomes:

```text
                 SAME x_T
                    │
         ┌──────────┴──────────┐
         │                     │
     Baseline                PPO
         │                     │
         └──── SAME z_999 ─────┘
         │                     │
         ▼                     ▼
     x_998^B                x_998^P
         │                     │
         └──── SAME z_998 ─────┘
         │                     │
         ▼                     ▼
        ...                   ...
         │                     │
         ▼                     ▼
       x_0^B                 x_0^P
```

The repository includes the fixed random variables used for this experiment:

```text
initial_noise.pt
trajectory_reverse_noises.pt
```

Under this setup, stochastic sampling noise is matched between the two runs.

This allowed me to focus on differences introduced by the changed model and their subsequent propagation through the trajectory.

---

# 10. Same Noise, Different Generation

The controlled experiment revealed a particularly interesting sample.

![Full Denoising Trajectory](assets/full_trajectory.png)

Both trajectories begin under the same stochastic conditions.

Yet:

```text
Baseline DDPM → 5
PPO-DDPM      → 9
```

At high-noise timesteps, the difference is extremely difficult to interpret visually.

As denoising proceeds, however, the trajectories gradually develop different structures.

The baseline trajectory eventually forms a `5`, while the PPO trajectory develops into a `9`.

This became the sample used for the detailed trajectory analysis.

---

# 11. Detailed Denoising Trajectory

I first inspected the later denoising process more closely.

![Detailed Denoising Trajectory](assets/detailed_trajectory.png)

The trajectory was saved every 25 timesteps from approximately

\[
t=400 \rightarrow 0.
\]

At roughly \(t=250\sim200\), the states are still noisy, but structural differences become increasingly visible.

By around \(t=150\sim100\), the baseline and PPO trajectories visually resemble different digit structures much more clearly.

However, this observation alone does **not** identify an exact semantic transition timestep.

So I zoomed in further.

---

# 12. Zooming Into \(t=400\rightarrow250\)

![Trajectory Zoom](assets/trajectory_zoom.png)

The interval

\[
t=400 \rightarrow 250
\]

was inspected at steps of 10.

But directly comparing the raw \(x_t\) images was difficult.

The diffusion states still contain substantial noise, which can hide relatively small model-dependent differences.

So instead of asking

> "Can I visually recognize different digits yet?"

I changed the analysis to

> **"Where are the two trajectories actually different?"**

---

# 13. Difference Maps

For every selected timestep, I calculated

\[
\Delta_t
=
\left|
x_t^{PPO}
-
x_t^{Baseline}
\right|.
\]

![Trajectory Difference Auto Scale](assets/trajectory_difference_autoscale.png)

The difference maps immediately revealed spatial structure.

However, there was a visualization problem.

If every subplot is automatically normalized to its own range, a very small difference at \(t=400\) can appear almost as bright as a much larger difference at \(t=100\).

That makes it difficult to compare the **magnitude** of the difference across time.

---

# 14. Shared-Scale Difference Map

To correct this, all heatmaps were plotted using the same global color scale.

![Trajectory Difference Shared Scale](assets/trajectory_difference_shared.png)

This changes the interpretation considerably.

At \(t=400\), the difference is spatially structured but still relatively weak.

As reverse diffusion proceeds:

```text
t = 400
small structured difference
        │
        ▼
t = 350 ~ 300
difference gradually strengthens
        │
        ▼
t = 275 ~ 250
structure becomes clearer
        │
        ▼
t = 225 ~ 175
difference strongly amplifies
        │
        ▼
t = 150 ~ 100
large stroke-shaped differences
```

The difference does not suddenly appear at one timestep.

Instead, it appears to be **progressively amplified through denoising**.

Importantly, the presence of a spatial difference at \(t=400\) does not mean that the semantic `5` vs `9` decision has already been made.

This visualization measures pixel-space differences, not semantic identity.

---

# 15. Quantifying Trajectory Divergence

Finally, I measured the pixel-space MSE between the baseline and PPO trajectories:

\[
D_t
=
\operatorname{MSE}
\left(
x_t^{PPO},
x_t^{Baseline}
\right).
\]

![Trajectory Divergence](assets/trajectory_mse.png)

For the analyzed sample:

| t | MSE |
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

The curve grows smoothly rather than showing one obvious discontinuity.

This suggests a process closer to:

\[
\text{small difference}
\rightarrow
\text{accumulation}
\rightarrow
\text{amplification}
\rightarrow
\text{different final structure}.
\]

---

# 16. Why Does the Difference Grow?

Suppose PPO slightly changes the model's predicted noise:

\[
\Delta\epsilon_t
=
\epsilon_{\theta_{PPO}}(x_t,t)
-
\epsilon_{\theta_{Base}}(x_t,t).
\]

That changes the reverse-process mean.

Therefore,

\[
\Delta\epsilon_t
\rightarrow
\Delta x_{t-1}.
\]

At the next timestep, the two models are now operating on slightly different states.

This can create another difference in their noise predictions:

\[
\Delta x_{t-1}
\rightarrow
\Delta\epsilon_{t-1}.
\]

The effect can therefore propagate:

\[
\boxed{
\Delta\epsilon_t
\rightarrow
\Delta x_{t-1}
\rightarrow
\Delta\epsilon_{t-1}
\rightarrow
\Delta x_{t-2}
\rightarrow
\cdots
\rightarrow
\Delta x_0
}
\]

This provides one way of interpreting the smooth increase observed in the MSE curve.

A relatively small policy modification can affect a reverse step, which changes the state presented to the model at the following step, allowing the difference to accumulate through the recursive denoising process.

---

# 17. What I Learned From the Experiment

The initial question was simple:

> **Can PPO be added to a DDPM?**

But the experiment produced a more interesting set of observations.

### Reward

PPO slightly increased the classifier-confidence reward:

\[
0.9154 \rightarrow 0.9203.
\]

### Distribution

At the same time, the generated class distribution changed substantially.

In particular:

\[
P(\text{digit}=1):
7.3\% \rightarrow 21.1\%.
\]

### Trajectory

Even under matched initial and reverse-process noise, the baseline and PPO models could converge to different digits.

### Divergence

The difference between their trajectories did not appear as one sudden event.

Instead, pixel-space divergence increased progressively as denoising proceeded.

So:

\[
\boxed{
\text{small reward change}
\not\Rightarrow
\text{small generative behavior change}
}
\]

and

\[
\boxed{
\text{reward improvement}
\neq
\text{distribution preservation}.
}
\]

---

# 18. Limitations

This project is an exploratory MNIST-scale experiment, and several limitations are important.

### Simple reward

The reward is only the maximum confidence of an MNIST classifier.

\[
R(x_0)=\max_kP(y=k|x_0).
\]

It does not explicitly measure:

- perceptual quality
- diversity
- class balance
- similarity to the original DDPM distribution

Therefore, reward optimization can create unintended distribution shifts.

### Reward model bias

The strong increase in digit `1` suggests that the reward may favor some generated structures over others.

However, this experiment does not determine whether this is caused by classifier calibration, digit complexity, the DDPM itself, PPO optimization, or another factor.

### Small dataset and architecture

MNIST and the small U-Net make the experiment easy to inspect, but the result cannot automatically be generalized to large-scale diffusion models.

### Simplified PPO

The PPO implementation is intentionally lightweight.

The value network estimates terminal reward from intermediate diffusion states, and the experiment does not use a full diffusion-specific RL framework.

### Pixel MSE is not semantic distance

The trajectory MSE tells us

> how different the two states are,

but not

> whether the models already represent different semantic classes.

A large pixel difference does not necessarily imply semantic divergence, and a small difference may still matter semantically.

### Single-trajectory interpretation

The detailed `5 → 9` analysis demonstrates a concrete trajectory phenomenon, but one trajectory is not sufficient to establish a general law about PPO-fine-tuned diffusion models.

---

# 19. What I Want to Try Next

The next step is to move from **pixel-space trajectory analysis** toward **semantic trajectory analysis**.

## 1. Predicted clean image at each timestep

For each intermediate state,

\[
\hat{x}_0^{(t)}
=
\frac{
x_t
-
\sqrt{1-\bar{\alpha}_t}
\epsilon_\theta(x_t,t)
}{
\sqrt{\bar{\alpha}_t}
}.
\]

Instead of looking only at noisy \(x_t\), this allows us to inspect what clean image the model currently predicts.

The question becomes:

```text
t       Baseline x̂0       PPO x̂0

400          ?                ?
350          ?                ?
300          ?                ?
250        5-like?          9-like?
200          5                9
...
0            5                9
```

This may help distinguish

```text
pixel-space divergence
```

from

```text
semantic divergence.
```

## 2. Analyze many trajectories

Rather than selecting one interesting sample, the same controlled experiment can be repeated across many initial noises.

Possible measurements include:

- trajectory MSE
- class-change frequency
- timestep of semantic divergence
- final reward difference
- feature-space trajectory distance

This would make it possible to determine whether progressive divergence is a general pattern or specific to individual samples.

## 3. Better reward design

The current reward could be extended to include terms for:

\[
R
=
R_{\text{confidence}}
+
\lambda_1R_{\text{diversity}}
+
\lambda_2R_{\text{distribution}}.
\]

This could test whether PPO can increase a desired reward without producing such a large class-distribution shift.

## 4. Move beyond MNIST

The same idea could eventually be tested on more complex diffusion architectures and datasets, where trajectory changes may involve not only class identity but also local structure, attributes, and semantic features.

---

# 20. Repository Structure

```text
DDPM-with-PPO--MNIST/
│
├── DDPM.ipynb
│
├── README.md
│
├── assets/
│   ├── ddpm_training_loss.png
│   ├── ddpm_baseline_samples.png
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

# Final Takeaway

This project started as a small exercise to understand DDPM by implementing it from scratch.

Then I asked:

> **What if reinforcement learning is added to the reverse diffusion process?**

PPO provided a way to experiment with that idea.

But the most interesting part was not the small increase in reward itself.

It was discovering that the reward optimization also changed the generated distribution, and then tracing that change back through the denoising process.

The controlled trajectory experiment showed:

\[
\boxed{
\text{PPO update}
\rightarrow
\text{small trajectory perturbation}
\rightarrow
\text{progressive divergence}
\rightarrow
\text{different final generation}
}
\]

This small experiment made me interested in diffusion models not only as models that transform noise into an image, but as **dynamic generative processes whose intermediate trajectories can themselves be analyzed**.
