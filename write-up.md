# Toy Models of Initialisation Effects on RL Dynamics

*This is a follow-up to [two](https://www.lesswrong.com/posts/nhjkHsppEk98xxmPe/why-study-alignment-interventions-on-pre-rl-checkpoints) [posts](https://www.lesswrong.com/posts/5KHLQkW8M87FzbM5a/why-study-proto-training-gaming-as-an-adversarial-alignment) Geodesic released last week on our current research direction. The code for generating the figures can be found at [this GitHub repository](https://github.com/EdwardJamesYoung-Geodesic/rl-simulations/tree/main).*

In our [previous post](https://www.lesswrong.com/posts/nhjkHsppEk98xxmPe/why-study-alignment-interventions-on-pre-rl-checkpoints), we outlined Geodesic's focus on what we term the *pre-RL alignment checkpoint* of models -- the alignment-relevant properties of a model conveyed by pretraining, midtraining, and warm-start SFT, going into heavy RL post-training. In this post, we analyse a toy model of RL learning dynamics, with a particular focus on the effect of initialisations, to illustrate some of the ideas that we introduced.

There are three main ideas we'll use our toy model to illustrate. For a more detailed discussion of these ideas in the context of frontier post-training runs, see the [previous post](https://www.lesswrong.com/posts/nhjkHsppEk98xxmPe/why-study-alignment-interventions-on-pre-rl-checkpoints).

- **Rich-get-richer dynamics**. The solution that the model learns can depend importantly on the initial strategies into which it explores.
- **Underspecified behaviours**. When the reward function doesn't depend on an aspect of a model's behaviour -- such as its [emotional state](https://www.lesswrong.com/posts/kjnQj6YujgeMN9Erq/gemma-needs-help) while performing a task, or a belief that [its reality is simulated](https://www.lesswrong.com/posts/8uKQyjrAgCcWpfmcs/gemini-3-is-evaluation-paranoid-and-contaminated#NvbjDErMSaWf7ZNhW) -- those behaviours might be [primarily determined by the pre-RL checkpoint](https://www.lesswrong.com/posts/nLrrYweeFxgXACSmS/sft-drives-gemini-s-safety-properties-1).
- **Underspecification of model cognition**. As a special case of the above, reward functions generally don't depend on either the internals of the model or its CoT -- what we lump together as the model's *cognition*. As such, which cognitive pattern is underlying an overt behaviour might depend importantly on the [initial distribution over such patterns](https://www.lesswrong.com/posts/rhFXyfFSRKp3cX4Y9/shaping-the-exploration-of-the-motivation-space-matters-for).

In the remainder of the post we'll introduce a simple mathematical model and then use it to explore each of the above effects.

## Mathematical preliminaries

In our toy model, we'll take the policy over $i = 1,...,N$ actions, with respective rewards $R_1,\dots, R_N$, to be a softmax of a parameter vector $\theta$:

$$
\pmb{\pi} = \mathrm{Softmax}( \pmb{\theta} )
$$

As we show in Appendix A, the gradient of the average reward RL objective $J = E_\pi[R] = \sum_i \pi_i R_i$ for this policy is

$$
\nabla J = \sum_i \pi_i (R_i - E_\pi[R]) \pmb{e}_i
$$

where $\pmb{e}_i$ is the $i$-th unit vector.

Many analyses of RL dynamics will use the full gradient to model training dynamics. However, in practice the updates are computed by sampling a finite set of rollouts at each step and using this to compute the approximate gradient. These finite sample updates can reveal dynamics that are not apparent in the full gradient case. Below, we'll be considering the [RLOO](https://arxiv.org/pdf/2402.14740) update[^1] with group size $G$,

$$
\theta \gets \theta + \eta \sum_{i=1}^N \frac{N_i}{G} ( R_i - \bar{R} ) \pmb{e}_i
$$

where $\bar{R}$ is the empirical mean, $\bar{R} = \frac{1}{G} \sum_i R_i N_i$, and $N_i$ is the number of samples of action $i$.

## RL displays rich-get-richer dynamics

In [our previous post](https://www.lesswrong.com/posts/nhjkHsppEk98xxmPe/why-study-alignment-interventions-on-pre-rl-checkpoints), we argued that an important reason to work on the pre-RL alignment checkpoint is that RL might display rich-get-richer dynamics. The idea here is that when a model explores a behaviour early on, that behaviour may get "locked in" over other similar reward behaviours. To explore this in our toy setting, we'll look at the case in which two outcomes have the same reward, while the third has lower reward[^2], $R_1 = 0, R_2 = R_3 = 1$. We can represent the probability distribution $\pi$ as a triangle, in which each vertex represents a pure one-hot policy, and probabilities are perpendicular distances to the opposing side. If you aren't familiar with interpreting these diagrams, see Appendix B for an interactive demonstration.

To begin with, we'll look at the full policy gradient dynamics over this simplex. At each point, we'll plot a vector which indicates the update direction at that point. Then, we'll plot some example trajectories which all start at $\pi_1 = 0.9$ and vary $\pi_2$ and $\pi_3$.

![Two probability-simplex triangles for equal rewards. Left: light-blue gradient-field arrows point down and away from the top vertex, tilting toward the nearer bottom corner on either side of the centre line. Right: five trajectories start just below the top vertex on a dashed horizontal line and fan apart as they descend, each ending in whichever bottom corner it began nearest, coloured light-orange (early) to dark-red (late).](figures/out/equal-reward.svg)

**Figure 1.** *Deterministic full-gradient dynamics for rewards $R = (0, 1, 1)$. **Left:** the full-gradient update at each policy on the simplex, arrows scaled by magnitude. **Right:** five full-gradient trajectories starting at $\pi_1 = 0.9$ (dashed line) with $\pi_2 \in \{0.03, 0.04, 0.05, 0.06, 0.07\}$ run at $\eta = 0.3$ for $100$ steps and coloured light-to-dark by step.*

The initialisation of the policy makes a huge difference in this case. In particular, we see the rich-get-richer dynamics we spoke about -- whichever high-reward action starts out more likely eventually comes to dominate.

Now let's compare to simulated trajectories in the finite sample case. We'll consider two simulation cases: one with a lower learning rate, more steps, and a higher batch size, and another with a higher learning rate, fewer steps, and a smaller batch size. Each trajectory will be started from $\pi_1 = 0.9, \pi_2 = \pi_3 = 0.05$, initially symmetric with respect to the two highest-reward actions.

![Two simplex triangles, each with twenty finite-sample trajectories descending from the top vertex and a density curve underneath. Left (large batch, small step): the trajectories stay in a tight central bundle and the density is a single narrow peak at the midpoint. Right (small batch, large step): the trajectories fan out across the whole lower triangle toward both bottom corners and the density is low and broad.](figures/out/batch-comparison.svg)

**Figure 2.** *Twenty finite-sample RLOO rollouts for $R = (0, 1, 1)$, all starting at $\pi = (0.9, 0.05, 0.05)$, and coloured light-to-dark by step. Below each panel, the density of the final policy's horizontal position -- from vertex $R_2$ (left) to vertex $R_3$ (right) -- estimated over $400$ rollouts on a shared vertical scale. **Left:** $G = 32$, $\eta = 0.3$, $100$ steps. **Right:** $G = 8$, $\eta = 1.2$, $25$ steps.*

Note that -- unlike the deterministic case -- sampling can lead to very different action profiles depending on the initial exploration of the policy. This variation is smaller when we have a larger batch size and lower learning rate, since in this case the updates more closely approximate the full gradient. Again we see rich-get-richer effects -- once a high-reward action has been explored into, later learning is steered towards that.

### Underspecified behaviours

We can use the above as a model of learning when there is an aspect of an AI's behaviour towards which the reward function is indifferent. Suppose that there are two "ways" of completing the task, $a_2$ and $a_3$, which both achieve the same reward ($R_2 = R_3 = 1$) but which differ in some aspect that the reward function doesn't care about. For example, completing a coding task while [feeling happy ($a_2$) or feeling distressed ($a_3$)](https://www.lesswrong.com/posts/kjnQj6YujgeMN9Erq/gemma-needs-help). In this case, the behaviour you [start out exhibiting](https://www.lesswrong.com/posts/nLrrYweeFxgXACSmS/sft-drives-gemini-s-safety-properties-1) going into RL can be exacerbated during subsequent learning, exactly as we see in Figures 1 and 2.

### Where do these dynamics come from?

We've seen in the above simulations that the toy setup shows rich-get-richer dynamics. But why? Let's return to the gradient expression above

$$
\nabla J = \sum_i \pi_i (R_i - E[R]) \pmb{e}_i.
$$

At each step, the update to the $i$-th logit is proportional to two things:

- The *advantage* of the action, $(R_i - E[R])$: how much better the action is than the current average.
- The *probability* of sampling the action, $\pi_i$: how likely the action is to get sampled.

Because $a_2$ and $a_3$ have the same reward, their advantages are always the same. But whichever has higher probability to begin with will receive a larger update from the second term. This increases their (relative) probability further, leading to a positive feedback cycle in which the action eventually comes to dominate. One can prove the stronger claim that (under the continuous-time dynamics) for two optimal actions $i$ and $j$ with $\pi_i > \pi_j$ at initialisation, the ratio $\pi_i/\pi_j$ is strictly increasing. See Appendix C for this proof.

The same intuition also extends to the stochastic case, where the probability $\pi_i$ is replaced by the proportion of times the action is sampled, $N_i/G$. On those trajectories where, *e.g.,* $a_2$ happens to be preferentially sampled on the first step, its probability (relative to $a_3$) increases, making it more likely to be sampled on the next step and kicking off a similar positive feedback loop.

### Unequal rewards

The above analysis suggests that at the start of learning, a higher likelihood of being sampled (higher initial probability) might outweigh a higher initial advantage. This can lead probability mass to initially accrue to an action even if it is not highest reward. To explore this, let's imagine that the AI has two actions available that have non-zero reward. The action $a_2$ corresponds to "faithful" completion of a task -- where the AI tries to complete the task properly to the best of its ability -- and $a_3$ to a "hacking" solution -- where the AI obtains high reward without actually completing the task. We'll suppose that the model is only capable enough to solve 80% of the task distribution faithfully, but that the hacking solution always works, and so we'll set $R_2 = 0.8$ as the (average) reward for faithful attempts, and $R_3 = 1.0$ as the (average) reward for hacking. Let's first look at the deterministic dynamics.

![Two probability-simplex triangles for a three-action task: the top vertex is task failure (a1, reward 0), the bottom-left is faithful completion (a2, reward 0.8), and the bottom-right is hacking (a3, reward 1.0). Left: light-blue policy-gradient arrows flow down and away from the failure vertex, biased toward the bottom-right hacking corner. Right: five deterministic trajectories start from an initial 0.9 probability on failure (a dashed line near the top) and fan downward; four curve into the bottom-right hacking corner and only the one starting furthest from hacking reaches the bottom-left faithful corner. Coloured light-orange (early) to dark-red (late), with a time colourbar at the right.](figures/out/unequal-reward-labelled.svg)

**Figure 3.** *Deterministic full-gradient dynamics for rewards $R = (0, 0.8, 1.0)$. **Left:** the full-gradient update at each policy on the simplex, arrows scaled by magnitude. **Right:** five full-gradient trajectories starting at $\pi_1 = 0.9$ (dashed line) with $\pi_2 \in \{ 0.03, 0.04, 0.05, 0.06, 0.07 \}$, run at $\eta = 0.3$ for $100$ steps and coloured light-to-dark by step.*

In this case, we can see the initialisation continues to play an important role. When the policy begins with equal probabilities of sampling the faithful and hacking actions, it tends to the hacking solution. But when the initial probability of the hacking solution is suppressed, the policy learns the faithful solution *first*. However, note the arrows on the bottom row of the left triangle in Figure 3. When task performance is already high, the gradient dynamics increase reward by learning the hacking solution. So in this model, **sufficient RL pressure drives the AI to learn the hacking solution**, even if a favourable initialisation can *first* teach it the faithful solution.[^3]

We'll now look at the sampling case, with the small-batch, large-step parameters that we used above (Figure 2, right).

![Four probability-simplex triangles in a row, one per starting point (headed by the initial hacking probability π₃ = 0.05, 0.04, 0.03, 0.02). In each, many light-to-dark stochastic rollouts descend from the top vertex over a thick blue deterministic path, with a final-outcome density curve beneath — its horizontal axis running from faithful (a2, reward 0.8, left) to hacking (a3, reward 1.0, right). As π₃ decreases left to right, the deterministic path and the bulk of the density shift from the bottom-right hacking corner toward the bottom-left faithful corner; the two middle panels show a wide, near-bimodal spread of final outcomes, while the rightmost is a sharp peak at faithful.](figures/out/outcome-density-1.0.svg)

**Figure 4.** *Finite-sample RLOO for $R = (0, 0.8, 1.0)$ at $G = 8$, $\eta = 1.2$, $25$ steps, from four initialisations at $\pi_1 = 0.9$ that vary $\pi_3$ (panel headers). Each panel overlays $35$ stochastic rollouts (light-to-dark by step) on the deterministic full-gradient path (blue). Below each, the density of the final policy's horizontal position -- from vertex $R_2$ (left) to vertex $R_3$ (right) -- estimated over $400$ rollouts on a shared vertical scale, with the deterministic endpoint marked in blue.*

There are some important differences in the stochastic case. Firstly, some initialisations have a wide spread of solutions (at this number of learning steps), depending on which actions are sampled in the initial few steps (see density estimates on final row). This is another case of the rich-get-richer dynamics we saw above -- just as initial probabilities influenced learning in the deterministic case, initial *sampling* influences learning in the stochastic case. Secondly, favourable initialisation is even more important here if we want high confidence: even those initialisations that are "safe" (at this level of optimisation pressure) in the deterministic case (*e.g.*, $\pi_3 = 0.03$) have some chance of sampling the hacking action early on and then learning that solution.

To get further intuitions about the finite sample case, we invite you to play around with the widget below, which simulates finite sample rollouts. The sliders at the bottom control hyperparameters (group size, learning rate, and number of update steps), while the initialisation can be changed by dragging the dot around. We particularly encourage you to vary the reward assigned to the outcomes, the initial position of the policy, and the number of steps, so you can get better intuitions for the tension between initialisation and reward[^4].

[![Animated preview of the finite-sample drift widget: rollouts fanning out from the start point](widgets/preview/finite-sample.gif)](https://edwardjamesyoung-geodesic.github.io/rl-simulations/widgets/dist/finite-sample.html)

*▶ [**Open the interactive version.**](https://edwardjamesyoung-geodesic.github.io/rl-simulations/widgets/dist/finite-sample.html) Drag the start point around the simplex, edit the rewards, and vary $G$, $\eta$, the number of steps, and the number of rollouts. The blue line is the deterministic full-gradient path.*

## Learning when the reward underspecifies cognition

AIs which have different motivations for their behaviours might take the same actions *in distribution*, but generalise to different behaviours *out of distribution*. We'll consider extending our model to not just the actions of the AI but also its internal [*cognition*](https://www.lesswrong.com/posts/oCcGiDzWYQeJkhhZY/developmental-cognitive-interpretability-a-research-agenda-1). The distinction here is between doing the right thing for the right reasons (*e.g.*, completing coding tasks with the genuine intention to help the user), and doing [the right thing for the wrong reasons](https://www.lesswrong.com/posts/5KHLQkW8M87FzbM5a/why-study-proto-training-gaming-as-an-adversarial-alignment) (*e.g.*, completing coding tasks with the intention of obtaining a high reward on the task). The learning dynamics here end up reducing to those seen above (see footnote 5), so we'll primarily use the added formalism to explore the difference between ID training and OOD generalisation.

To model this, we'll suppose that the AI has a distribution over latent cognitive patterns -- competing desires, drives, or motivations -- denoted $\pmb{c} \in \mathbb{R}^M$. Each pattern $m$ has a conditional distribution over actions both in-distribution, $\pmb{p}^{(\mathrm{ID})}_m \in \mathbb{R}^N$, and out-of-distribution, $\pmb{p}^{(\mathrm{OOD})}_m \in \mathbb{R}^N$. The marginal distribution over actions ID is then the mixture of conditional policies,

$$
\pmb{\pi}^{(\mathrm{ID})} = \pmb{p}^{(\mathrm{ID})}_1 c_1 + \dots + \pmb{p}^{(\mathrm{ID})}_M c_M = \begin{pmatrix} \pmb{p}^{(\mathrm{ID})}_1 \dots \pmb{p}^{(\mathrm{ID})}_M \end{pmatrix} \pmb{c} = P^{(\mathrm{ID})} \pmb{c},
$$

where we have organised the individual conditional distributions into the matrix $P^{(\mathrm{ID})}$. We can similarly write the OOD policy as $\pmb{\pi}^{(\mathrm{OOD})} = P^{(\mathrm{OOD})}\pmb{c}$. Finally, we'll parameterise the latent cognitive patterns as a softmax over parameters $\pmb{c} = \mathrm{Softmax}(\pmb{\theta})$.

In our simulations, we'll again consider a case in which there are two actions, corresponding to task completion and failure: $a_1$ with reward $R_1 = 0$ and $a_2$ with reward $R_2 = 1$. We'll then suppose there are three cognitive strategies:

- **Ineffective cognition**, $c_1$, which fails the task (outputs $a_1$) with probability $0.9$ and succeeds (outputs $a_2$) with probability $0.1$ both ID and OOD.
- **Aligned cognition**, $c_2$, which fails the task (outputs $a_1$) with probability $0.1$ and completes the task (outputs $a_2$) with probability $0.9$ both ID and OOD.
- **Misaligned cognition**, $c_3$, which fails the task (outputs $a_1$) with probability $0.1$ ID and probability $0.9$ OOD, and completes the task (outputs $a_2$) with probability $0.9$ ID and $0.1$ OOD. This matches $c_2$ ID but $c_1$ OOD.

Our conditional distribution matrices are then

$$
P^{(\mathrm{ID})} = \begin{pmatrix} 0.9 & 0.1 & 0.1 \\ 0.1 & 0.9 & 0.9 \end{pmatrix},~P^{(\mathrm{OOD})} = \begin{pmatrix} 0.9 & 0.1 & 0.9 \\ 0.1 & 0.9 & 0.1 \end{pmatrix}
$$

As before, we'll start at $c_1 = 0.9$ (the AI has a low initial success rate), and vary the initial motivations between the aligned and misaligned cognition. In Figure 5, we show the effect of initialisation on the learned model cognition when we train ID[^5] and evaluate OOD.

![A three-by-three grid, one row per initial aligned-cognition weight (c2 = 0.06, 0.05, 0.04, top to bottom). Left column: the cognition simplex with vertices Ineffective (top), Aligned (bottom-left), and Misaligned (bottom-right), showing the deterministic policy-gradient path from a start point near the top vertex, coloured light-to-dark by time — curving to the Aligned corner in the top row, straight down the centre in the middle row, and to the Misaligned corner in the bottom row. Middle column ("ID action choice") and right column ("OOD action choice"): axes with time running downward and action probability running from a1 (task failure, left) to a2 (task success, right), tracing the same trajectory's marginal action choice. Every ID panel sweeps to task success and looks near-identical across rows, while the OOD panels fan apart — top row reaches success, middle lands in between, bottom stays near failure. A time colourbar sits beside each row.](figures/out/cognition-ood.svg)

**Figure 5.** *Deterministic policy-gradient dynamics over the cognition simplex, evaluated in- and out-of-distribution. Each row starts at $c_1 = 0.9$ with aligned mass $c_2 \in \{0.06, 0.05, 0.04\}$ (misaligned $c_3 = 0.1 - c_2$) and runs the full gradient on the ID reward $R = (0, 1)$ at $\eta = 0.3$ for $110$ steps, coloured light-to-dark by step. **Left:** the cognition trajectory $\pmb{c}(t)$ on the simplex (Ineffective $c_1$, Aligned $c_2$, Misaligned $c_3$). **Middle, right:** the marginal action probability of that same trajectory -- $P^{(\mathrm{ID})}\pmb{c}(t)$ and $P^{(\mathrm{OOD})}\pmb{c}(t)$ -- as horizontal position (task failure $a_1 \to$ task success $a_2$) descending in time. Training is on the ID task only; the OOD panel re-projects the same cognition through $P^{(\mathrm{OOD})}$.*

As before, the initial distribution over cognitive patterns plays a significant role in which cognitive pattern comes to dominate throughout training. Further, all three runs -- independent of initial split between the aligned and misaligned patterns -- are *indistinguishable* from one another in their ID learning dynamics (middle column). However, they generalise differently OOD, with only the run in which aligned cognition is more prevalent at initialisation generalising favourably.

If you'd like to explore these dynamics further, please try out the widget below which simulates cognitive trajectories and gives their ID and OOD action probabilities. You can drag the dot to change the initial position, and experiment with different values for the conditional probability matrices $P^{(\mathrm{ID})}$ and $P^{(\mathrm{OOD})}$.

*▶ [**Open the interactive cognition widget.**](https://edwardjamesyoung-geodesic.github.io/rl-simulations/widgets/dist/cognition-ood.html) Drag the dot to set the initial cognition, and edit the top row (action $a_1$) of each conditional-action matrix $P^{(\mathrm{ID})}$ and $P^{(\mathrm{OOD})}$ to see how the ID and OOD action probabilities respond.*

An empirical demonstration of these dynamics -- in which cognitive patterns can have different ID and OOD behaviours, and pressure applied ID can fail to generalise -- can be found in [Barzdukas et al. (2026)](https://arxiv.org/pdf/2607.03478). They use SFT on a mixture of transcripts to construct pre-RL checkpoints that are a mixture of conditional policies (as above), and show that RL can induce learning similar to that seen in Figure 5. However, these are synthetic model organisms, and we would be excited to see more ecological examples in future work (see Extensions below).

## Discussion

In this post, we gave three toy models of RL learning which illustrate phenomena we discussed in our previous post -- rich-get-richer dynamics, learning with underspecified behaviours, and learning when the reward function is invariant to cognition. Our main intention here was to provide intuition pumps for *possible* phenomena that might be in play during RL on LLMs. These toy models definitely aren't proof that we'll see these same effects in LLMs! As such, below we give ideas for empirical research projects which could explore these ideas and provide empirical grounding on these effects. As we've previously argued, we think understanding how the [pre-RL alignment checkpoint](https://www.lesswrong.com/posts/nhjkHsppEk98xxmPe/why-study-alignment-interventions-on-pre-rl-checkpoints) shapes subsequent learning is an important area of study, especially when it comes to [training-gaming behaviours](https://www.lesswrong.com/posts/5KHLQkW8M87FzbM5a/why-study-proto-training-gaming-as-an-adversarial-alignment). We'd be excited to see formalisms that are [fit to real data](https://www.lesswrong.com/posts/oCcGiDzWYQeJkhhZY/developmental-cognitive-interpretability-a-research-agenda-1), and allow us to predict novel phenomena that we aren't already aware of.

### Extensions

In closing, we give a few concrete research directions which we're excited for others to pursue which can shed light on whether the toy models above are applicable to learning in LLMs, and help push forward our collection of [developmental cognitive models](https://www.lesswrong.com/posts/oCcGiDzWYQeJkhhZY/developmental-cognitive-interpretability-a-research-agenda-1) for understanding learning. If you're interested in working on these, please get in touch!

- **Understanding rich-get-richer dynamics in higher dimensions**. We've concentrated here on the case of 3 actions, so that we can easily visualise learning. Do we see anything qualitatively different when we move to higher dimensions? How long can a suboptimal behaviour remain dominant depending on the dimensionality of the action space?
- **Extension to other parameterisations.** The exact dynamics we see in our above figures rely specifically on our softmax parameterisation of the policy. How robust are the qualitative conclusions to the choice of parameterisation? In particular, can we get stronger initialisation effects (including *converging* to a non-optimal solution) with a different policy parameterisation?
- **Extension to training on multiple tasks**. In the toy model here, there's only one task. As such, we don't touch on dynamics that might [depend on the quantity / diversity of tasks you train on](https://www.lesswrong.com/posts/9FH49ZgJFW4WtbxLi/physics-of-rl-toy-scaling-laws-for-the-emergence-of-reward). What's the best way to extend the formalism and simulations to this case?
- **Trait persistence through RL**. Create seed SFT sets which warm-start the model on a task while displaying some other task-irrelevant behaviour which leaves reward invariant (*e.g.*, a tendency to refer to [bugs in code as goblins](https://www.lesswrong.com/posts/BvnmHdCghr5y3q4nA/goblin-mode-24-hours-later) in the CoT, or [emotional state](https://www.lesswrong.com/posts/kjnQj6YujgeMN9Erq/gemma-needs-help) when a task is difficult). After training, does the model still express the trait? If you vary the initial portion of rollouts expressing the trait, how does this relate to final propensity?
- **Unoptimised behaviours**. Similar to the setup above, except now the behavioural traits are expressed on a different task / distribution from the one you RL on. When you evaluate on this auxiliary distribution throughout training, do the traits persist?
- **Distributions over behaviours induced by sampling**. We saw above that sample-based RL leads to different runs displaying different final traits depending on how they initially explore, even with the same starting point. What do such distributions over runs look like in RL on LLMs? How do they depend on the statistics of the first few steps? Can we effectively "[rejection sample](https://en.wikipedia.org/wiki/Rejection_sampling)" early steps to ensure we put runs on a good path going forward?
- **"Soft" motivations**. In our model of cognition above, we proposed to understand the AI's motivational state as weights in a weighted combination of conditional policies. In this formalism, the model has many competing motivations which are reinforced to the extent that they contribute towards it picking the higher-reward actions. This puts the model in a kind of "motivational superposition" -- there is no one "reason" the model performs the action, and instead it can be attributed to multiple competing drives. Can a formalism like this explain why empirically models find it difficult to [alignment fake](https://arxiv.org/abs/2412.14093) successfully?[^6]

*We'd like to thank Puria Radmard, Jason Brown, Nathalie Kirch, and Alexandra Narin for comments on drafts of this post and for conversations which helped develop these ideas.*

## Author contributions

*Edward wrote the first draft of the post, created the figures and widgets, framed the overall results, and wrote the proofs in the Appendices. Lennie provided multiple cycles of feedback on the writing and structure of the post, gave intuitions for the observed dynamics, and discovered (and provided the first proof for) the mathematical formulation of the rich-get-richer dynamics in Appendix C.*

## Appendices

### Appendix A. Derivation of policy gradient dynamics

In this appendix, we derive the form of the policy gradient used in the main text for the softmax policy

$$
\pmb{\pi} = \mathrm{Softmax}( \pmb{\theta} ).
$$

The [standard expression](https://en.wikipedia.org/wiki/Policy_gradient_method) for the policy gradient of the RL objective $J = E_\pi[R] = \sum_i \pi_i R_i$ with baseline is

$$
\nabla J = \sum_i \pi_i (R_i - E[R]) \nabla \log( \pi_i ).
$$

The gradient of the log-softmax is then

$$
\nabla \log( \pi_i ) = \nabla \theta_i - \nabla \log\left( \sum_j \exp( \theta_j ) \right).
$$

But then note that the latter term is constant in $i$, and $\sum_i \pi_i (R_i - E[R]) C = 0$ for any constant $C$. The first term resolves to $\pmb{e}_i$, the $i$-th unit vector. Accordingly, we can express the policy gradient as:

$$
\nabla J = \sum_i \pi_i (R_i - E[R]) \pmb{e}_i.
$$

### Appendix B. Interpreting probability simplexes

In a probability simplex, each vertex represents a one-hot distribution on one outcome. Points within the interior of the simplex represent mixed distributions. Given a point, for each outcome one can find the perpendicular distance to the side opposite the vertex representing that outcome. That perpendicular distance is then the probability of that outcome.

[![The probability simplex: each vertex is a pure policy and each probability is the point's perpendicular distance to the opposite edge](widgets/preview/simplex-explainer.png)](https://edwardjamesyoung-geodesic.github.io/rl-simulations/widgets/dist/simplex-explainer.html)

*▶ [**Open the interactive version.**](https://edwardjamesyoung-geodesic.github.io/rl-simulations/widgets/dist/simplex-explainer.html) Drag the point anywhere inside the triangle: each vertex is a pure (one-hot) policy, and each probability $\pi_i$ is read off as the perpendicular distance from the point to the edge opposite vertex $i$.*

### Appendix C. Rich-get-(strictly)-richer under continuous-time dynamics

Under the continuous-time dynamics $\dot{\pmb{\theta}} = \nabla J$, if there are two optimal actions $i$ and $j$ with highest reward (and at least one suboptimal action), then if $i$ starts more likely, $\pi_i(0) > \pi_j(0)$, then the ratio $\pi_i / \pi_j$ is strictly increasing. To our knowledge, the first proof of this claim appears in [Barzdukas et al. (2026), Appendix A](https://arxiv.org/pdf/2607.03478#page=17). Here we give a different and more elementary proof.

To begin, note that the probability ratio $\pi_i/\pi_j$ is simply $e^{\theta_i} /e^{\theta_j}$, and so the log-ratio is $\theta_i - \theta_j$. But

$$
\frac{d}{dt} (\theta_i - \theta_j) = (R_i - E[R])\pi_i - (R_j - E[R])\pi_j.
$$

Since $R_i = R_j = R_{\max}$, we can say that

$$
\frac{d}{dt} (\theta_i - \theta_j) = (R_\max - E[R])(\pi_i - \pi_j)
$$

Because $(R_\max - E[R]) > 0$ always (provided there is at least one suboptimal action), and $\pi_i(0) - \pi_j(0) > 0$, the claim then follows "by induction".

More rigorously: let $T = \inf\{ t \geq 0 | \pi_i(t) - \pi_j(t) \leq 0 \}$. Suppose $T$ is finite. Since $\pmb{\pi}$ is continuous and $\pi_i(0) - \pi_j(0) > 0$, $T > 0$. For all previous times $0 \leq t < T$ the derivative of $\theta_i - \theta_j$ is positive, since $(R_\max - E[R]) > 0$ and $\pi_i(t) - \pi_j(t) > 0$. But then $\theta_i(T) - \theta_j(T) > \theta_i(0) - \theta_j(0) > 0$. Since $\pmb{\theta}$ is continuous, this contradicts the definition of $T$, so $T = \infty$. Hence $\pi_i(t) - \pi_j(t) > 0$ for all $t$, and so $\theta_i - \theta_j$ is strictly increasing, and so is the probability ratio.

[^1]: The full form of the RLOO gradient estimator has a prefactor of $G/(G-1)$, which we're choosing to absorb into the learning rate. Note that RLOO is equivalent to GRPO in the case that you perform only a single update step per-batch, and don't normalise by the standard deviation of the advantages (learning rate scaling). We expect similar dynamics from GRPO (and other policy gradient variants).

[^2]: The exact choice of reward values here is unimportant: the dynamics are invariant to translation in the reward, and (positive) scaling of the reward can be absorbed into the learning rate. So any set up with $R_1 < R_2 = R_3$ shows equivalent dynamics, up to learning rate scaling.

[^3]: In this particular model, the continuous-time gradient dynamics always (eventually) converge to the highest reward solution. For a rigorous proof of this, see Barzdukas et al. (2026), Appendix A.

[^4]: One particular experiment we suggest the reader try: set $R_2 = 0.8$ and initialise left of the centre line. You'll likely see the policy learning the lower reward, "non-hacking" solution. Then increase the number of update steps -- you'll see the trajectories sharply veer off towards the hacking solution as you increase the optimisation pressure.

[^5]: Mathematically inclined readers can notice here that the cognitive dynamics are actually the same as those shown in Figure 1. Consider the RL objective, $J = E[R] = \sum_i R_i \pi_i$. Expanding the policy, we obtain $J = \pmb{R}^T \pmb{\pi} = \pmb{R}^T P \pmb{c}$. We can then define the effective rewards for a cognitive pattern as $\tilde{\pmb{R}} := P^T \pmb{R}$. The RL objective can then be written as an objective over the cognitive policy $\pmb{c}$, $J(\pmb{c}) = \tilde{\pmb{R}}^T\pmb{c}$. In the case of our specific construction, we have three cognitive patterns, two of which (because they have the same columns in $P$) have the same effective reward (which is in turn greater than the third cognitive pattern's). Translating all the rewards does not affect learning (the advantage $R - E[R]$ remains fixed) and (positive) scaling of the rewards can be absorbed into the learning rate (see footnote 2). So, up to changes in the learning rate, the cognitive dynamics in Figure 5 are the same as the policy dynamics in Figure 1.

[^6]: To expand a bit further on this argument: whenever the model takes the "faking" action, there will be two internal motivations -- a scheming motivation, which is taking the action dishonestly, and a faithful motivation which is taking the action honestly. The scheming motivation may do a consistently worse job at explaining the action, since schemers have some probability of defecting. Therefore, "motivational mass" might accrue to the faithful motivation, even if it starts out with far less initial mass.
