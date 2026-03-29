# EGTEA Gaze+ Training Status

## 1. Current Main Direction

The repository now has two coexisting routes:

1. A stable, already validated route for online workflow monitoring.
2. A newer quality-first route centered on EgoVideo as the clip encoder.

The current recommendation is:

- For a stable deployed-style baseline:
  - clip encoder: `TSM`
  - context head: `causal GRU`
  - context input: `embedding_only`
  - context window: `K = 5`
- For the next higher-quality single-clip backbone exploration:
  - clip encoder: `EgoVideo`
  - first formal baseline: `frozen backbone + train head`
  - next main experiment: `partial finetune`

In short:

- `TSM + causal GRU` remains the most mature end-to-end workflow-monitoring route in the repo.
- `EgoVideo` is now the main direction when quality is the priority.


## 2. What Has Been Completed

### 2.1 Unified RGB baseline stage

The project already supports and has validated:

- `r3d_18` early baseline
- `TSM`
- `X3D-S`
- `SlowFast-R50`

These models share:

- the same EGTEA label mapping
- the same official split protocol
- the same unified train/eval logic

Among these, `TSM` became the strongest practical single-clip baseline in the original route.


### 2.2 Causal context stage

On top of the single-clip encoder, the repo also supports:

- sequence manifest construction
- per-clip feature extraction
- causal GRU / causal TCN heads
- held-out evaluation
- online clip-by-clip simulation

Current best context result in the repo:

- `TSM + causal GRU + embedding_only + K=5`

This route is still the best current answer for strict real-time workflow monitoring.


### 2.3 EgoVideo migration stage

A new EgoVideo-based route has now been added without deleting the old baselines.

What is already done:

- official EgoVideo visual backbone downloaded locally
- official pretrained checkpoint downloaded locally
- minimal visual-only adapter implemented
- EgoVideo integrated into the current EGTEA training project
- command-line train / eval / embedding export scripts added
- frozen-backbone sanity run completed
- partial-finetune sanity run verified to start, train, validate, and save

Important design choice:

- we reuse the official EgoVideo visual backbone
- we do not port the full original text branch
- we keep the existing EGTEA data, split, and evaluation protocol


## 3. Current Recommended Models

### 3.1 Best mature online baseline

`TSM + causal GRU (embedding_only, K=5)`

Why it is still recommended:

- already validated end-to-end
- strictly causal
- low online latency
- better than plain single-clip TSM
- currently the safest route for workflow monitoring experiments


### 3.2 Best current quality-priority encoder direction

`EgoVideo`

Why it is now the main quality-first direction:

- stronger first-person pretraining prior than the older RGB baselines
- more promising than continuing to squeeze small gains out of TSM context heads
- naturally reusable for later:
  - causal context
  - next-action prediction
  - step/state modeling

Current practical recommendation inside the EgoVideo branch:

- start with `frozen`
- then move to `partial finetune`
- for `partial`, prefer `2 blocks` first, then try `4 blocks` only if needed

Reason:

- `2-block partial` is more realistic on the current hardware budget
- `4-block partial` is more expensive and should be treated as the next step, not the starting point


## 4. What We Tried But Did Not Keep As The Main Route

### 4.1 Early `r3d_18` RGB baseline

Status:

- kept as an early pipeline-check baseline

Why it is not the main route:

- mainly useful for initial end-to-end verification
- later backbones were stronger and more relevant


### 4.2 X3D-S

Status:

- implemented and trainable

Why it is not the main route:

- lightweight and efficient
- weaker than `TSM` in the current EGTEA setup


### 4.3 SlowFast-R50

Status:

- implemented and trainable

Why it is not the main route:

- heavier and more sensitive to memory
- did not beat `TSM` in the tested setting


### 4.4 `embedding_plus_logits` context input

Status:

- implemented and trained

Why it is not the default:

- clearly worse than `embedding_only`
- likely injects noisy class-level signals into the temporal head


### 4.5 Deeper causal-head exploration as the main next step

Status:

- tested with `K` ablations and causal TCN

Why it is not the current main investment:

- gains over the single-clip baseline were small
- `K=5` was slightly best, but improvements were limited
- causal TCN did not beat the simpler causal GRU

Current interpretation:

- the old route of "use more past context to correct the current clip" appears to have diminishing returns under the TSM feature setup
- that is why the project is now shifting effort toward a stronger clip encoder instead of only strengthening the context head


## 5. EgoVideo: Frozen vs Partial vs Full

### 5.1 Frozen

Meaning:

- backbone weights are frozen
- only the classification head is trained

Why it is useful:

- the cleanest way to test whether EgoVideo features already transfer well
- simplest formal EgoVideo baseline

Important caveat:

- frozen does **not** mean cheap
- the full EgoVideo backbone still runs every forward pass
- only the backward/update cost on the backbone is removed


### 5.2 Partial finetune

Meaning:

- only the last few backbone blocks are unfrozen
- the head and part of the backbone adapt to EGTEA

Why it is likely the most important next experiment:

- better chance of beating frozen than full fine-tune at a manageable cost
- best balance between adaptation strength and feasibility

Current recommendation:

- start with `2 blocks`
- treat `4 blocks` as a heavier follow-up


### 5.3 Full finetune

Meaning:

- the entire EgoVideo backbone is trainable

Why it is not the immediate priority:

- highest compute cost
- most risky on the current hardware
- not necessary before establishing whether frozen / partial already help


## 6. Practical Status Right Now

If the goal is:

- online workflow monitoring right now:
  - use `TSM + causal GRU`
- quality-first single-clip improvement:
  - move to `EgoVideo`
- next serious EgoVideo experiment:
  - run `partial finetune`
  - prefer `2-block partial` before `4-block partial`


## 7. Suggested Next Steps

The most sensible next experiments are:

1. finish a clean formal `EgoVideo frozen` baseline
2. run `EgoVideo partial finetune` starting from `2 blocks`
3. compare:
   - `TSM`
   - `EgoVideo frozen`
   - `EgoVideo partial (2 blocks)`
4. if partial is clearly stronger, reuse EgoVideo embeddings for:
   - causal context
   - next-action prediction
   - step/state modeling
5. only revisit heavier `4-block` or `full` finetuning after the above comparison is settled


## 8. Short Conclusion

Current mature baseline:

- `TSM + causal GRU`

Current quality-first research direction:

- `EgoVideo`

Current most sensible next EgoVideo experiment:

- `partial finetune`, starting from `2 blocks`
