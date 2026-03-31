# EGTEA Gaze+ Training Status

## Quick Start

The repository is set up so teammates can reproduce the current default demo-ready pipeline with a few commands.

1. Prepare the EGTEA Gaze+ dataset:

```powershell
python scripts/setup_dataset.py
```

2. Prepare EgoVideo code and checkpoint:

```powershell
python scripts/setup_egovideo.py
```

3. Run the default strongest pipeline:

```powershell
python scripts/run_default_training.py
```

4. Run a 1-epoch quick version that still produces a connected testing bundle:

```powershell
python scripts/run_default_training.py --quick-train --force-rebuild
```

5. For a fast end-to-end sanity check before full training:

```powershell
python scripts/run_default_training.py --smoke-test --force-rebuild
```

6. Predict one clip:

```powershell
python scripts/predict_single_clip.py OP03-R06-GreekSalad-331160-332380-F007944-F007981.mp4
```

7. Predict one session folder:

```powershell
python scripts/predict_clip_folder.py OP03-R06-GreekSalad
```

Default demo outputs are written to:

- `outputs/demo_ready/default_pipeline/`

The training/inference bundle lives at:

- `outputs/demo_ready/default_pipeline/bundle.json`

For the simplified teammate-facing commands, see:

- `scripts/SIMPLE_RUNS.md`

## Practical Notes

- The first call to `predict_single_clip.py` or `predict_clip_folder.py` will feel slower because the model weights and EgoVideo backbone need to be loaded first.
- After this warm-up, the actual per-clip prediction time is much shorter than the first observed run.
- Compared with the lighter `TSM` route we used earlier, the current EgoVideo single-clip training is roughly `2.2x` slower per epoch in our setup.
- In exchange, the first-epoch accuracy gain is roughly `10x` larger than what we typically observed from `TSM`.
- So although the per-epoch cost is higher, the accuracy-per-epoch tradeoff is still good enough for practical reproduction.

## 1. Current Main Route

Note:
- The detailed sections below include older historical route notes from earlier stages of the project.
- The current teammate-facing reproducible entrypoint is the Quick Start above.

This repository currently follows a practical two-stage route for EGTEA Gaze+ RGB-only action recognition and online workflow monitoring:

1. Train a strong single-clip RGB classifier.
2. Reuse that trained clip encoder to build a strictly causal context model that only looks at past and current clips.

The current recommended main route is:

- Single-clip model: `TSM`
- Context model: `2-layer causal GRU`
- Context input: `embedding_only`
- Context window: `K = 5`

This route is the most stable and the most aligned with the real-time constraint:

- no future clip leakage
- reusable offline feature extraction
- fast online inference
- better held-out accuracy than the plain single-clip baseline


## 2. What Has Been Completed

### Stage A. Single-clip RGB baselines

We first built and validated a simple RGB-only pipeline:

- data reading
- frame sampling
- training
- validation
- checkpoint saving

Then we unified the project around three stronger RGB action backbones:

- `TSM`
- `X3D-S`
- `SlowFast-R50`

These models share:

- the same EGTEA label mapping
- the same official split protocol
- the same unified train/eval scripts
- Kinetics-pretrained initialization when available

Among them, `TSM` became the strongest and most practical single-clip baseline in the current repo.


### Stage B. Causal context model for workflow monitoring

After the single-clip route was stable, we added a second stage for session-level online recognition:

- recover clip order inside each session
- build sequence manifests
- extract per-clip features from the trained single-clip encoder
- train a causal GRU on windows of past clips
- evaluate held-out performance
- simulate online inference clip by clip

Important design choice:

- the causal model predicts the label of the current clip `y_t`
- it only uses `[t-K+1, ..., t]`
- it never uses future clips


## 3. Current Recommended Models

### 3.1 Recommended single-clip model

`TSM`

Why it is the current main single-clip backbone:

- best overall performance among the tested RGB backbones
- stable training
- easy to fine-tune
- strong enough to act as a feature extractor for the context stage


### 3.2 Recommended context model

`Causal GRU (embedding_only)`

Why it is the current main context model:

- strictly causal
- simple and cheap
- improves over the single-clip baseline
- very low added online latency


## 4. What We Tried But Did Not Keep As The Main Route

### 4.1 Early `r3d_18` RGB baseline

Status:

- kept in the repo as an early baseline
- not the main route anymore

Why we moved on:

- its role was mainly to verify the pipeline end-to-end
- later unified baselines were stronger and more comparable
- `TSM` gave a better main backbone for downstream context modeling


### 4.2 X3D-S

Status:

- fully implemented and trainable
- not selected as the main route

Why it was not chosen:

- lighter and faster, but weaker than `TSM` in our current EGTEA setup
- less attractive than `TSM` as the main clip encoder once accuracy became the priority

When it may still be useful:

- fast iteration
- lightweight deployment experiments
- future efficiency-focused comparisons


### 4.3 SlowFast-R50

Status:

- fully implemented and trainable
- not selected as the current main route

Why it was not chosen:

- heavier engineering cost
- more sensitive to batch size and memory
- did not outperform `TSM` in the current setting

When it may still be useful:

- future multimodal fusion
- more ambitious temporal modeling work
- larger-scale experiments when compute budget is higher


### 4.4 Context input: `embedding_plus_logits`

Status:

- implemented and trained
- not selected as the default context input

Why it was not chosen:

- underperformed `embedding_only`
- likely introduced noisy or overly confident class-level signals into the GRU
- the cleaner pre-classifier embedding transferred better

Current conclusion:

- use `embedding_only` by default
- keep `embedding_plus_logits` only as an ablation/reference


### 4.5 End-to-end joint training of clip encoder + context model

Status:

- not pursued in the current stage

Why we deferred it:

- current goal is stable online workflow monitoring first
- offline feature extraction is simpler, cheaper, and easier to debug
- freezing the clip encoder avoids coupling too many failure modes at once

This remains a possible future direction after the current causal pipeline is fully settled.


## 5. Why The Current Route Was Chosen

The current main route was chosen because it balances:

- accuracy
- simplicity
- reproducibility
- online deployment friendliness

In short:

- `TSM` is the best current single-clip backbone in this repo.
- `Causal GRU + embedding_only` is the best current online context model.
- This pair gives us a realistic workflow-monitoring baseline without violating causality.


## 6. Current Practical Conclusion

If we continue from the current codebase, the recommended main baseline is:

- clip encoder: `TSM`
- context model: `Causal GRU`
- input mode: `embedding_only`
- window size: `K=5`

This should be treated as the default starting point for:

- online workflow monitoring
- future gaze integration
- future causal temporal model upgrades


## 7. Suggested Next Steps

The most sensible next experiments are:

1. tune the causal context model around the current best route
2. test larger context windows such as `K=7` or `K=9`
3. try stronger causal temporal heads, for example a causal Transformer
4. add gaze features on top of the current `TSM + causal GRU` pipeline
5. revisit end-to-end fine-tuning only after the frozen-feature version is fully stable
