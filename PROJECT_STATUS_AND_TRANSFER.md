# Project Status And Transfer Notes

## Scope

This document summarizes:

- what has been implemented so far in this EGTEA Gaze+ workflow monitoring project
- which lines of work became the current mainline
- which attempted methods were not kept as the main path and why
- how the current design can be ported into other real-time workflow monitoring projects
- a compact metric comparison across the main milestones

This summary reflects the current verified `split1` results. Running `split2` and `split3` with the same pipeline is the next obvious consolidation step.

## Current Mainline

The current formal action-recognition mainline is:

- backbone: `EgoVideo frozen-backbone`
- base action prediction: `single-clip EgoVideo`
- candidate reranking: `learned transition-aware reranker`
- state prior: `soft state prior`
- `candidate_k=10`
- `history_len=3`
- `prev_mode=prev3`

Current best verified result on `split1`:

- raw action baseline test top-1: `77.10`
- best transition-aware reranker test top-1: `79.72`
- best soft state-prior reranker test top-1: `80.66`

Why this is the current mainline:

- it is the strongest fully verified action pipeline in the repository
- it remains fully causal
- it is modular and easy to deploy in stages
- it is simpler and more stable than the more complex learned state-aware or session-consistency variants

Reference: [baseline_summary.md](/c:/Users/18447/Egtea_Gaze_Training/outputs/final_baselines/best_soft_state_prior/baseline_summary.md)

## What Has Been Done

### 1. Early action baselines

Implemented and compared several RGB action backbones:

- TSM
- X3D-S
- SlowFast-R50

These established the first unified training/evaluation pipeline but did not become the long-term mainline once EgoVideo was introduced.

### 2. Causal context on weaker backbones

Built causal context models on top of earlier single-clip encoders:

- GRU context head
- TCN context head
- K-window ablations

Main finding:

- context helped a bit
- improvements were small
- this suggested the main bottleneck was not just “missing temporal context”, but also the quality ceiling of the encoder itself

### 3. Switch to EgoVideo

EgoVideo was integrated as the new first-person clip encoder.

This became a major turning point:

- raw single-clip action accuracy jumped to a much stronger level
- top-5 and top-10 coverage became very high
- the task shifted from “find the correct class from scratch” toward “re-rank a small plausible candidate set”

Key observed coverage:

- top-5: about `97.4%`
- top-10: about `98.6%`
- top-15: about `99.4%`

This is why later work focused on reranking and workflow plausibility instead of rebuilding the whole classifier again.

### 4. Top-k reranking

Implemented:

- plain Top-k reranker
- transition-aware reranker
- state-constrained reranker

Main finding:

- plain reranking already helped
- transition-aware reranking helped more
- the best result came from a very simple `soft state prior`, not from the more complicated learned state-aware version

This established the current best action-recognition line.

### 5. State modeling

Implemented:

- v1 coarse state taxonomy
- v2 finer state taxonomy
- single-clip state models
- causal state models
- state-constrained action reranking

Main finding:

- `state` is easier and more stable than fine-grained `action`
- v1 state prior helped action reranking
- v2 taxonomy improved state classification itself, but did not improve the final action reranker over v1

So the project kept:

- `v1` as the practical action-reranking prior
- `v2` as an informative design experiment, not as the new mainline

### 6. Session-level consistency

Implemented:

- simple online consistency filter
- lightweight learned session-consistency residual

Main finding:

- these helped smoothing and reduced some jitter
- but neither beat the best soft state-prior reranker on test top-1

Conclusion:

- useful as deployment-time optional smoothing
- not worth promoting to the accuracy mainline

### 7. Next-action prediction

Implemented:

- naive transition baseline
- embedding-only MLP
- causal GRU next-action baseline
- GRU plus state
- state-conditioned next-action
- next-state to next-action variants

Main finding:

- next-action is a meaningful benchmark for workflow monitoring
- but current learned models did not clearly beat the strong naive transition baseline
- simply concatenating state was not enough
- more structured next-state / next-action modeling may still be worthwhile later

### 8. Workflow deviation / anomaly detection

Implemented:

- proxy anomaly benchmark
- transition violation score
- next-action mismatch score
- combined workflow deviation score

Main finding:

- deviation scoring works meaningfully even without full real anomaly labels
- the combined score is much stronger than any single signal

This branch is already useful as an online warning prototype.

### 9. Online warning system prototype

Implemented:

- anomaly schema
- annotation-ready anomaly template
- causal online warning aggregator
- case-study generation

Main finding:

- the system is already good enough for a prototype demo
- rules are strong enough for now
- a learned warning head is not necessary yet

## Compact Comparison

### Action Mainline

| stage | val_top1 | test_top1 | notes |
| --- | ---: | ---: | --- |
| Raw EgoVideo single-clip | 78.60 | 77.10 | reference action baseline |
| Best learned transition-aware reranker | 80.62 | 79.72 | action-only causal reranker |
| Best soft state-prior reranker | 82.00 | 80.66 | current formal mainline |

### State Modeling

| model | taxonomy | val_top1 | test_top1 | notes |
| --- | --- | ---: | ---: | --- |
| Single-clip state | v1 | 89.07 | 87.88 | coarse 6-state baseline |
| Causal GRU state h=3 | v1 | 89.35 | 87.54 | state sequence model |
| Single-clip state | v2 | 88.43 | 88.97 | finer 10-state taxonomy |
| Causal GRU state h=3 | v2 | 87.88 | 87.24 | did not help final action reranking |

### Session Consistency

| model | test_top1 | notes |
| --- | ---: | --- |
| Best soft state-prior reranker | 80.66 | reference |
| Session consistency simple filter | 80.42 | smoother but weaker |
| Session consistency learned | 80.46 | also weaker |

### Next-Action

| model | test_top1 | test_top5 | notes |
| --- | ---: | ---: | --- |
| naive transition | 25.09 | 50.54 | strong lower bound |
| MLP current embedding | 24.89 | 51.46 | weak anticipatory signal |
| GRU h=3 embeddings only | 23.86 | 52.73 | not better than naive |
| GRU h=3 embeddings + state | 25.04 | 53.45 | tiny help only |

### Deviation Detection

| model | test_auroc | test_auprc | test_balanced_acc | notes |
| --- | ---: | ---: | ---: | --- |
| transition_violation | 0.6516 | 0.7667 | 0.6107 | plausibility only |
| next_action_mismatch | 0.6575 | 0.7710 | 0.6166 | anticipatory mismatch only |
| combined_workflow_score | 0.8683 | 0.9389 | 0.8201 | current anomaly mainline |

## What Was Tried But Not Kept As Mainline

### TSM / X3D / SlowFast as long-term backbone

Why not kept:

- they were valuable as early baselines
- EgoVideo gave a much stronger first-person feature space
- once EgoVideo was available, further engineering on weaker backbones stopped being cost-effective

### Plain causal context as the main answer

Why not kept:

- it helped modestly on weaker baselines
- but later analysis showed the stronger gain came from a better encoder and better candidate reranking, not just a larger temporal head

### Learned state-aware reranker

Why not kept:

- it was more complex
- it underperformed the simpler soft state prior
- the simpler method was easier to interpret and deploy

### Session-level consistency as accuracy mainline

Why not kept:

- it reduced jitter a bit
- but did not improve test top-1 over the best soft state-prior reranker
- best use is optional smoothing during deployment

### v2 state taxonomy as the default state prior

Why not kept:

- it improved state classification itself
- but did not improve the final action reranker
- likely because the finer taxonomy injected more noise into the prior than benefit at the action-correction stage

### State-conditioned next-action as current mainline future branch

Why not kept:

- it was conceptually promising
- but current versions did not beat the simpler next-action baselines
- the benchmark is still worth keeping, but it is not yet a deployment-driving branch

## How To Transfer This To Other Real-Time Monitoring Projects

The project is already organized as a reusable monitoring stack, not just a classifier. The most portable design is:

### Layer 1: strong clip encoder

Use a reliable egocentric or domain-specific visual encoder.

Porting rule:

- if the new project already has a strong encoder, keep it
- if not, start by solving clip-level perception before adding workflow logic

### Layer 2: Top-k candidate generation

Do not rely only on top-1.

Porting rule:

- keep top-5 or top-10 candidates
- monitor candidate coverage
- if coverage is high, reranking is more valuable than rebuilding the classifier

### Layer 3: causal workflow-aware reranking

Use:

- transition-aware reranking
- simple state prior

This is currently the strongest balance of accuracy, interpretability, and deployability.

Porting rule:

- build train-only transition priors
- define a lightweight state taxonomy from label semantics
- add soft state prior to candidate reranking

### Layer 4: workflow deviation scoring

Use combined workflow plausibility signals:

- transition violation
- next-action mismatch
- confidence / entropy
- state support

Porting rule:

- start with a rule-based deviation score
- do not wait for a large anomaly dataset before prototyping
- use proxy abnormal sequences first

### Layer 5: online warning aggregator

Use a rule-based warning layer above the scores.

Porting rule:

- keep it causal
- emit structured outputs, not only labels
- start with interpretable rules and thresholds
- only add learned warning heads after real labels exist

## Recommended Minimal Transfer Recipe

If this pipeline were to be re-used in another workflow monitoring project, the minimum useful subset would be:

1. clip encoder
2. top-k candidate dump
3. transition-aware reranker
4. soft state prior
5. combined workflow deviation score
6. online warning aggregator

This is the smallest stack that already behaves like a usable workflow monitor instead of only an action classifier.

## What To Reuse First In Another Repo

Most reusable ideas are:

- `datasets/` pattern for session-ordered manifests
- cached candidate dump structure: embeddings, logits, top-k ids, top-k probs
- action-to-state mapping logic
- transition-prior construction
- soft state-prior reranking
- combined workflow deviation score
- rule-based online warning schema and annotation templates

## Suggested Next Steps

The project is at a point where the most valuable next step is not another small action-accuracy tweak.

Recommended order:

1. run `split2` and `split3` and report cross-split mean for the current mainline
2. add a small amount of real anomaly / deviation annotation
3. validate the warning system on real annotations
4. then consider:
   - next-action linked warning logic
   - finer anomaly definitions
   - optional LLM explanation layer

## Bottom Line

What worked best:

- strong egocentric encoder
- causal Top-k reranking
- simple state prior
- combined workflow deviation scoring

What did not become the mainline:

- weaker backbones
- over-complicated state-aware rerankers
- session-consistency as an accuracy layer
- current state-conditioned next-action variants

The project has already evolved from “action classification experiments” into a real prototype workflow monitoring stack.
