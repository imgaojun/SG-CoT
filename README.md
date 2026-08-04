# SG-CoT: Schema-Grounded Structured Chain-of-Thought for Unseen-Type Event Extraction

Code, prompts, split protocols, and generation/verification/evaluation scripts for the paper
*Schema-Grounded Structured Chain-of-Thought for Unseen-Type Event Extraction*.

SG-CoT trains an event extractor whose reasoning trace mirrors the task's sub-decisions —
recall-first candidate audit, trigger-anchor lock, contrastive event-type arbitration, and local
argument attachment. Traces are verbalized by a teacher LLM from gold structures and admitted only
after hard (format plus exact surface/evidence match) and semantic verification. On Rich ERE the
method improves unseen-type Argument/Event/Trigger F1 over a direct baseline by
`+0.083 / +0.074 / +0.187`, and ablation, inference-time deletion, and counterfactual editing of the
verbalized arbitration localize the gain to the type-arbitration step.

## What is and is not in this repository

**Included:** model and training code (`src/`), the full experiment driver and analysis scripts
(`scripts/`), training configurations (`configs/`), the event-schema definitions and derived
confusable-cluster maps (`data/schema/`), dataset *metadata* including window/event counts and
seen/unseen type lists (`data/**/*.meta.json`), and tests (`tests/`).

**Deliberately excluded — corpus text.** ACE 2005 and Rich ERE are licensed by the
[Linguistic Data Consortium](https://www.ldc.upenn.edu/) and cannot be redistributed. Every
serialized dataset and every generated reasoning trace embeds sentences from those corpora, so no
`.jsonl` data file or trace file is published here. Obtain the corpora from the LDC under your own
license and regenerate them with the scripts below; the schema files, split protocols, and metadata
in this repository fully determine what is produced.

**Also excluded:** model checkpoints and raw run outputs (large, and their prediction dumps quote
corpus text).

## Reproducing the data

1. Obtain ACE 2005 (LDC2006T06) and Rich ERE (LDC2015E29 / LDC2016E31 and related releases) from the
   LDC, and preprocess them with [TextEE](https://github.com/ej0cl6/TextEE), whose sample unit,
   field definitions, and surface-plus-offset scoring convention this work adopts.
2. Build the candidate-conditioned windows. The main regime is oracle mixed-noise top-10 with
   deterministic sample-level shuffling; the seen/unseen partition is `balanced-subtype-v1`, which
   holds out eight Rich ERE subtypes (`Contact:Broadcast`, `Contact:Correspondence`,
   `Justice:Arrest-Jail`, `Justice:Sentence`, `Life:Injure`, `Manufacture:Artifact`,
   `Personnel:Elect`, `Transaction:Transaction`). Each `data/**/*.meta.json` records the exact
   window and event counts a correct rebuild should reproduce.
3. Generate and verify CoT traces with the `scripts/run_e81_trigger_locked_generation_*.sh` family.
   Generation calls an OpenAI-compatible endpoint; set `LLM_BASE_URL` and `LLM_API_KEY` (or `OPENAI_API_KEY`) first.
   The paper's teacher is glm-5.1 and its semantic verifier is deepseek-v4-pro.
4. Train and evaluate with the configs in `configs/generated/`. All CoT runs are full fine-tunes at
   learning rate `2e-6` for 3 epochs with cutoff 1536 and effective batch size 16, warm-started from
   a direct checkpoint; the direct baseline runs 16 epochs at `1e-5` with cutoff 4096. Decoding is
   greedy except for the self-consistency experiments.

## Notes on portability

These scripts were run on a single-node 8×A800 machine and invoke training inside a container, so
many configs and shell scripts contain absolute host paths (model directories, dataset roots, cache
mounts) and container mount specifications from that environment. They are recorded verbatim for
provenance rather than rewritten, and must be adapted to your own layout. Endpoint URLs have been
replaced by the `LLM_BASE_URL` environment variable.

Experiment scripts are named `*_<experiment-id>_<date>.{sh,py}` and correspond to the experiment
identifiers (E77, E80, E81, E84, …) used in the paper's supplementary material.

## Citation

A BibTeX entry will be added once the paper's publication venue is final.

## License

Code in this repository is released for research use. The ACE 2005 and Rich ERE corpora are **not**
covered by this and remain subject to your own LDC license.
