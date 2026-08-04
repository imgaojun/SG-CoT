# Stage2 Benchmark Datasets

This directory stores small SFT-style benchmark slices for stage-2 model and throughput testing with `LLaMA-Factory`.

## Expected Format

Each benchmark file should be JSONL.

Each line should contain:

- `instruction`
- `input`
- `output`

Example:

```json
{"instruction":"Extract events from the text using only the candidate event types and output JSON.","input":"Text: ...\\nCandidate types: ...\\nSchema cards: ...","output":"{\"events\": [...]}"} 
```

## Registration

After placing a benchmark JSONL file here, register it for `LLaMA-Factory` with:

```bash
python src/stage2_benchmark/register_llamafactory_dataset.py \
  --slice_key richere_balanced_split1_predicted_top5
```

This updates:

- `data/stage2_benchmark_datasets/dataset_info.json`

## Current Planned Slices

- `richere_balanced_subtype_v1_split1_predicted_top5_bench.jsonl`
- `ace05_balanced_subtype_v1_split1_predicted_top5_bench.jsonl`
