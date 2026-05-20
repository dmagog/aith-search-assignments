# Colab run instructions for HA5

Use this when running the generative SLM part on GPU.

## Files

- Notebook: `notebooks/ha5_colab_llm_inference.ipynb`
- Input archive to upload in Colab: `colab_bundle/dz5_colab_input.zip`

## Full run

1. Open the notebook in Google Colab.
2. Use `Runtime -> Disconnect and delete runtime` if you previously ran an install cell that upgraded `numpy`, `torch`, or CUDA packages.
3. Set runtime to GPU.
4. Run cells from top to bottom.
5. In the configuration cell keep the full-run guard enabled:

```python
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LIMIT = None
MIN_EXPECTED_ROWS = 1000
BATCH_SIZE = 8
MAX_NEW_TOKENS = 16
CONTEXT_BUDGET_CHARS = 6000
RUN_GROUP = "core_top5"
USE_4BIT = True
USE_GOOGLE_DRIVE_CHECKPOINTS = True
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/dz5_colab_checkpoints"
```

6. When prompted, upload `colab_bundle/dz5_colab_input.zip`.
7. The completion-check cell must print `OK` for every experiment before the final download cell runs.
8. Download `dz5_colab_outputs.zip` at the end.

The install cell intentionally uses `--no-deps` and does not reinstall `numpy`, `torch`, `pandas`, `requests`, or CUDA packages. Colab's base stack should stay intact.

The default notebook is optimized for Colab interruptions:

- `Qwen/Qwen2.5-1.5B-Instruct` is the fast default; `Qwen/Qwen2.5-3B-Instruct` is a slower quality variant.
- `RUN_GROUP = "core_top5"` runs the main generative comparison: closed-book, oracle, top-5 mixture, top-5 dense, and MIRAGE mixed.
- After that finishes, set `RUN_GROUP = "top1_optional"` and rerun if we decide to add generative top-1 results. Checkpoints are cumulative.
- If GPU memory fails, reduce `BATCH_SIZE` from `8` to `4`.

## Deliberate smoke run

For a short sanity check, change both values together:

```python
LIMIT = 50
MIN_EXPECTED_ROWS = 50
```

Do not keep `MIN_EXPECTED_ROWS = 50` for the final run. With the full-run guard, the notebook refuses to download outputs when fewer than the expected number of unique `query_id`s is present in any prediction file.

## Resume behavior

If Colab disconnects, rerun from the setup cells; the experiment cell appends to existing JSONL files and skips already completed `query_id`s.

The notebook checkpoints to Google Drive after every batch. If the runtime is reset completely, rerun the notebook with the same `DRIVE_CHECKPOINT_DIR`; it restores `predictions/*.jsonl`, compacts broken/duplicate rows, and continues from missing questions.

The final download cell is intentionally guarded. It raises an error unless the completion-check cell has run and confirmed that all configured examples are complete.

## What to return locally

Return the downloaded `dz5_colab_outputs.zip`. It should contain:

- `run_manifest.json`
- `predictions/*.jsonl`
- `metrics/*.json`
- `tables/qa_results_summary.csv`

These files can be copied into local `dz5/artifacts/` for report building.

`run_manifest.json` records the actual `LIMIT`, number of loaded rows, expected rows per experiment, model settings, and per-file prediction counts. Use it to diagnose accidental short runs.

## Notes

- The default model is Qwen because it is not gated and stays under the `<=4B` limit.
- To use `google/gemma-2-2b-it`, set `MODEL_ID` accordingly and add `HF_TOKEN` in Colab secrets.
- Keep `MAX_NEW_TOKENS` low (`32` or lower), matching the course discussion about compute cost.
- For full runs, keep Drive checkpointing enabled. Do not delete `MyDrive/dz5_colab_checkpoints` until the final output zip has been returned locally.
