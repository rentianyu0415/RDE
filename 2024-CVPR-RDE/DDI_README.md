# RDE Dual-grained Disagreement Interaction

This extension keeps the trained RDE model frozen and adds the DDI interaction
loop described in `../main.pdf`. It runs the 200-query RSTPReid experiment,
three DDI rounds, and the joint-Top-5 ablation.

## API configuration

Use an OpenAI-compatible endpoint that provides `qwen3.6-flash`. The runner
first tries the pinned `qwen3.6-flash-2026-04-16` model and falls back to the
rolling alias only when the endpoint reports that the pinned model is missing.

```bash
conda activate rde
export DASHSCOPE_API_KEY='...'
export DASHSCOPE_BASE_URL='https://YOUR_ENDPOINT/compatible-mode/v1'
```

The key is read only from the environment and is never written to experiment
files. Model responses, usage metadata, prompt hashes, and image-derived cache
keys are stored under the selected output directory so an interrupted run can
resume without repeating successful API calls.

## Run the complete experiment

```bash
cd /root/RDE/2024-CVPR-RDE
python run_ddi_experiment.py --mode all
```

Useful alternatives:

```bash
# Three-round DDI main experiment only
python run_ddi_experiment.py --mode main

# One-round DDI plus the joint-Top-5 ablation
python run_ddi_experiment.py --mode ablation

# Use a separate output directory for another endpoint or model
python run_ddi_experiment.py --mode all --output-dir ddi_outputs/another_run
```

The default checkpoint is the existing clean RSTPReid `best.pth`. Override
`--config-file`, `--checkpoint`, or `--root-dir` when needed. The experiment
lock rejects attempts to mix a different endpoint, model, checkpoint, prompt,
or seed into an existing output directory.

The default output directory is `ddi_outputs/rstpreid_200q_qwen36_flash`, so
the existing `qwen3-vl-flash` experiment under `ddi_outputs/rstpreid_200q` is
preserved.

## Outputs

- `manifest.json`: fixed 200-query sample, gallery, model and checkpoint lock
- `gallery_features.pt`: frozen BGE/TSE gallery cache
- `qwen_cache.json`: resumable, credential-free API response cache
- `main_trajectories.jsonl` and `ablation_trajectories.jsonl`
- `main_summary.json`, `ablation_summary.json`, and `round_metrics.csv`
- `table4.md`, `table5.md`, and `qualitative.md`

The preflight performs one four-image question request and one source-image
answer request before the full run. Do not use `--skip-preflight` for a new API
endpoint.

Before contacting the API, the runner also evaluates all 2,000 standard test
captions and checks the existing checkpoint against its recorded RDE metrics.
`--skip-baseline-check` is intended only for diagnostics or a deliberately
different checkpoint.

## Tests

```bash
conda activate rde
cd /root/RDE/2024-CVPR-RDE
python -m unittest discover -s tests -p 'test_ddi*.py'
```
