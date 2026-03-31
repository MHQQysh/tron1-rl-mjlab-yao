# tron1-rl-mjlab

## Installation

```shell
uv sync
```

## Usage

### Training

```shell
uv run python scripts/rsl_rl/train.py Mjlab-WF-Tron
```

### Evaluation

```shell
uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron --wandb-run-path <project>/mjlab_wf_tron # --viewer viser
```


uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron --wandb-run-path `<project>`/mjlab_wf_tron # --viewer viser



uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron --checkpoint logs/rsl_rl/wf_tron/2026-03-31_16-38-57/model_800.pt --viewer viser
