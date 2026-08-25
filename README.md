## Environment

The reference environment used Python 3.12.7 with CPU-based PyTorch. Install the required packages with:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Linux and macOS users should activate the environment with `source .venv/bin/activate`.

## Run the main comparison

The default command trains and evaluates UGQP-GNN, frozen DTODRL, jointly trained DTODRL, and PPO-MLP on seeds 2024-2028:

```bash
python run_main_comparison.py --output outputs/main_comparison
```

Use `--resume` to reuse completed model and evaluation files. The default protocol uses 60,000 training interactions and 400 evaluation episodes per seed.

## Other experiments

```bash
python run_queue_control.py --output outputs/queue_observation_control
python run_ablation.py --output outputs/load_stratified_ablation
python reproduce_tables.py --data ../data
```

The scripts in `experiments/` reproduce the utility, sensitivity, workflow-scale, graph-density, scheduler-overhead, architecture-scale, domain-workflow, and prototype measurements. Prototype serving scripts require a local Qwen3 model and the additional packages listed in `requirements-prototype.txt`.

The DTODRL pretraining checkpoint is stored in `models/dtodrl_gat_pretrained.pt`. Model and data paths are resolved from the command-line arguments or the repository-relative defaults.
