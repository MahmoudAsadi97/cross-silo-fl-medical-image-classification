from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config

cfg = load_yaml_config(PROJECT_ROOT / "configs/experiments/base_experiment.yaml")
print("Loaded config successfully.")
print("Project:", cfg["project"]["name"])
print("Dataset:", cfg["project"]["dataset"])
print("Model:", cfg["model"]["name"])
print("Clients:", cfg["federated"]["num_clients"])
