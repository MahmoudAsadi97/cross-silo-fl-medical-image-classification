# Reproducible entry points. On Windows without `make`, run the commands shown
# under each target directly, or use `python tasks.py <target>`.

PY ?= python
PYTHONPATH := src

export PYTHONPATH

.PHONY: help install install-torch fixture verify test smoke dev heterogeneity figures lint demo-site demo-site-fixture clean

help:
	@echo "install        - editable install of the pure-python core"
	@echo "install-torch  - install the optional torch/opacus/dashboard extras"
	@echo "fixture        - (re)generate the tiny committed synthetic fixture"
	@echo "verify         - torch-free correctness checks (aggregation/SCAFFOLD/metrics)"
	@echo "test           - full pytest suite (torch tests auto-skip if torch absent)"
	@echo "smoke          - end-to-end FedAvg on the fixture (needs torch)"
	@echo "dev            - small real-data run on CPU (needs torch + real data)"
	@echo "heterogeneity  - Phase-1 non-IID analysis + figures"
	@echo "figures        - regenerate report figures from experiments/"
	@echo "lint           - ruff (if installed)"
	@echo "demo-site      - open the site with real-data training controls"
	@echo "demo-site-fixture - open the site with real training on the test fixture"

install:
	$(PY) -m pip install -e .

install-torch:
	$(PY) -m pip install -e ".[torch,privacy,dashboard,dev]"

fixture:
	$(PY) scripts/generate_fixture.py

verify:
	$(PY) scripts/verify_core_math.py

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/run_experiment.py --config configs/fedavg.yaml --tier smoke

dev:
	$(PY) scripts/run_experiment.py --config configs/fedavg.yaml --tier dev

heterogeneity:
	$(PY) scripts/analyze_heterogeneity.py

figures:
	$(PY) scripts/make_figures.py

lint:
	-$(PY) -m ruff check src scripts tests

demo-site:
	$(PY) scripts/demo/control_server.py

demo-site-fixture:
	$(PY) scripts/demo/control_server.py --allow-fixture

clean:
	rm -rf experiments/*_smoke_* experiments/heterogeneity **/__pycache__ .pytest_cache
