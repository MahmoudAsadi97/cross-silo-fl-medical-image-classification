"""End-to-end smoke: one federated run on a freshly-generated tiny fixture."""
import pytest


@pytest.mark.torch
def test_federated_smoke_runs(tmp_path):
    from fl_med.data.fixture import generate_fixture
    from fl_med.data.loaders import build_client_dataloaders, list_clients
    from fl_med.models import model_builder_from_config
    from fl_med.seeding import set_seed
    from fl_med.strategies import build_strategy
    from fl_med.engine.server import run_federated

    raw = tmp_path / "raw"
    generate_fixture(raw)

    config = {
        "_meta": {"tier": "smoke"},
        "seed": 0,
        "data": {"root": str(raw), "image_size": 32, "batch_size": 8, "num_workers": 0},
        "model": {"name": "small_cnn", "num_classes": 8, "norm": "group"},
        "optimizer": {"name": "adam", "lr": 1e-3},
        "federated": {"rounds": 2, "local_epochs": 1, "max_batches": 2},
        "strategy": {"name": "fedavg"},
    }
    set_seed(0)
    clients = list_clients(config)
    assert len(clients) >= 2

    _, test_loader = build_client_dataloaders(config, clients[0])
    result = run_federated(
        config=config,
        model_builder=model_builder_from_config(config),
        strategy=build_strategy(config),
        client_ids=clients,
        client_loader_fn=lambda cid: build_client_dataloaders(config, cid)[0],
        test_loader=test_loader,
        device="cpu",
    )
    assert len(result["history"]) == 2
    bal = result["final_metrics"]["test_balanced_accuracy"]
    assert 0.0 <= bal <= 1.0
