"""FedProx reduces to FedAvg when mu = 0."""
import pytest

from fl_med.strategies.fedprox import FedProx


def test_extra_loss_zero_when_mu_zero():
    # Short-circuits before any torch call, so runs without torch.
    assert FedProx(mu=0.0).extra_loss(model=None, global_model=None) == 0.0


@pytest.mark.torch
def test_fedprox_mu0_equals_fedavg_client_update():
    """With identical seed/data/model, a FedProx(mu=0) local update must equal a
    FedAvg local update bit-for-bit (regression guarding the fair comparison)."""
    import torch

    from fl_med.engine.client import local_train
    from fl_med.models import build_small_cnn
    from fl_med.seeding import set_seed
    from fl_med.strategies.fedavg import FedAvg
    from fl_med.strategies.fedprox import FedProx as FP

    def make_batch():
        g = torch.Generator().manual_seed(0)
        x = torch.rand(6, 3, 16, 16, generator=g)
        y = torch.randint(0, 8, (6,), generator=g)
        return [{"image": x, "label": y}]

    class Loader(list):
        @property
        def dataset(self):
            return list(range(6))

    def run(strategy, global_model=None):
        set_seed(0)
        model = build_small_cnn()
        gstate = {k: v.clone() for k, v in model.state_dict().items()}
        gm = build_small_cnn() if strategy.needs_global_model else None
        if gm is not None:
            gm.load_state_dict(gstate)
        return local_train(
            client_id=0, model=model, train_loader=Loader(make_batch()),
            device="cpu", strategy=strategy,
            optimizer_cfg={"name": "sgd", "lr": 0.1}, local_epochs=1,
            global_state=gstate, global_model=gm,
        )["state_dict"]

    s_avg = run(FedAvg())
    s_prox = run(FP(mu=0.0))
    for k in s_avg:
        assert torch.allclose(s_avg[k], s_prox[k], atol=1e-6), k
