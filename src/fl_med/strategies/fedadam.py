"""FedAdam (Reddi et al., 2021): an adaptive SERVER optimizer for federated learning.

FedAvg's server step is a plain mean. FedOpt instead treats the aggregated client update
as a *pseudo-gradient* and applies a server-side optimizer to it. FedAdam uses Adam::

    delta_t   = mean_i(w_i) - w_global               # aggregated pseudo-gradient
    m_t       = b1 * m_{t-1} + (1 - b1) * delta_t
    v_t       = b2 * v_{t-1} + (1 - b2) * delta_t^2
    w_global <- w_global + eta * m_hat / (sqrt(v_hat) + tau)

Adaptive per-coordinate step sizes help under heterogeneity; FedOpt variants are reported
to be among the strongest methods on Fed-ISIC2019. Server-side math only, so it reuses the
exact same client loop as every other strategy. With no ``global_state`` supplied it falls
back *exactly* to FedAvg (guarded by a torch-free check in ``verify_core_math``).

Convention (per the paper): FedAdam pairs an SGD ClientOpt with the Adam ServerOpt — see
``configs/fedadam.yaml``. This is the method's defining setup, not a protocol violation.
"""
from __future__ import annotations

from collections import OrderedDict

from .aggregation import weighted_average
from .base import Strategy


class FedAdam(Strategy):
    name = "fedadam"

    def __init__(self, server_lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.99,
                 tau: float = 1e-3, **kwargs) -> None:
        super().__init__(server_lr=server_lr, beta1=beta1, beta2=beta2, tau=tau, **kwargs)
        self.eta = float(server_lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.tau = float(tau)
        self._m = None   # first-moment buffers (per parameter), created lazily
        self._v = None   # second-moment buffers
        self._t = 0      # step count (for bias correction)

    def aggregate(self, client_states, weights, global_state=None):
        agg = weighted_average(client_states, weights)     # FedAvg target = mean of clients
        if global_state is None:
            return agg                                     # fallback == FedAvg
        import torch

        if self._m is None:
            self._m = {k: torch.zeros_like(agg[k].float()) for k in agg}
            self._v = {k: torch.zeros_like(agg[k].float()) for k in agg}
        self._t += 1
        out: "OrderedDict[str, object]" = OrderedDict()
        for k in agg.keys():
            if not torch.is_floating_point(agg[k]):        # integer buffers: pass through
                out[k] = agg[k]
                continue
            delta = agg[k].float() - global_state[k].float()
            self._m[k] = self.b1 * self._m[k] + (1.0 - self.b1) * delta
            self._v[k] = self.b2 * self._v[k] + (1.0 - self.b2) * delta * delta
            m_hat = self._m[k] / (1.0 - self.b1 ** self._t)
            v_hat = self._v[k] / (1.0 - self.b2 ** self._t)
            new = global_state[k].float() + self.eta * m_hat / (torch.sqrt(v_hat) + self.tau)
            out[k] = new.to(global_state[k].dtype)
        return out
