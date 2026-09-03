"""Pure contract tests for networked federated-learning telemetry."""
from __future__ import annotations

from fl_med.federated_live import local_examples_seen, local_num_examples


class _Dataset:
    def __len__(self):
        return 18


class _Loader:
    dataset = _Dataset()
    batch_size = 8


def test_processed_examples_are_separate_from_partition_cardinality():
    loader = _Loader()
    assert len(loader.dataset) == 18
    assert local_num_examples(loader, max_batches=1) == 8
    assert local_num_examples(loader, max_batches=None) == 18


def test_examples_seen_sums_all_local_epochs():
    history = [{"num_examples": 8}, {"num_examples": 8}, {"num_examples": 2}]
    assert local_examples_seen(history) == 18
