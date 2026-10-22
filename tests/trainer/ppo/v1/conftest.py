"""Keep CPU TransferQueue integration tests off any existing training cluster."""

import tempfile

import pytest
import ray


@pytest.fixture(scope="session")
def isolated_tq_ray_cluster():
    if ray.is_initialized():
        pytest.fail(
            "TransferQueue integration tests require a dedicated pytest process, disconnected from training Ray"
        )
    # TQ uses a fixed actor namespace/name, so separate test partitions alone
    # cannot isolate its init()/close() lifecycle from a production controller.
    # A short, separate temp root also preserves /tmp/ray's training-cluster
    # discovery files and stays within Unix socket path length limits.
    with tempfile.TemporaryDirectory(prefix="verl-tq-", dir="/tmp") as directory:
        try:
            ray.init(
                address="local",
                _temp_dir=directory,
                include_dashboard=False,
                num_cpus=16,
                num_gpus=0,
                resources={"NPU": 0},
                object_store_memory=256 * 1024**2,
            )
            yield
        finally:
            ray.shutdown()
