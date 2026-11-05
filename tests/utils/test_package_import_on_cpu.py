"""CPU reward imports must not load optional device-transfer dependencies."""

import os
import subprocess
import sys
from pathlib import Path


def test_reward_import_without_transfer_queue_or_mooncake():
    source = """
import importlib.abc
import os
import sys

class BlockTransport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'transfer_queue', 'mooncake'}:
            raise AssertionError('CPU reward import loaded transport: ' + fullname)
        return None

sys.meta_path.insert(0, BlockTransport())
import verl
assert 'DataProto' in dir(verl)
assert 'DataProto' not in vars(verl)
assert verl.__version__
from verl.experimental.trajectory_async.reasoning_math_reward import compute_score
assert callable(compute_score)
assert 'verl.protocol' not in sys.modules
assert not any(name.startswith(('transfer_queue', 'mooncake')) for name in sys.modules)
assert 'lazy_dataproto_bootstrap' not in sys.modules
assert os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] == '0'
try:
    verl.this_attribute_does_not_exist
except AttributeError:
    pass
else:
    raise AssertionError('Unknown package attribute was accepted')
"""
    environment = dict(os.environ)
    for name in ("LD_PRELOAD", "VERL_USE_EXTERNAL_MODULES", "VERL_LAZY_DATAPROTO_CPU_PROBE",
                 "VERL_BLOCK_NPU_TRANSPORT_IMPORT", "RAY_ENABLE_UV_RUN_RUNTIME_ENV"):
        environment.pop(name, None)
    environment.update(VERL_USE_EXTERNAL_PLUGINS="none", ASCEND_RT_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[2],
                            env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
