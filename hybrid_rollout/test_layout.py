"""Lightweight checks of backend entry points, shared configuration and assets."""
import importlib
import re
from pathlib import Path

from . import settings


def test_parallel_backend_entrypoints_and_shared_settings():
    root = Path(__file__).parent
    robolab = importlib.import_module('hybrid_rollout.robolab.settings')
    assert (robolab.MODEL, robolab.EFFORT) == (settings.MODEL, settings.EFFORT) == ('gpt-6-astra', 'xhigh')
    robodojo = importlib.import_module('hybrid_rollout.robodojo.settings')
    assert (robodojo.PROVIDER, robodojo.MODEL, robodojo.EFFORT) == ('LiteLLM', 'gpt-6-astra', 'xhigh')
    for backend in ('robolab', 'robodojo'):
        source = (root/backend/'run_local.sh').read_text()
        entrypoints = re.findall(r'-m (hybrid_rollout\.[a-z_0-9\.]+)', source)
        for entry in entrypoints:
            assert entry.startswith(f'hybrid_rollout.{backend}.')
            assert root.parent.joinpath(*entry.split('.')).with_suffix('.py').is_file()
        assert {f'hybrid_rollout.{backend}.pi05_server.server',
                f'hybrid_rollout.{backend}.{backend}_server.server',
                f'hybrid_rollout.{backend}.skill.run'}.issubset(entrypoints)
        for suffix in re.findall(r'/(hybrid_rollout/[^"\s]+/runtime\.sh)', source):
            assert (root.parent/suffix).is_file()
        assert (root/backend/'skill/SKILL.md').is_file()
    for retired_flat_path in ('pi05_server', 'robolab_server', 'skill', 'run_local.sh', 'io.py'):
        assert not (root/retired_flat_path).exists()


def test_both_recorders_find_the_shared_font(monkeypatch):
    monkeypatch.delenv('DEBUG_VIDEO_FONT', raising=False)
    expected = (Path(__file__).parent/'assets/fonts/NotoSansCJKsc-Regular.otf').resolve()
    for backend, simulator in (('robolab', 'robolab_server'), ('robodojo', 'robodojo_server')):
        module = importlib.import_module(f'hybrid_rollout.{backend}.{simulator}.debug_recorder')
        _, actual = module.load_font(16)
        assert Path(actual).resolve() == expected
