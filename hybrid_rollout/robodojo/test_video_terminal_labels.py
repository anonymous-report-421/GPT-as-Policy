import pytest

from .robodojo_server.video_panel import terminal_badge


@pytest.mark.parametrize('result, expected', [
    (dict(complete=True, success=True, terminated=True, truncated=False), 'Task success'),
    (dict(complete=True, success=False, terminated=False, truncated=True), 'Task timeout'),
    (dict(complete=True, success=False, terminated=True, truncated=False), 'Task failed · native'),
    (dict(complete=False, success=False, terminated=False, truncated=False), 'Run ended · incomplete'),
    (dict(complete=False, success=False, terminated=True), 'Run ended · incomplete'),
    (dict(complete=True, success=False), 'Run ended · incomplete'),
    ({}, 'Run ended · incomplete'),
])
def test_native_terminal_labels_do_not_reclassify_incomplete(result, expected):
    before = dict(result)
    assert terminal_badge(result) == expected
    assert result == before
