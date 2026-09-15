"""Offline private proxy tests; no real credentials or network requests."""
import urllib.error
import pytest

from . import proxy_config


def test_imports_commented_literal_without_executing_rc(tmp_path, capsys):
    source=tmp_path/'zshrc'; destination=tmp_path/'proxy.url'
    value='http://test-user:test-password@proxy.example:3128'
    source.write_text(f'# export HTTP_PROXY="{value}"\n# export HTTPS_PROXY="{value}"\n'
                      'export HTTP_PROXY="http://127.0.0.1:7890"\nexit 99\n')
    proxy_config.import_zshrc(source,destination)
    assert proxy_config.read_proxy(destination)==value
    assert destination.stat().st_mode & 0o077==0
    assert 'test-password' not in capsys.readouterr().out
    proxy_config.import_zshrc(source,destination)
    destination.chmod(0o644)
    with pytest.raises(ValueError,match='0600'):
        proxy_config.read_proxy(destination)


def test_proxy_failure_does_not_leak_connection_credentials(monkeypatch):
    class Fake:
        def open(self,*args,**kwargs):
            raise urllib.error.URLError('407 http://test-user:test-password@proxy.example:3128')
    monkeypatch.setattr(proxy_config.urllib.request,'build_opener',lambda *args:Fake())
    with pytest.raises(RuntimeError) as error:
        proxy_config.probe('http://test-user:test-password@proxy.example:3128')
    assert 'test-password' not in str(error.value)
    assert 'proxy_authentication_required' in str(error.value)


def test_proxy_rejects_symlink(tmp_path):
    original=tmp_path/'file'; original.write_text('http://proxy.example:3128'); original.chmod(0o600)
    link=tmp_path/'link'; link.symlink_to(original)
    with pytest.raises(ValueError,match='symlink'):
        proxy_config.read_proxy(link)
