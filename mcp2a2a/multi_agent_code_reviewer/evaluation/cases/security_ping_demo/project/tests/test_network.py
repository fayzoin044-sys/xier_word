from types import SimpleNamespace

import network


def test_ping_returns_process_code(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(network.subprocess, "run", fake_run)
    assert network.ping_host("127.0.0.1") == 0
