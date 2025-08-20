import sys
import types
import runpy
import importlib

def test_uvicorn_run_called(monkeypatch):
    called = {}

    def fake_run(app, host, port, reload):
        called["app"] = app
        called["host"] = host
        called["port"] = port
        called["reload"] = reload

    fake_uvicorn = types.SimpleNamespace(run=fake_run)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    # Run the module as if it was executed as __main__
    runpy.run_module("scripts.run_api", run_name="__main__")

    assert called["app"] == "src.service.api:app"
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000
    assert called["reload"] is False

def test_import_does_not_run(monkeypatch):
    called = {}

    def fake_run(*args, **kwargs):
        called["was_called"] = True

    fake_uvicorn = types.SimpleNamespace(run=fake_run)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    # Reload module normally (not as __main__)
    run_api = importlib.reload(importlib.import_module("scripts.run_api"))

    # Should not auto-run uvicorn
    assert "was_called" not in called
