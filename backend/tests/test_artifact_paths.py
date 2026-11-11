"""Regression tests for working-directory-independent model loading."""

import importlib.util
from pathlib import Path
import builtins
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1] / "main.py"


def test_artifacts_resolved_relative_to_backend(tmp_path, monkeypatch):
    """Launching the API from another directory must load backend artifacts."""
    monkeypatch.chdir(tmp_path)
    model_paths = []

    def fake_load(path):
        model_paths.append(Path(path))
        return object()

    original_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if Path(path) == BACKEND.parent / "metrics.json":
            return original_open(tmp_path / "metrics_fixture.json", *args, **kwargs)
        return original_open(path, *args, **kwargs)

    (tmp_path / "metrics_fixture.json").write_text('{"accuracy": 0.9}')
    spec = importlib.util.spec_from_file_location("spam_api_path_test", BACKEND)
    module = importlib.util.module_from_spec(spec)
    with patch("joblib.load", side_effect=fake_load), patch("builtins.open", side_effect=fake_open):
        spec.loader.exec_module(module)

    assert model_paths == [BACKEND.parent / "model.joblib", BACKEND.parent / "vectorizer.joblib"]
    assert module.METRICS == {"accuracy": 0.9}
    assert module.model is not None
    assert module.vectorizer is not None
