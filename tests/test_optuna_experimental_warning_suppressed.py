import warnings

import pytest


def test_optuna_constraints_func_experimental_warning_is_suppressed(monkeypatch):
    """Optuna's constraints_func is experimental and can spam logs in worker processes.

    This test ensures we suppress the warning by default (unless explicitly enabled via env var),
    without changing optimizer behavior.
    """

    pytest.importorskip("optuna")

    try:
        from optuna._experimental import ExperimentalWarning
    except Exception:
        pytest.skip("Optuna ExperimentalWarning type not available")

    from optuna.samplers import NSGAIISampler

    from backtest_engine.optimizer.optuna_optimizer import (
        _configure_optuna_experimental_warnings,
    )

    monkeypatch.delenv("OPTUNA_SHOW_EXPERIMENTAL_WARNINGS", raising=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.simplefilter("always", ExperimentalWarning)
        _configure_optuna_experimental_warnings()

        # Trigger the warning (if not filtered) by instantiating NSGAIISampler with constraints_func.
        NSGAIISampler(seed=1, constraints_func=lambda ft: (0.0,))

        assert not any(
            isinstance(w.message, ExperimentalWarning)
            and "constraints_func" in str(w.message)
            for w in recorded
        )
