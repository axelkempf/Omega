from __future__ import annotations

from pathlib import Path


def test_stable_int_seed_is_stable_and_bounded() -> None:
    from backtest_engine.optimizer.final_param_selector import _stable_int_seed

    s1 = _stable_int_seed(123, "final_selection", "combo_a")
    s2 = _stable_int_seed(123, "final_selection", "combo_a")
    s3 = _stable_int_seed(123, "final_selection", "combo_b")

    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**32


def test_final_param_selector_does_not_use_global_np_random_for_param_jitter() -> None:
    # Regression test: jitter sampling must be driven by a local RNG to stay
    # reproducible under joblib (instead of using np.random.randint/uniform).
    project_root = Path(__file__).resolve().parent.parent
    path = (
        project_root
        / "src"
        / "backtest_engine"
        / "optimizer"
        / "final_param_selector.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "np.random.randint" not in text
    assert "np.random.uniform" not in text


def test_runner_dev_mode_seeding_is_deterministic_and_prefers_execution_seed() -> None:
    from backtest_engine.core.slippage_and_fee import SlippageModel
    from backtest_engine.runner import _maybe_seed_deterministic_rng

    # When both are present, execution.random_seed must win over reporting.dev_seed.
    cfg_both = {
        "execution": {"random_seed": 111},
        "reporting": {"dev_mode": True, "dev_seed": 222},
    }
    cfg_exec_only = {"execution": {"random_seed": 111}}

    _maybe_seed_deterministic_rng(cfg_both)
    m1 = SlippageModel(fixed_pips=0.0, random_pips=1.0)
    p1 = m1.apply(1.0, "long", pip_size=1.0)

    _maybe_seed_deterministic_rng(cfg_exec_only)
    m2 = SlippageModel(fixed_pips=0.0, random_pips=1.0)
    p2 = m2.apply(1.0, "long", pip_size=1.0)

    _maybe_seed_deterministic_rng(cfg_both)
    m3 = SlippageModel(fixed_pips=0.0, random_pips=1.0)
    p3 = m3.apply(1.0, "long", pip_size=1.0)

    assert abs(p1 - p2) < 1e-12
    assert abs(p1 - p3) < 1e-12


def test_runner_dev_mode_seeding_falls_back_to_reporting_seed() -> None:
    from backtest_engine.core.slippage_and_fee import SlippageModel
    from backtest_engine.runner import _maybe_seed_deterministic_rng

    cfg = {"reporting": {"dev_mode": True, "dev_seed": 123}}

    _maybe_seed_deterministic_rng(cfg)
    m1 = SlippageModel(fixed_pips=0.0, random_pips=1.0)
    p1 = m1.apply(1.0, "short", pip_size=1.0)

    _maybe_seed_deterministic_rng(cfg)
    m2 = SlippageModel(fixed_pips=0.0, random_pips=1.0)
    p2 = m2.apply(1.0, "short", pip_size=1.0)

    assert abs(p1 - p2) < 1e-12
