"""Budget tracker — daily cap enforcement, daily rollover."""
from pathlib import Path

from ai_employee.config import BudgetConfig, PricingConfig
from ai_employee.heartbeat.budget import BudgetTracker


def _make_tracker(tmp_path: Path, cap: float = 1.00,
                  input_rate: float = 1.0, output_rate: float = 2.0) -> BudgetTracker:
    config = BudgetConfig(
        daily_usd_cap=cap,
        pricing=PricingConfig(
            input_per_1m_usd=input_rate,
            output_per_1m_usd=output_rate,
        ),
    )
    return BudgetTracker(tmp_path / "budget.json", config)


def test_initial_status_is_zero(tmp_path):
    s = _make_tracker(tmp_path).status()
    assert s.input_tokens == 0
    assert s.output_tokens == 0
    assert s.spend_usd == 0.0


def test_record_accumulates(tmp_path):
    t = _make_tracker(tmp_path)
    t.record(500_000, 100_000)
    s = t.status()
    assert s.input_tokens == 500_000
    assert s.output_tokens == 100_000
    # 0.5M × $1/M + 0.1M × $2/M = $0.50 + $0.20 = $0.70
    assert abs(s.spend_usd - 0.70) < 1e-6


def test_exceeded_below_cap(tmp_path):
    t = _make_tracker(tmp_path, cap=1.00)
    t.record(100_000, 100_000)  # $0.10 + $0.20 = $0.30
    assert not t.exceeded()


def test_exceeded_at_cap(tmp_path):
    t = _make_tracker(tmp_path, cap=0.50)
    t.record(300_000, 100_000)  # $0.30 + $0.20 = $0.50
    assert t.exceeded()


def test_zero_cap_means_unlimited(tmp_path):
    t = _make_tracker(tmp_path, cap=0.0)
    t.record(1_000_000_000, 1_000_000_000)
    assert not t.exceeded()
