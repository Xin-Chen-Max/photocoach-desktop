from decimal import Decimal

import pytest

from photocoach.budget import BudgetExceeded, SessionBudget


def test_failed_and_inflight_attempts_count_toward_hard_limit():
    budget = SessionBudget(call_limit=2)
    failed = budget.reserve()
    budget.settle(failed, success=False)
    budget.reserve()
    assert budget.attempts == 2
    assert budget.in_flight == 1
    assert budget.unresolved == 2
    assert budget.known_cost_usd is None
    assert budget.allocated_usd is None
    with pytest.raises(BudgetExceeded, match="request limit"):
        budget.reserve()
    assert budget.attempts == 2


def test_exact_dollar_boundary_and_failure_keep_allocations():
    budget = SessionBudget(budget_usd="0.10", reserve_usd="0.05")
    a, b = budget.reserve(), budget.reserve()
    budget.settle(a, success=False)
    budget.settle(b, input_tokens=10, output_tokens=3, success=True)
    assert budget.allocated_usd == Decimal("0.10")
    assert budget.known_cost_usd is None
    with pytest.raises(BudgetExceeded, match="dollar allocation"):
        budget.reserve()


def test_actual_usage_below_reservation_remains_conservative():
    budget = SessionBudget(reserve_usd="0.10", input_usd_per_million="2", output_usd_per_million="4")
    ticket = budget.reserve()
    assert budget.known_cost_usd is None
    budget.settle(ticket, input_tokens=1000, output_tokens=500, success=True)
    assert budget.known_cost_usd == Decimal("0.004")
    assert budget.allocated_usd == Decimal("0.10")
    assert budget.unresolved == 0
    assert budget.in_flight == 0


def test_overrun_blocks_future_sends_without_claiming_billing_ceiling():
    budget = SessionBudget(budget_usd="0.10", reserve_usd="0.05",
                           input_usd_per_million="2", output_usd_per_million="4")
    ticket = budget.reserve()
    budget.settle(ticket, input_tokens=100000, output_tokens=100000, success=True)
    assert budget.allocated_usd == Decimal("0.6")
    assert budget.known_cost_usd == Decimal("0.6")
    with pytest.raises(BudgetExceeded, match="provider charges may differ"):
        budget.reserve()


def test_config_changes_preserve_calls_and_snapshot_old_prices_and_reservations():
    budget = SessionBudget(call_limit=2, reserve_usd="0.05",
                           input_usd_per_million=2, output_usd_per_million=4)
    ticket = budget.reserve()
    budget.configure(input_usd_per_million=200, output_usd_per_million=400, reserve_usd="0.10")
    budget.settle(ticket, input_tokens=1000, output_tokens=500, success=True)
    assert budget.known_cost_usd == Decimal("0.004")
    assert budget.allocated_usd == Decimal("0.05")
    assert budget.remaining_calls == 1
    budget.configure(call_limit=1)
    with pytest.raises(BudgetExceeded):
        budget.reserve()
    budget.configure(call_limit=3)
    budget.reserve()
    assert budget.attempts == 2


def test_enabling_dollar_cap_after_unknown_attempt_blocks_new_requests():
    budget = SessionBudget()
    ticket = budget.reserve()
    budget.settle(ticket, success=False)
    budget.configure(budget_usd="1", reserve_usd="0.1")
    with pytest.raises(BudgetExceeded, match="Earlier session costs are unknown"):
        budget.reserve()


def test_unknown_usage_and_unpriced_attempts_stay_unknown_after_configuration():
    budget = SessionBudget(reserve_usd="0.1")
    ticket = budget.reserve()
    budget.configure(input_usd_per_million=2, output_usd_per_million=4)
    budget.settle(ticket, input_tokens=1, output_tokens=1, success=True)
    assert budget.known_cost_usd is None
    assert budget.unresolved == 1
    ticket = budget.reserve()
    budget.settle(ticket, input_tokens=1, success=True)
    assert budget.unresolved == 2


def test_explicit_free_prices_with_reported_usage_are_zero_not_unknown():
    budget = SessionBudget(input_usd_per_million=0, output_usd_per_million=0)
    assert budget.known_cost_usd is None
    ticket = budget.reserve()
    budget.settle(ticket, input_tokens=1, output_tokens=1, success=True)
    assert budget.known_cost_usd == Decimal(0)
    assert budget.allocated_usd == Decimal(0)


@pytest.mark.parametrize("settings", [
    {"call_limit": -1}, {"call_limit": True}, {"call_limit": 1.5},
    {"budget_usd": "-1"}, {"budget_usd": True}, {"budget_usd": "nan"},
    {"reserve_usd": 0}, {"reserve_usd": float("inf")},
    {"reserve_usd": "-1"}, {"input_usd_per_million": -2, "output_usd_per_million": 1},
    {"input_usd_per_million": 1}, {"budget_usd": 1}, {"unknown": 1},
])
def test_bad_settings_do_not_mutate_existing_ledger(settings):
    budget = SessionBudget(call_limit=3)
    budget.reserve()
    with pytest.raises(ValueError):
        budget.configure(**settings)
    assert budget.call_limit == 3
    assert budget.attempts == 1
    assert budget.budget_usd is None


@pytest.mark.parametrize("tokens", [-1, True, 1.5, float("nan"), "2"])
def test_invalid_usage_does_not_settle_ticket(tokens):
    budget = SessionBudget()
    ticket = budget.reserve()
    with pytest.raises(ValueError):
        budget.settle(ticket, input_tokens=tokens, output_tokens=2, success=True)
    assert budget.in_flight == 1


def test_unknown_duplicate_ticket_and_invalid_success_rejected():
    budget = SessionBudget()
    with pytest.raises(ValueError):
        budget.settle("missing")
    ticket = budget.reserve()
    with pytest.raises(ValueError):
        budget.settle(ticket, success=1)
    budget.settle(ticket)
    with pytest.raises(ValueError):
        budget.settle(ticket)
    assert budget.attempts == 1


@pytest.mark.parametrize("settings", [{"call_limit": 0}, {"budget_usd": 0, "reserve_usd": "0.1"}])
def test_zero_caps_disable_dispatch(settings):
    budget = SessionBudget(**settings)
    with pytest.raises(BudgetExceeded):
        budget.reserve()
    assert budget.attempts == 0
