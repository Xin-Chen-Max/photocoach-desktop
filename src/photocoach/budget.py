"""Session-only request and cost allocation guard; never a billing guarantee.

Owned by the GUI thread. Reserve immediately before dispatch and settle every
result (including errors). Keep the same instance when changing adapters.
Prices and reservations are supplied by the user; no model prices are assumed.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from uuid import uuid4


class BudgetExceeded(RuntimeError):
    pass


def _money(value, name, *, positive=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a finite dollar amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite dollar amount") from exc
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return amount


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass
class _Attempt:
    reservation: Decimal | None
    input_rate: Decimal | None
    output_rate: Decimal | None
    settled: bool = False
    success: bool = False
    cost: Decimal | None = None

    @property
    def allocation(self):
        if self.cost is None:
            return self.reservation
        return max(self.reservation or Decimal(0), self.cost)


class SessionBudget:
    """Hard request cap and conservative user-configured dollar allocations.

    Failed/in-flight requests consume a call and keep their reservations. A
    successful response resolves cost only with both usage counts and explicit
    prices. Dollar overruns block later requests; they cannot undo provider costs.
    """

    _FIELDS = frozenset(("call_limit", "budget_usd", "reserve_usd",
                         "input_usd_per_million", "output_usd_per_million"))

    def __init__(self, call_limit=5, budget_usd=None, reserve_usd=None,
                 input_usd_per_million=None, output_usd_per_million=None):
        self._attempts: dict[str, _Attempt] = {}
        self.configure(call_limit=call_limit, budget_usd=budget_usd,
                       reserve_usd=reserve_usd,
                       input_usd_per_million=input_usd_per_million,
                       output_usd_per_million=output_usd_per_million)

    def configure(self, **settings):
        """Change limits/prices atomically without resetting prior usage."""
        if settings.keys() - self._FIELDS:
            raise ValueError("Unknown budget setting")
        values = {key: settings.get(key, getattr(self, key, None)) for key in self._FIELDS}
        values["call_limit"] = _count(values["call_limit"], "call_limit")
        for key in self._FIELDS - {"call_limit"}:
            values[key] = _money(values[key], key, positive=key == "reserve_usd")
        if values["budget_usd"] is not None and values["reserve_usd"] is None:
            raise ValueError("A dollar cap requires a positive per-call reservation")
        if (values["input_usd_per_million"] is None) != (values["output_usd_per_million"] is None):
            raise ValueError("Provide both input and output prices, or leave both unknown")
        for key, value in values.items():
            setattr(self, key, value)

    @property
    def attempts(self):
        return len(self._attempts)

    @property
    def remaining_calls(self):
        return max(0, self.call_limit - self.attempts)

    @property
    def in_flight(self):
        return sum(not item.settled for item in self._attempts.values())

    @property
    def unresolved(self):
        return sum(item.cost is None for item in self._attempts.values())

    @property
    def allocated_usd(self):
        allocations = [item.allocation for item in self._attempts.values()]
        return None if None in allocations else sum(allocations, Decimal(0))

    @property
    def known_cost_usd(self):
        # Unknown usage is not displayed as $0, including before the first call.
        if not self._attempts or self.unresolved:
            return None
        return sum((item.cost for item in self._attempts.values()), Decimal(0))

    @property
    def blocked_reason(self):
        if self.remaining_calls == 0:
            return "Session request limit reached (failed and in-flight calls count)."
        if self.budget_usd is not None:
            allocation = self.allocated_usd
            if allocation is None:
                return "Earlier session costs are unknown; a dollar cap cannot authorize another request."
            if allocation + self.reserve_usd > self.budget_usd:
                return "Session dollar allocation limit reached. Actual provider charges may differ."
        return None

    def reserve(self):
        """Count a single attempt before network dispatch; never retry implicitly."""
        if self.blocked_reason:
            raise BudgetExceeded(self.blocked_reason)
        ticket = uuid4().hex
        self._attempts[ticket] = _Attempt(self.reserve_usd, self.input_usd_per_million,
                                         self.output_usd_per_million)
        return ticket

    def settle(self, ticket, input_tokens=None, output_tokens=None, success=False):
        """Finish a ticket once; an error or incomplete usage keeps its estimate."""
        if ticket not in self._attempts or self._attempts[ticket].settled:
            raise ValueError("Unknown or already settled budget ticket")
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
            if value is not None:
                _count(value, name)
        item = self._attempts[ticket]
        if success and input_tokens is not None and output_tokens is not None and item.input_rate is not None:
            item.cost = (input_tokens * item.input_rate + output_tokens * item.output_rate) / Decimal(1_000_000)
        item.settled = True
        item.success = success
