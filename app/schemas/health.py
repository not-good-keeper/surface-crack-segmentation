"""Typed API models for status checks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CheckOut(BaseModel):
    key: str
    label: str
    state: str
    state_label: str
    detail: str
    action: str = ""
    value: str = ""
    checked_at: str


class StatusOut(BaseModel):
    overall: str
    overall_label: str
    checked_at: str
    checks: list[CheckOut]
    failing: list[str]
    degraded: list[str]
    blocks_inspection: bool
    provider: str
    simulation: str
    simulations: dict[str, Any]
