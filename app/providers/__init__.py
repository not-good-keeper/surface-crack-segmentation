"""Provider selection.

``INSPECTION_PROVIDER=mock`` (the default) never touches the real pipeline; the real
provider module is not even imported.  ``INSPECTION_PROVIDER=real`` loads
``RealInspectionProvider``, which verifies the model hash and constructs
``app.inference.Inspector``.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.providers.base import (  # re-exported for convenience
    ALL_STATUSES,
    CLASS_BY_CHANNEL,
    STATUS_ACQUISITION_FAILURE,
    STATUS_CLEAN,
    STATUS_PROCESSING_FAILURE,
    STATUS_REGIONS_FOUND,
    InspectionProvider,
    InspectionResult,
    ProviderUnavailable,
    RegionRecord,
)

__all__ = [
    "ALL_STATUSES",
    "CLASS_BY_CHANNEL",
    "STATUS_ACQUISITION_FAILURE",
    "STATUS_CLEAN",
    "STATUS_PROCESSING_FAILURE",
    "STATUS_REGIONS_FOUND",
    "InspectionProvider",
    "InspectionResult",
    "ProviderUnavailable",
    "RegionRecord",
    "build_provider",
    "get_provider",
    "provider_error",
    "reset_provider",
]

_provider: Any = None
_provider_error: ProviderUnavailable | None = None


def build_provider(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    if settings.is_mock:
        from app.providers.mock_provider import MockInspectionProvider

        return MockInspectionProvider(settings)

    from app.providers.real_provider import RealInspectionProvider

    return RealInspectionProvider(settings)


def get_provider(settings: Settings | None = None) -> Any:
    """Process-wide provider.

    A real-mode failure is cached, not raised on every request: the Status screen has
    to be able to explain the problem, and inspection stays blocked until it is fixed.
    """
    global _provider, _provider_error
    if _provider is not None:
        return _provider
    if _provider_error is not None:
        raise _provider_error
    try:
        _provider = build_provider(settings)
    except ProviderUnavailable as exc:
        _provider_error = exc
        raise
    return _provider


def provider_error() -> ProviderUnavailable | None:
    return _provider_error


def reset_provider() -> None:
    global _provider, _provider_error
    _provider = None
    _provider_error = None
