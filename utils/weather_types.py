"""Provider-agnostic weather types shared across all weather client modules."""


class LocationNotFoundError(RuntimeError):
    """Raised when a location string cannot be resolved to coordinates."""

    pass
