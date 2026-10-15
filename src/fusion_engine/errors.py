"""Actionable errors that the CLI can report without tracebacks or credentials."""


class FusionError(Exception):
    """Base error for validation, adapters and acceptance failures."""


class ConfigError(FusionError):
    """The task configuration or its assets violate the contract."""


class AdapterError(FusionError):
    """A provider adapter cannot perform the requested operation."""


class AcceptanceError(FusionError):
    """A run cannot pass the configured machine or human gates."""
