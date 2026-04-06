"""Custom exceptions for Vault-Synth"""


class VaultSynthError(Exception):
    """Base exception for Vault-Synth"""
    pass


class InvalidDataError(VaultSynthError):
    """Raised when input data is invalid"""
    pass


class InsufficientDataError(VaultSynthError):
    """Raised when insufficient data is provided"""
    pass


class InvalidQubitCountError(VaultSynthError):
    """Raised when qubit count is invalid"""
    pass


class HardwareConstraintError(VaultSynthError):
    """Raised when hardware constraints are not met"""
    pass


class ConvergenceError(VaultSynthError):
    """Raised when training fails to converge"""
    pass


class UntrainedCircuitError(VaultSynthError):
    """Raised when attempting to use an untrained circuit"""
    pass
