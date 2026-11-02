"""Adapter interfaces and model-specific implementations.

Importing this package is intentionally dependency-light.  The generic
evaluation driver loads a requested implementation lazily, so an adapter for
another framework does not import ARIADNE or PyTorch.
"""

__all__ = []
