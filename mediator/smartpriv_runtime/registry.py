from __future__ import annotations

from typing import Any, Dict, Type

from .base import Operator, OperatorError

_REGISTRY: Dict[str, Type[Operator]] = {}


def register(cls: Type[Operator]) -> Type[Operator]:
    _REGISTRY[cls.operator_id] = cls
    return cls


def get_operator_class(operator_id: str) -> Type[Operator]:
    # Import lazily to populate the registry.
    from . import operators  # noqa: F401

    if operator_id not in _REGISTRY:
        raise OperatorError(f"No executable operator registered for {operator_id}")
    return _REGISTRY[operator_id]


def make_operator(operator_id: str, **params: Any) -> Operator:
    return get_operator_class(operator_id)(**params)


def registered_operator_ids() -> list[str]:
    from . import operators  # noqa: F401

    return sorted(_REGISTRY)
