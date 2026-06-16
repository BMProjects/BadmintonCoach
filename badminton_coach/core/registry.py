"""Component registry + factory.

Backends self-register with @register(kind, name); build() instantiates by name
from a config block. This is the seam that lets any backend be swapped via config
without touching pipeline or upstream code.
"""

from __future__ import annotations

from typing import Any, TypeVar

from .interfaces import INTERFACE_BY_KIND, Component

C = TypeVar("C", bound=Component)

_REGISTRY: dict[str, dict[str, type[Component]]] = {}


def register(kind: str, name: str):
    """Class decorator: register a backend under (kind, name).

    Validates that the class implements the interface declared for `kind`.
    """
    if kind not in INTERFACE_BY_KIND:
        raise ValueError(
            f"Unknown component kind: {kind!r}. Expected one of {list(INTERFACE_BY_KIND)}"
        )

    def decorator(cls: type[C]) -> type[C]:
        expected = INTERFACE_BY_KIND[kind]
        if not issubclass(cls, expected):
            raise TypeError(
                f"{cls.__name__} must subclass {expected.__name__} to register as {kind!r}"
            )
        cls.backend_name = name
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls

    return decorator


def get_backend(kind: str, name: str) -> type[Component]:
    try:
        return _REGISTRY[kind][name]
    except KeyError as exc:
        available = sorted(_REGISTRY.get(kind, {}))
        raise KeyError(
            f"No backend {name!r} registered for kind {kind!r}. Available: {available}. "
            f"Did you import its module? (backends self-register on import)"
        ) from exc


def build(kind: str, config: dict[str, Any]) -> Component:
    """Instantiate the backend named by config['backend'], passing the rest as config."""
    name = config.get("backend")
    if not name:
        raise ValueError(f"Config for {kind!r} is missing required 'backend' key.")
    cls = get_backend(kind, name)
    if not cls.is_available():
        raise RuntimeError(
            f"Backend {kind}:{name} is registered but not available "
            f"(missing dependency, weights, or submodule). See its is_available()."
        )
    return cls(config)


def available_backends(kind: str | None = None) -> dict[str, list[str]]:
    """List registered backend names, optionally filtered to one kind."""
    if kind is not None:
        return {kind: sorted(_REGISTRY.get(kind, {}))}
    return {k: sorted(v) for k, v in _REGISTRY.items()}
