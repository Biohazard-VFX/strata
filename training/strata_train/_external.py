"""Typed boundary for third-party callables whose upstream annotations are incomplete."""


def resolve(module: object, name: str) -> object:
    return getattr(module, name)
