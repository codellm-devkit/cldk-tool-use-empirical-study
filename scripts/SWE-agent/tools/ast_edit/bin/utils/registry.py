# utils/registry.py

_registry = {}

def get(key, default=None):
    return _registry.get(key, default)

def set(key, value):
    _registry[key] = value

registry = type("SimpleRegistry", (), {"get": get, "__setitem__": set})
