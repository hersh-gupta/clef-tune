import importlib.util

_TRAINING_STACK = ("torch", "torchvision", "transformers", "peft")

# The base install has no torch: skip the torch tests instead of failing to import them.
collect_ignore = [] if all(importlib.util.find_spec(m) for m in _TRAINING_STACK) else ["torch"]
