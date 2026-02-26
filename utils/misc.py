from copy import deepcopy

import yaml
import re
import sys
try:
    import pympler.asizeof as asizeof
except ImportError:
    asizeof = None
import time
import inspect
import functools
from types import SimpleNamespace


def lift_and_remove_nested_keys(d: dict) -> dict:
    """Flatten a nested dictionary by lifting all inner key-value pairs to the top level.

    A deep copy is made first so the original dict is not mutated.
    In case of key collisions, the most deeply nested value wins.

    Args:
        d: The dictionary to flatten.

    Returns:
        A new, flat dictionary.
    """
    d = deepcopy(d)
    all_dicts = []
    queue = [d]
    visited_ids = {id(d)}

    while queue:
        current_dict = queue.pop(0)
        all_dicts.append(current_dict)
        for value in current_dict.values():
            if isinstance(value, dict) and id(value) not in visited_ids:
                visited_ids.add(id(value))
                queue.append(value)

    for nested_dict in reversed(all_dicts):
        if nested_dict is d:
            continue
        keys_to_remove = []
        for key, value in nested_dict.items():
            d[key] = value
            keys_to_remove.append(key)
        for key in keys_to_remove:
            del nested_dict[key]

    return d


def load_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_dict_as_yaml(data: dict, filename: str):
    """
    Save a Python dictionary as a YAML config file.

    Args:
        data (dict): The dictionary to save.
        filename (str): The target YAML file path.
    """
    with open(filename, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)

def dict_to_filename(input_dict):
    """Converts a dictionary to a filename-friendly string."""
    items = []
    for key, value in input_dict.items():
        # Sanitize key and value to remove invalid characters
        sanitized_key = re.sub(r'[^\w\-]', '_', str(key))
        sanitized_value = re.sub(r'[^\w\-]', '_', str(value))
        items.append(f"{sanitized_key}-{sanitized_value}")
    return "_".join(items)


def get_formatted_size_kb(obj, method='deep'):
    """
    Calculates the size of an object and returns it formatted in KB.

    Args:
        obj: The Python object to measure.
        method: 'deep' (uses pympler.asizeof) or 'shallow' (uses sys.getsizeof).
                'deep' is generally recommended for total footprint.

    Returns:
        A string representing the size in KB (e.g., "15.23 KB").
    """
    if method == 'deep':
        try:
            size_bytes = asizeof.asizeof(obj)
        except NameError:
            return "Error: Pympler not installed or imported correctly."
        except Exception as e:
            return f"Error during deep size calculation: {e}"
    elif method == 'shallow':
        size_bytes = sys.getsizeof(obj)
    else:
        raise ValueError("Method must be 'deep' or 'shallow'")

    # Convert bytes to Kilobytes (1 KB = 1024 bytes)
    size_kb = size_bytes / 1024.0

    # Format to 2 decimal places
    return f"{size_kb:.2f} KB"


def timer(_func=None, *, label=None, logger=print, unit="ms", threshold=None, store=False):
    """
    Time function calls.

    Parameters (all optional, use as @timer or @timer(...)):
      label:      Custom name in log (default: func.__qualname__)
      logger:     Callable to receive the message (default: print)
                  e.g., logger=logging.getLogger(__name__).info
      unit:       's' | 'ms' | 'us' | 'ns'  (default: 'ms')
      threshold:  Only log if elapsed >= threshold (same unit as 'unit')
      store:      Keep running stats on wrapper.timer_stats

    Works for sync and async callables. Return value is preserved.
    """
    factors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if unit not in factors:
        raise ValueError(f"unit must be one of {list(factors)}")

    def _decorate(func):
        name = label or func.__qualname__
        factor = factors[unit]

        stats = SimpleNamespace(count=0, total=0.0, min=float("inf"), max=0.0)
        def _update_stats(elapsed):
            stats.count += 1
            stats.total += elapsed
            stats.min = elapsed if stats.count == 1 else min(stats.min, elapsed)
            stats.max = max(stats.max, elapsed)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    dt = (time.perf_counter() - t0) * factor
                    if store: _update_stats(dt)
                    if threshold is None or dt >= threshold:
                        logger(f"[{name}] {dt:.3f} {unit}")
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    dt = (time.perf_counter() - t0) * factor
                    if store: _update_stats(dt)
                    if threshold is None or dt >= threshold:
                        logger(f"[{name}] {dt:.3f} {unit}")

        # expose stats and convenience properties if requested
        wrapper.timer_stats = stats if store else None
        if store:
            wrapper.mean_time = lambda: (stats.total / stats.count) if stats.count else 0.0
        return wrapper

    return _decorate if _func is None else _decorate(_func)
