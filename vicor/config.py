"""Load portable experiment templates, rejecting unset environment variables."""
import os
import re

import yaml


def expand_config(value):
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        unresolved = re.findall(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded)
        if unresolved:
            raise ValueError("Unset configuration environment variable(s): " + ", ".join(unresolved))
        return expanded
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config(item) for key, item in value.items()}
    return value


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    return expand_config(config)
