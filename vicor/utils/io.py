import json
import yaml
import os
import pickle
from typing import Any, Dict, List, Union

def load_json(path: str) -> Union[Dict, List]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, path: str, indent: int = 2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_yaml(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
