import importlib
from typing import List, Dict, Any

def load_plugins(specs: List[Dict[str, Any]]):
    """
    specs: [{"name": "src.plugins.rag_text_prompt:RAGTextPrompt", "params": {...}}, ...]
    """
    instances = []
    for spec in specs or []:
        target = spec["name"]
        params = spec.get("params", {})
        if ":" in target:
            module_name, cls_name = target.split(":", 1)
        else:
            parts = target.split(".")
            module_name, cls_name = ".".join(parts[:-1]), parts[-1]
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        instances.append(cls(**params))
    return instances
