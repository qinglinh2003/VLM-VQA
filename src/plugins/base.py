from typing import Any, Dict, Tuple

class EvalPlugin:
    """Base class for pluggable evaluation components."""
    def __init__(self, **cfg): 
        self.cfg = cfg

    # ---- sample-level hooks ----
    def before_build_prompt(self, sample: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return possibly modified sample. sample keys: image_path/question/(label|answer|forbidden)"""
        return sample

    def before_encode(self, prompt: str, image, ctx: Dict[str, Any]) -> Tuple[str, Any]:
        """Return (prompt, image)."""
        return prompt, image

    def before_generate(self, model_inputs: Dict[str, Any], gen_kwargs: Dict[str, Any], ctx: Dict[str, Any]):
        """In-place modify model_inputs/gen_kwargs if needed."""
        return

    def after_generate(self, raw_text: str, normalized: str, ctx: Dict[str, Any]) -> Tuple[str, str]:
        """Return possibly modified (raw_text, normalized)."""
        return raw_text, normalized

    # ---- run-level hooks ----
    def init_metrics(self): 
        """Initialize internal accumulators; return optional dict for exposure."""
        return {}

    def update_metrics(self, record: Dict[str, Any]): 
        """Per-sample logging for plugin-specific metrics."""
        return

    def finalize_metrics(self) -> Dict[str, Any]:
        """Return summary dict to be merged into the overall metrics."""
        return {}
