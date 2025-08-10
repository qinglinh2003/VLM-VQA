import re
from typing import Dict, Any, List
import torch
from PIL import Image
from transformers import AutoProcessor

# Compatible import across llava releases
try:
    from transformers import LlavaNextForConditionalGeneration as LlavaCls
except Exception:
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration as LlavaCls


class LlavaRunner:
    """
    Yes/No-only LLaVA runner.
    Always prompts the model to answer only 'Yes' or 'No' and normalizes output
    to {'yes','no','unknown'} for robust evaluation.
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16", device: str = "cuda", trust_remote_code: bool = True):
        torch_dtype = torch.bfloat16 if dtype in ("bfloat16", "bf16") else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.model = LlavaCls.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto", low_cpu_mem_usage=True
        )
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        self.device = device

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str, max_new_tokens: int = 6,
               temperature: float = 0.0, top_p: float = 1.0) -> Dict[str, Any]:
        prompt = self._build_prompt_yesno(question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        raw = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return {"text": raw, "normalized": self.yes_no_from_text(raw)}

    @torch.inference_mode()
    def answer_batch(self, images: List[Image.Image], questions: List[str], max_new_tokens: int = 6,
                     temperature: float = 0.0, top_p: float = 1.0) -> List[Dict[str, Any]]:
        assert len(images) == len(questions)
        prompts = [self._build_prompt_yesno(q) for q in questions]
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        raws = [self.processor.decode(o, skip_special_tokens=True).strip() for o in output_ids]
        return [{"text": r, "normalized": self.yes_no_from_text(r)} for r in raws]

    def _build_prompt_yesno(self, question: str) -> str:
        question = f"{question}\nAnswer only 'Yes' or 'No'."
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
            add_generation_prompt=True,
        )

    @staticmethod
    def yes_no_from_text(raw_text: str) -> str:
        """Normalize free-form text to {'yes','no','unknown'}."""
        t = raw_text.strip()
        # Cut possible chat residues
        for m in ["][/INST]", "[/INST]", "<|assistant|>", "Assistant:", "assistant:"]:
            if m in t:
                t = t.split(m, 1)[-1].strip()

        # Take the first sentence
        first = re.split(r'(?<=[.!?])\s+', t, maxsplit=1)[0].lower()

        # Uncertainty -> unknown
        if re.search(r"\b(maybe|probably|possibly|uncertain|not sure|unsure|don't know|cannot|can't|unknown)\b", first):
            return "unknown"

        # Exact yes/no
        if re.match(r"^\s*yes\b\.?\s*$", first):
            return "yes"
        if re.match(r"^\s*no\b\.?\s*$", first):
            return "no"

        # Strong negatives
        neg_patterns = [
            r"\bthere (is|are) (no|not any)\b",
            r"\bno\b(?!tice)",
            r"\bnot present\b", r"\bnot visible\b",
            r"\bnone\b", r"\babsent\b",
        ]
        if any(re.search(p, first) for p in neg_patterns):
            if "except" not in first:
                return "no"

        # Positives when not negated
        pos_patterns = [
            r"\byes\b", r"\bthere (is|are)\b",
            r"\bvisible\b", r"\bpresent\b", r"\bexist(s)?\b",
        ]
        if any(re.search(p, first) for p in pos_patterns) and not re.search(r"\b(no|not|none|absent)\b", first):
            return "yes"

        # Conflict resolve by earliest index
        def first_idx(text: str, keys: List[str]) -> int:
            idxs = [text.find(k) for k in keys if text.find(k) != -1]
            return min(idxs) if idxs else 10**9

        y_idx = first_idx(first, [" yes", "there is", "there are", " present", " visible", " exist"])
        n_idx = first_idx(first, [" no", "there is not", "there are no", " not ", " none", " absent"])
        if y_idx < n_idx:
            return "yes"
        if n_idx < y_idx:
            return "no"
        return "unknown"
