from typing import Any, Dict, Tuple, List
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch, faiss
from transformers import CLIPModel, CLIPProcessor
from .base import EvalPlugin

class RAGClipTopK(EvalPlugin):
    def __init__(
        self,
        index_dir: str = "data/retrieval/clip_l14_336",
        model_id: str = "openai/clip-vit-large-patch14-336",
        topk: int = 3,
        support_tau: float = 0.18,
        refusal: bool = True,
        **cfg
    ):
        super().__init__(**cfg)
        self.index_dir = Path(index_dir)
        self.topk = topk
        self.support_tau = support_tau
        self.refusal = refusal

        self.idx = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.names = json.loads((self.index_dir / "images.json").read_text())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clip = CLIPModel.from_pretrained(
            model_id,
            use_safetensors=True,                            
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
        ).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

        self.use_cnt = 0
        self.refuse_cnt = 0

    def _encode_image(self, img: Image.Image) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.proc(images=img, return_tensors="pt").to(self.device)
            feat = self.clip.get_image_features(**inputs)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().float().numpy()  # (1, D)

    def _encode_text(self, text: str) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.proc(text=text, return_tensors="pt", padding=True).to(self.device)
            feat = self.clip.get_text_features(**inputs)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().float().numpy()  # (1, D)

    @staticmethod
    def _extract_cat(q: str) -> str:
        ql = q.lower()
        head, tail = "are there any ", " in this image"
        if head in ql and tail in ql:
            return ql.split(head,1)[1].split(tail,1)[0].strip()
        return ""

    def before_build_prompt(self, sample: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        img_path = sample.get("image_path") or sample.get("image")
        img = Image.open(img_path).convert("RGB")
        qcat = self._extract_cat(sample["question"])

        qimg = self._encode_image(img)  # (1, D)
        sims, ids = self.idx.search(qimg, self.topk)  # sims: (1,k)
        ids = ids[0].tolist(); sims = sims[0].tolist()
        names = [self.names[i] for i in ids]
        ctx["rag_topk"] = [{"name": n, "sim": float(s)} for n, s in zip(names, sims)]

        s_max = float(max(sims)) if len(sims) > 0 else 0.0

        support = s_max
        if qcat:
            tfeat = self._encode_text(f"a photo of {qcat}")  # (1, D)
            cos_t_q = float((tfeat @ qimg.T)[0, 0])  
            cos_t_q = max(cos_t_q, 0.0)
            support = s_max * cos_t_q

        support = max(0.0, min(1.0, support))
        ctx["support_score"] = support          
        ctx["support_scores"] = [float(s) for s in sims]  
        ctx["support_agg"] = "max_x_textcos" if qcat else "max"
        ctx["rag_support"] = support            

        ev_line = f"Context: retrieved {self.topk} visually similar images. Support={support:.3f}."
        if qcat:
            ev_line += f" Category='{qcat}'."
        sample = dict(sample)
        sample["question"] = sample["question"] + "\n" + ev_line

        self.use_cnt += 1
        ctx["rag_used"] = True
        return sample


    def after_generate(self, raw_text: str, normalized: str, ctx: Dict[str, Any]) -> Tuple[str, str]:
        if self.refusal and normalized == "yes":
            s = ctx.get("rag_support")
            if s is not None and s < self.support_tau:
                self.refuse_cnt += 1
                return raw_text, "unknown"
        return raw_text, normalized

    def finalize_metrics(self) -> Dict[str, Any]:
        return {
            "rag_used_count": int(self.use_cnt),
            "rag_refuse_count": int(self.refuse_cnt),
            "rag_support_tau": float(self.support_tau),
            "rag_topk": int(self.topk),
        }
