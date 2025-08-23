from typing import Any, Dict, Tuple, List
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from .base import EvalPlugin

def extract_category(q: str) -> str:
    ql = q.lower()
    head, tail = "are there any ", " in this image"
    if head in ql and tail in ql:
        return ql.split(head, 1)[1].split(tail, 1)[0].strip()
    return ""

class GroundingOWLVit(EvalPlugin):
    def __init__(
        self,
        model_id: str = "google/owlvit-base-patch16",
        box_threshold: float = 0.25,
        device: str = None,
        **cfg
    ):
        super().__init__(**cfg)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(
            model_id, 
            use_safetensors=True 
        ).to(self.device).eval()

        self.pred_yes = 0
        self.pred_yes_with_box = 0
        self.gt_neg = 0
        self.gt_neg_with_box = 0
        self.box_threshold = float(box_threshold)

    def init_metrics(self) -> None:
        self.pred_yes = 0
        self.pred_yes_with_box = 0
        self.gt_neg = 0
        self.gt_neg_with_box = 0

    def before_build_prompt(self, sample: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["image_path"] = sample.get("image_path") or sample.get("image")
        ctx["question_cat"] = extract_category(sample["question"])
        return sample

    def before_encode(self, prompt: str, img: Image.Image, ctx: Dict[str, Any]) -> Tuple[str, Image.Image]:
        return prompt, img

    def _detect(self, img: Image.Image, label: str) -> List[Dict[str, Any]]:
        if not label:
            return []
        inputs = self.processor(text=[[label]], images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]], device=self.device)  
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.box_threshold
        )[0]
        boxes = results["boxes"].detach().cpu().tolist()
        scores = results["scores"].detach().cpu().tolist()
        labels = results["labels"].detach().cpu().tolist()  
        dets = []
        for b, s, li in zip(boxes, scores, labels):
            if li == 0:  
                x1, y1, x2, y2 = map(float, b)
                dets.append({"box": [x1, y1, x2, y2], "score": float(s), "label": label})
        return dets

    def after_generate(self, raw: str, pred: str, ctx: Dict[str, Any]) -> Tuple[str, str]:
        label = ctx.get("question_cat", "")
        img_path = ctx.get("image_path")
        if not img_path or not label:
            ctx["grounding"] = {"label": label, "boxes": [], "threshold": self.box_threshold}
            return raw, pred

        img = Image.open(img_path).convert("RGB")
        dets = self._detect(img, label)
        ctx["grounding"] = {"label": label, "boxes": dets, "threshold": self.box_threshold}

        gt = ctx.get("label")  

        if pred == "yes":
            self.pred_yes += 1
            if len(dets) > 0:
                self.pred_yes_with_box += 1

        return raw, pred

    def update_metrics(self, record: Dict[str, Any]) -> None:
        label = record.get("label")
        boxes = (record.get("ctx", {}).get("grounding", {}) or {}).get("boxes", [])
        if label == 0:  
            self.gt_neg += 1
            if len(boxes) > 0:
                self.gt_neg_with_box += 1

    def finalize_metrics(self) -> Dict[str, Any]:
        gy = self.pred_yes if self.pred_yes > 0 else 1
        gn = self.gt_neg if self.gt_neg > 0 else 1
        return {
            "ground_yes_precision": self.pred_yes_with_box / gy,  
            "ground_fp_neg": self.gt_neg_with_box / gn,           
            "ground_box_threshold": self.box_threshold,
        }
