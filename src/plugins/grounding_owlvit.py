"""
Enhanced OWL-ViT Grounding Plugin with Hallucination Mitigation
"""

from typing import Any, Dict, Tuple, List, Optional
from PIL import Image
import torch
import re
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from .base import EvalPlugin

def extract_category_enhanced(q: str) -> str:
    """Enhanced category extraction supporting multiple question formats"""
    ql = q.lower().strip()
    
    # Pattern 1: "Are there any X in this image?"
    match = re.search(r"are there any ([^?]+?) in this image", ql)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "Is there a/an X in the image?"
    match = re.search(r"is there an? ([^?]+?) in", ql)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: "Do you see any X?"
    match = re.search(r"do you see any ([^?]+)", ql)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: "Can you see X?"
    match = re.search(r"can you see ([^?]+)", ql)
    if match:
        return match.group(1).strip()
    
    # Pattern 5: "Does this image contain X?"
    match = re.search(r"does this image contain ([^?]+)", ql)
    if match:
        return match.group(1).strip()
        
    # Pattern 6: "Are there X visible?"
    match = re.search(r"are there ([^?]+?) visible", ql)
    if match:
        return match.group(1).strip()
    
    return ""

class GroundingOWLViTEnhanced(EvalPlugin):
    def __init__(
        self,
        model_id: str = "google/owlvit-base-patch16",
        box_threshold: float = 0.25,
        mitigation_strategy: str = "conservative",  # conservative, moderate, aggressive
        confidence_threshold: float = 0.4,
        min_detections: int = 1,
        enable_mitigation: bool = True,
        fallback_on_parse_fail: bool = True,
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

        # Mitigation configuration
        self.mitigation_strategy = mitigation_strategy
        self.confidence_threshold = confidence_threshold
        self.min_detections = min_detections
        self.enable_mitigation = enable_mitigation
        self.fallback_on_parse_fail = fallback_on_parse_fail
        self.box_threshold = float(box_threshold)

        # Metrics tracking
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metric counters"""
        self.metrics = {
            # Original metrics
            "pred_yes": 0,
            "pred_yes_with_box": 0, 
            "gt_neg": 0,
            "gt_neg_with_box": 0,
            
            # New mitigation metrics
            "mitigation_applied": 0,
            "mitigation_yes_to_unknown": 0,
            "mitigation_no_to_yes": 0,
            "mitigation_parse_failures": 0,
            "mitigation_skipped_low_conf": 0,
            "detection_support_cases": 0,
            "detection_conflict_cases": 0,
            "avg_detection_confidence": 0.0,
            "total_detections": 0
        }

    def init_metrics(self) -> None:
        self.reset_metrics()

    def before_build_prompt(self, sample: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["image_path"] = sample.get("image_path") or sample.get("image")
        ctx["question_cat"] = extract_category_enhanced(sample["question"])
        
        if not ctx["question_cat"] and self.fallback_on_parse_fail:
            # Log parsing failure but continue
            self.metrics["mitigation_parse_failures"] += 1
            
        return sample

    def _detect(self, img: Image.Image, label: str) -> List[Dict[str, Any]]:
        """Run OWL-ViT detection with enhanced result processing"""
        if not label:
            return []
            
        try:
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
            
            detections = []
            for box, score, label_idx in zip(boxes, scores, labels):
                if label_idx == 0:  # First (and only) query label
                    x1, y1, x2, y2 = map(float, box)
                    detections.append({
                        "box": [x1, y1, x2, y2], 
                        "score": float(score), 
                        "label": label
                    })
                    
            return detections
            
        except Exception as e:
            print(f"Detection failed for label '{label}': {e}")
            return []

    def _calculate_detection_metrics(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from detections"""
        if not detections:
            return {
                "num_detections": 0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "confident_detections": 0
            }
        
        scores = [d["score"] for d in detections]
        return {
            "num_detections": len(detections),
            "avg_confidence": sum(scores) / len(scores),
            "max_confidence": max(scores),
            "confident_detections": sum(1 for s in scores if s >= self.confidence_threshold)
        }

    def _should_override_prediction(
        self, 
        model_pred: str, 
        detections: List[Dict[str, Any]], 
        detection_metrics: Dict[str, float]
    ) -> Tuple[bool, str, str]:
        """
        Determine if prediction should be overridden based on mitigation strategy
        
        Returns: (should_override, new_prediction, reason)
        """
        if not self.enable_mitigation:
            return False, model_pred, "mitigation_disabled"
        
        num_dets = detection_metrics["num_detections"]
        avg_conf = detection_metrics["avg_confidence"] 
        confident_dets = detection_metrics["confident_detections"]
        
        if self.mitigation_strategy == "conservative":
            # Conservative: Only override clear mismatches with high confidence
            if model_pred == "yes" and num_dets == 0:
                return True, "unknown", "no_detections_found"
            elif model_pred == "no" and confident_dets >= 2:
                return True, "yes", "multiple_confident_detections"
                
        elif self.mitigation_strategy == "moderate":
            # Moderate: Override with medium confidence requirements
            if model_pred == "yes" and num_dets == 0:
                return True, "unknown", "no_detections_found"
            elif model_pred == "yes" and avg_conf < 0.3:
                return True, "unknown", "low_confidence_detections"
            elif model_pred == "no" and confident_dets >= 1:
                return True, "yes", "confident_detection_found"
                
        elif self.mitigation_strategy == "aggressive":
            # Aggressive: Override more liberally
            if model_pred == "yes" and num_dets == 0:
                return True, "unknown", "no_detections_found"
            elif model_pred == "yes" and confident_dets == 0:
                return True, "unknown", "no_confident_detections" 
            elif model_pred == "no" and num_dets >= 1:
                return True, "yes", "detection_found"
            elif model_pred == "unknown" and confident_dets >= 2:
                return True, "yes", "strong_detection_evidence"
                
        return False, model_pred, "no_override_needed"

    def after_generate(self, raw: str, pred: str, ctx: Dict[str, Any]) -> Tuple[str, str]:
        """Enhanced post-processing with hallucination mitigation"""
        label = ctx.get("question_cat", "")
        img_path = ctx.get("image_path")
        
        # Initialize grounding context
        grounding_ctx = {
            "label": label, 
            "boxes": [], 
            "threshold": self.box_threshold,
            "mitigation_applied": False,
            "mitigation_reason": None,
            "detection_metrics": {}
        }
        
        # Skip detection if no category extracted
        if not img_path or not label:
            ctx["grounding"] = grounding_ctx
            if not label:
                self.metrics["mitigation_parse_failures"] += 1
            return raw, pred
            
        # Run detection
        img = Image.open(img_path).convert("RGB")
        detections = self._detect(img, label)
        detection_metrics = self._calculate_detection_metrics(detections)
        
        # Update context with detection results
        grounding_ctx.update({
            "boxes": detections,
            "detection_metrics": detection_metrics
        })
        
        # Update metrics tracking
        self.metrics["total_detections"] += len(detections)
        if len(detections) > 0:
            self.metrics["avg_detection_confidence"] += detection_metrics["avg_confidence"]
            self.metrics["detection_support_cases"] += 1
        
        if pred == "yes":
            self.metrics["pred_yes"] += 1
            if len(detections) > 0:
                self.metrics["pred_yes_with_box"] += 1

        # Apply mitigation logic
        should_override, new_pred, reason = self._should_override_prediction(
            pred, detections, detection_metrics
        )
        
        if should_override:
            self.metrics["mitigation_applied"] += 1
            grounding_ctx["mitigation_applied"] = True
            grounding_ctx["mitigation_reason"] = reason
            
            if pred == "yes" and new_pred == "unknown":
                self.metrics["mitigation_yes_to_unknown"] += 1
            elif pred == "no" and new_pred == "yes":
                self.metrics["mitigation_no_to_yes"] += 1
            elif pred != new_pred:
                self.metrics["detection_conflict_cases"] += 1
                
            pred = new_pred
            
        ctx["grounding"] = grounding_ctx
        return raw, pred

    def update_metrics(self, record: Dict[str, Any]) -> None:
        """Update metrics with ground truth comparison"""
        label = record.get("label")
        grounding = record.get("ctx", {}).get("grounding", {})
        boxes = grounding.get("boxes", [])
        
        if label == 0:  # Ground truth negative
            self.metrics["gt_neg"] += 1
            if len(boxes) > 0:
                self.metrics["gt_neg_with_box"] += 1

    def finalize_metrics(self) -> Dict[str, Any]:
        """Calculate final metrics including mitigation effectiveness"""
        # Avoid division by zero
        pred_yes = max(self.metrics["pred_yes"], 1)
        gt_neg = max(self.metrics["gt_neg"], 1) 
        support_cases = max(self.metrics["detection_support_cases"], 1)
        total_mit = max(self.metrics["mitigation_applied"], 1)
        
        final_metrics = {
            # Original metrics
            "ground_yes_precision": self.metrics["pred_yes_with_box"] / pred_yes,
            "ground_fp_neg": self.metrics["gt_neg_with_box"] / gt_neg,
            "ground_box_threshold": self.box_threshold,
            
            # New mitigation metrics  
            "mitigation_strategy": self.mitigation_strategy,
            "mitigation_enabled": self.enable_mitigation,
            "mitigation_rate": self.metrics["mitigation_applied"] / max(self.metrics["pred_yes"] + self.metrics.get("pred_no", 0), 1),
            "mitigation_yes_to_unknown_rate": self.metrics["mitigation_yes_to_unknown"] / total_mit,
            "mitigation_no_to_yes_rate": self.metrics["mitigation_no_to_yes"] / total_mit,
            "category_extraction_failure_rate": self.metrics["mitigation_parse_failures"] / max(pred_yes, 1),
            "avg_detection_confidence": self.metrics["avg_detection_confidence"] / support_cases if support_cases > 0 else 0,
            "detection_support_rate": support_cases / max(pred_yes, 1),
            
            # Configuration
            "confidence_threshold": self.confidence_threshold,
            "min_detections": self.min_detections,
        }
        
        return final_metrics