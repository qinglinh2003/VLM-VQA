from PIL import Image, ImageFilter
from typing import Dict, Any, Tuple

class VCDLightPlugin:
    def __init__(self, neg_view: str = "blur", blur_radius: int = 8,
                 rule: str = "penalize_yes_if_neg_not_no_and_low_support",
                 soft_tau: float = 0.075):
        self.neg_view = neg_view
        self.blur_radius = blur_radius
        self.rule = rule
        self.soft_tau = soft_tau
        self.stats = {"vcd_applied": 0, "vcd_downgrade": 0, "vcd_agree": 0, "vcd_conflict": 0,
                      "neg_yes": 0, "neg_no": 0, "neg_unk": 0}

    def init_metrics(self) -> None:
        self.stats = {k: 0 for k in self.stats}

    def before_build_prompt(self, sample: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        return sample

    def before_encode(self, prompt: str, img: Image.Image, ctx: Dict[str, Any]) -> Tuple[str, Image.Image]:
        return prompt, img

    def wants_contrast(self, ctx: Dict[str, Any]) -> bool:
        return ctx.get("pos", {}).get("pred") == "yes"

    def make_contrast_view(self, img: Image.Image, ctx: Dict[str, Any]) -> Image.Image:
        if self.neg_view == "blur":
            return img.filter(ImageFilter.GaussianBlur(self.blur_radius))
        elif self.neg_view == "grayscale":
            return img.convert("L").convert("RGB")
        elif self.neg_view == "center_cutout":
            w, h = img.size
            cw, ch = int(w * 0.4), int(h * 0.4)
            x0 = (w - cw) // 2
            y0 = (h - ch) // 2
            neg = img.copy()
            for x in range(x0, x0 + cw):
                for y in range(y0, y0 + ch):
                    neg.putpixel((x, y), (0, 0, 0))
            return neg
        else:
            return img.filter(ImageFilter.GaussianBlur(self.blur_radius))

    def after_generate(self, raw: str, pred: str, ctx: Dict[str, Any]) -> Tuple[str, str]:
        contrast = ctx.get("contrast")
        if not contrast:
            return raw, pred

        self.stats["vcd_applied"] += 1
        neg_pred = contrast.get("pred", "unknown")
        if   neg_pred == "yes":     self.stats["neg_yes"] += 1
        elif neg_pred == "no":      self.stats["neg_no"]  += 1
        else:                       self.stats["neg_unk"] += 1

        if self.rule == "penalize_yes_if_neg_not_no_and_low_support":
            support = ctx.get("support_score", 1.0)
            if pred == "yes" and neg_pred != "no" and support < self.soft_tau:
                self.stats["vcd_downgrade"] += 1
                return raw, "unknown"

        if pred == neg_pred:
            self.stats["vcd_agree"] += 1
        else:
            self.stats["vcd_conflict"] += 1
        return raw, pred

    def update_metrics(self, record: Dict[str, Any]) -> None:
        pass

    def finalize_metrics(self) -> Dict[str, Any]:
        return dict(self.stats)
