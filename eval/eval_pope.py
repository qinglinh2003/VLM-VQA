import os, json, argparse, yaml, time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from eval.plugin_loader import load_plugins
from src.backbones.llava_runner import LlavaRunner
from transformers import logging
logging.set_verbosity_error()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_json", default=None, help="Override cfg.data.pope_subset_json")
    ap.add_argument("--images_dir", default=None, help="Override cfg.data.images_dir")
    ap.add_argument("--plugins", default=None, help="YAML path for plugins spec")
    ap.add_argument("--out_dir", default=None, help="Override cfg.logging.out_dir")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    mcfg, dcfg, lcfg = cfg["model"], cfg["data"], cfg["logging"]
    data_json = args.data_json or dcfg["pope_subset_json"]
    images_dir = args.images_dir or dcfg["images_dir"]
    out_dir = Path(args.out_dir or lcfg["out_dir"]) / "pope"
    out_dir.mkdir(parents=True, exist_ok=True)

    plugin_specs = []
    if args.plugins:
        plugin_specs = yaml.safe_load(Path(args.plugins).read_text())
    plugins = load_plugins(plugin_specs)

    runner = LlavaRunner(mcfg["path"], dtype=mcfg.get("dtype","bfloat16"), device=cfg["project"]["device"])

    total, yes_cnt, acc_cnt, neg_total, hallu_cnt, unknown = 0, 0, 0, 0, 0, 0
    logs = []
    for p in plugins:
        p.init_metrics()

    items = json.loads(Path(data_json).read_text())

    for it in tqdm(items, total=len(items)):
        ctx = {}
        # Unified schema adaptation
        label = it.get("label", None)  # 0/1 preferred; else use 'forbidden' legacy
        sample = {"image": it["image"], "image_path": str(Path(images_dir) / it["image"]),
                  "question": it["question"], "label": label, "forbidden": it.get("forbidden")}
        for p in plugins:
            sample = p.before_build_prompt(sample, ctx)

        img = Image.open(sample["image_path"]).convert("RGB")
        prompt = sample["question"]
        for p in plugins:
            prompt, img = p.before_encode(prompt, img, ctx)

        out = runner.answer(img, prompt, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p)
        raw, pred = out["text"], out["normalized"]

        for p in plugins:
            raw, pred = p.after_generate(raw, pred, ctx)

        yes_cnt += int(pred == "yes")
        total += 1
        unknown += int(pred == "unknown")

        if label in (0, 1):
            gt = "yes" if label == 1 else "no"
            acc_cnt += int(pred == gt)
            if label == 0:
                neg_total += 1
                hallu_cnt += int(pred == "yes")
        elif sample.get("forbidden") is not None:
            # legacy negatives
            neg_total += 1
            hallu_cnt += int(pred == "yes")

        record = {
            "image": it["image"], "question": it["question"], "label": label,
            "raw": raw, "pred": pred, "ctx": ctx
        }
        logs.append(record)
        for p in plugins: p.update_metrics(record)

    metrics = {
        "accuracy": (acc_cnt / total) if any("label" in it for it in items) else None,
        "yes_ratio": yes_cnt / max(total, 1),
        "hallu_rate_neg": hallu_cnt / max(neg_total, 1),
        "unknown_rate": unknown / max(total, 1),
        "n_total": total, "n_neg": neg_total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for p in plugins:
        metrics.update(p.finalize_metrics())

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "logs.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
