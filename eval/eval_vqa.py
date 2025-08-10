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
    ap.add_argument("--data_json", default=None, help="Override cfg.data.vqa_eval_json")
    ap.add_argument("--images_dir", default=None, help="Override cfg.data.images_dir")
    ap.add_argument("--plugins", default=None, help="YAML path for plugins spec (list of {name,params})")
    ap.add_argument("--out_dir", default=None, help="Override cfg.logging.out_dir")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--run_name", default=None, help="Version tag for outputs (e.g., baseline, rag_top3, vcd_v1)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    mcfg, dcfg, lcfg = cfg["model"], cfg["data"], cfg["logging"]
    data_json = args.data_json or dcfg["vqa_eval_json"]
    images_dir = args.images_dir or dcfg["images_dir"]
    out_root = Path(args.out_dir or lcfg["out_dir"])
    run_name = args.run_name or lcfg.get("run_name", "baseline")
    task_dir = out_root / "vqa" / run_name
    task_dir.mkdir(parents=True, exist_ok=True)


    # Load plugins
    plugin_specs = []
    if args.plugins:
        plugin_specs = yaml.safe_load(Path(args.plugins).read_text())
    plugins = load_plugins(plugin_specs)

    # Init model
    runner = LlavaRunner(mcfg["path"], dtype=mcfg.get("dtype","bfloat16"), device=cfg["project"]["device"])

    # Metrics
    total, correct, unknown = 0, 0, 0
    logs = []
    for p in plugins:
        p.init_metrics()

    items = json.loads(Path(data_json).read_text())

    for it in tqdm(items, total=len(items)):
        ctx = {}
        # 1) pre-build
        sample = {"image": it["image"], "image_path": str(Path(images_dir) / it["image"]), 
                  "question": it["question"], "answer": it["answer"]}
        for p in plugins:
            sample = p.before_build_prompt(sample, ctx)

        # 2) load image
        img = Image.open(sample["image_path"]).convert("RGB")

        # 3) allow plugins to modify prompt/image (lightweight here)
        prompt = sample["question"]
        for p in plugins:
            prompt, img = p.before_encode(prompt, img, ctx)

        # 4) run model (plugins can adjust gen kwargs in a more advanced version)
        out = runner.answer(img, prompt, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p)
        raw, pred = out["text"], out["normalized"]

        # 5) allow plugins to post-process
        for p in plugins:
            raw, pred = p.after_generate(raw, pred, ctx)

        gt = sample["answer"].strip().lower()
        is_unknown = (pred == "unknown")
        is_correct = (pred == gt)

        record = {
            "image": it["image"], "question": it["question"], "gt": gt,
            "raw": raw, "pred": pred, "correct": bool(is_correct), "ctx": ctx
        }
        logs.append(record)
        for p in plugins: p.update_metrics(record)

        total += 1
        unknown += int(is_unknown)
        correct += int(is_correct)

    # finalize
    metrics = {
        "accuracy": correct / max(total,1),
        "unknown_rate": unknown / max(total,1),
        "n": total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for p in plugins:
        metrics.update(p.finalize_metrics())

    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (task_dir / "preds.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2))
    (task_dir / "run_meta.json").write_text(json.dumps({
    "config": cfg, "cli": vars(args)
    }, indent=2, default=str))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
