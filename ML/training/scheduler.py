import sys, os, json, time
from copy import deepcopy
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from config import get_config
from mode_config import get_mode_config
from training.trainer import train_embeddings

VALID_OVERRIDE_KEYS = {
    "mode",
    "submode",
    "load_encoder_model",
    "load_head_model",
    "save_encoder_model",
    "save_head_model",
    "use_dual_optimizer",
    "epochs",
    "learning_rate",
    "mix_ratio",
    "category_class_weights",
    "encoder_lr",   # new: per-job encoder learning rate (for dual optimizer)
    "head_lr",      # new: per-job head learning rate (for dual optimizer). If present and dual disabled, used as single LR
}

def _load_schedule(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    # Accept either top-level list or {"jobs": [...]}
    if isinstance(data, dict) and "jobs" in data:
        return data["jobs"]
    if isinstance(data, list):
        return data
    raise ValueError("Schedule file must be a JSON list or an object with a 'jobs' list")

def _apply_overrides(base_config: Dict[str, Any], base_mode_config: Dict[str, Any], entry: Dict[str, Any]):
    # Use a shallow copy for global config to avoid deep-copying unpickleable objects
    # (e.g. DB connections). We only mutate a few top-level keys, so shallow copy
    # is sufficient and avoids the psycopg2 pickling error.
    cfg = base_config.copy()
    mode_cfg = deepcopy(base_mode_config)

    # Global overrides
    for k in ("mode", "submode", "load_encoder_model", "load_head_model", "save_encoder_model", "save_head_model", "use_dual_optimizer", "mix_ratio", "category_class_weights"):
        if k in entry:
            cfg[k] = entry[k]

    # Learning rate overrides:
    # - head_lr supersedes 'learning_rate' for the head.
    # - encoder_lr is only used when use_dual_optimizer is true (trainer will decide).
    if "encoder_lr" in entry:
        cfg["encoder_lr"] = float(entry["encoder_lr"])
    if "head_lr" in entry:
        cfg["head_lr"] = float(entry["head_lr"])
        # Mirror to mode_cfg learning_rate for downstream compatibility
        mode_cfg["learning_rate"] = float(entry["head_lr"])
    # Backwards-compatible: numeric "learning_rate" in schedule applies to head
    if "learning_rate" in entry and "head_lr" not in entry:
        mode_cfg["learning_rate"] = float(entry["learning_rate"])
        cfg["head_lr"] = float(entry["learning_rate"])

    # Mode-specific overrides
    if "epochs" in entry:
        mode_cfg["default_epochs"] = int(entry["epochs"])

    return cfg, mode_cfg

def _summarize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    if not history:
        return {}
    s = {}
    # losses
    if "val_loss" in history and history["val_loss"]:
        s["best_val_loss"] = float(min(history["val_loss"]))
        s["final_val_loss"] = float(history["val_loss"][-1])
    if "loss" in history and history["loss"]:
        s["best_loss"] = float(min(history["loss"]))
        s["final_loss"] = float(history["loss"][-1])
    # other val metrics (take best = max)
    for k, v in history.items():
        if k.startswith("val_") and k != "val_loss" and v:
            s[f"best_{k}"] = float(max(v))
            s[f"final_{k}"] = float(v[-1])
    return s

def _summarize_evaluation(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    if not eval_result:
        return {}
    # pick a small set of commonly useful eval metrics if present
    keys_of_interest = ("accuracy", "mse", "mae", "corr", "bce", "bce_combo", "acc_combo", "total_loss", "category_accuracy")
    s = {k: float(eval_result[k]) for k in keys_of_interest if k in eval_result}
    return s

def run_schedule(schedule_path: str = None, output_path: str = None, base_config: Dict[str, Any] = None):
    base_config = base_config or get_config()
    schedule_path = schedule_path or base_config.get("schedule_file")
    output_path = output_path or base_config.get("schedule_output")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    jobs = _load_schedule(schedule_path)
    results = []

    for idx, entry in enumerate(jobs):
        print(f"\n=== Running job {idx+1}/{len(jobs)}: {entry.get('mode')} ===")
        start_ts = time.time()
        try:
            mode = entry.get("mode")
            if not mode:
                raise ValueError("Job missing 'mode'")

            base_mode_cfg = get_mode_config(mode)
            cfg, mode_cfg = _apply_overrides(base_config, base_mode_cfg, entry)

            # Ensure flags exist
            cfg.setdefault("save_model", True)
            cfg.setdefault("save_head_model", True)
            cfg.setdefault("save_encoder_model", True)

            # Run training, request info back (capture model for evaluation)
            model, info = train_embeddings(cfg, mode_cfg, return_info=True)

            # Post-training evaluation (if available)
            eval_result = None
            eval_fn = mode_cfg.get("evaluation_function")
            if eval_fn is not None:
                try:
                    # Build a validation generator similar to training script
                    val_config = dict(cfg)
                    val_config["is_validation"] = True
                    gen_cls = mode_cfg["generator"]
                    val_gen = gen_cls(val_config)
                    val_gen.preload_validation_data()

                    num_examples = entry.get("num_examples", 50)
                    eval_kwargs = mode_cfg.get("evaluation_function_kwargs", {})
                    eval_result = eval_fn(model, iter(val_gen), num_examples=num_examples, **eval_kwargs)
                except Exception as e:
                    eval_result = {"error": str(e)}

            result = {
                "job": entry,
                "start_time": info["start_time"],
                "end_time": info["end_time"],
                "duration_seconds": info["end_time"] - info["start_time"],
                "model_path": info.get("model_path"),
                "encoder_path": info.get("encoder_path"),
                "encoder_lr": info.get("encoder_lr"),
                "head_lr": info.get("head_lr"),
                "evaluation": eval_result,
                "history_keys": list(info["history"].keys()) if info["history"] else None,
                "success": True,
                "message": None
            }

            # Build a concise summary combining training history and evaluation metrics
            summary = {}
            summary.update(_summarize_history(info.get("history")))
            summary.update(_summarize_evaluation(eval_result))
            result["summary"] = summary

        except Exception as exc:
            result = {
                "job": entry,
                "start_time": start_ts,
                "end_time": time.time(),
                "duration_seconds": time.time() - start_ts,
                "model_path": None,
                "encoder_path": None,
                "history_keys": None,
                "success": False,
                "message": str(exc)
            }
            print("Job failed:", exc)

        results.append(result)

        # Persist intermediate results so long runs don't lose progress
        with open(output_path, "w") as fout:
            json.dump({"results": results}, fout, indent=2, default=str)

    print(f"\nSchedule finished, results written to {output_path}")
    return results

if __name__ == "__main__":
    cfg = get_config()
    run_schedule(cfg.get("schedule_file"), cfg.get("schedule_output"), cfg)
