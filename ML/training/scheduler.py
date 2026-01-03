import sys, os, json, time
from copy import deepcopy
from typing import Dict, Any, List
import yaml
import numpy as np

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
    "epochs",
    "learning_rate",
    "mix_ratio",
    "category_class_weights",
    "encoder_lr",   # new: per-job encoder learning rate (for dual optimizer)
    "head_lr",      # new: per-job head learning rate (for dual optimizer). If present and dual disabled, used as single LR
}

METRIC_FORMATS = {
    # plain floats
    "loss": {"decimals": 4},
    "val_loss": {"decimals": 4},
    "best_loss": {"decimals": 4},
    "final_loss": {"decimals": 4},
    "total_loss": {"decimals": 4},

    # accuracies as percentages
    "accuracy": {"decimals": 2, "percent": True},
    "categorical_accuracy": {"decimals": 2, "percent": True},
    "category_accuracy": {"decimals": 2, "percent": True},
    "acc_combo": {"decimals": 2, "percent": True},

    # correlations
    "corr": {"decimals": 4},

    # errors
    "mae": {"decimals": 4},
    "mse": {"decimals": 4},
}


def load_stage(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)["stage"]

def build_job_config(base_config, base_mode_config, stage, job):
    cfg = base_config.copy()
    mode_cfg = deepcopy(base_mode_config)

    # --- Stage-level IO ---
    cfg["load_encoder_model"] = stage["io"].get("load_encoder", False)
    cfg["load_head_model"] = stage["io"].get("load_head", False)
    cfg["save_encoder_model"] = stage["io"].get("save_encoder", True)
    cfg["save_head_model"] = stage["io"].get("save_head", True)

    # --- Learning rates (dual optimizer assumed) ---
    cfg["encoder_lr"] = job.get(
        "encoder_lr",
        stage["defaults"]["encoder_lr"]
    )
    cfg["head_lr"] = job.get(
        "head_lr",
        stage["defaults"]["head_lr"]
    )

    if job.get("reset_head", False):
        cfg["load_head_model"] = False

    # --- Mode ---
    cfg["mode"] = job["mode"]
    cfg["submode"] = job["submode"]

    # --- Epochs ---
    mode_cfg["default_epochs"] = job["epochs"]

    return cfg, mode_cfg

def _safe_float_values(seq):
    """Convert a sequence of items (maybe tf Tensors / numpy arrays / scalars) to a list of floats."""
    vals = []
    for x in seq:
        try:
            # tensorflow tensor or variable
            if hasattr(x, "numpy"):
                v = x.numpy()
            else:
                v = x
            # numpy array / list
            if isinstance(v, (np.ndarray, list, tuple)):
                flat = np.array(v).astype(float).ravel()
                vals.extend([float(t) for t in flat])
            else:
                vals.append(float(v))
        except Exception:
            # skip values that can't be converted
            continue
    return vals

def _summarize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    if not history:
        return {}
    s = {}
    # losses (safe conversion)
    if "val_loss" in history:
        vals = _safe_float_values(history["val_loss"])
        if vals:
            s["best_val_loss"] = float(min(vals))
            s["final_val_loss"] = float(vals[-1])
    if "loss" in history:
        vals = _safe_float_values(history["loss"])
        if vals:
            s["best_loss"] = float(min(vals))
            s["final_loss"] = float(vals[-1])
    # other val metrics (take best = max)
    for k, v in history.items():
        if k.startswith("val_") and k != "val_loss" and v is not None:
            vals = _safe_float_values(v)
            if vals:
                s[f"best_{k}"] = float(max(vals))
                s[f"final_{k}"] = float(vals[-1])
    return s

def _summarize_evaluation(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    if not eval_result:
        return {}
    keys_of_interest = ("accuracy", "mse", "mae", "corr", "bce_loss", "bce", "bce_combo", "acc_combo", "total_loss", "category_accuracy")
    s = {}
    for k in keys_of_interest:
        if k in eval_result:
            try:
                val = eval_result[k]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                if isinstance(val, (np.ndarray, list, tuple)):
                    val = np.array(val).ravel()[0]
                s[k] = float(val)
            except Exception:
                # skip non-convertible values
                continue
    return s

def run_schedule(schedule_path: str = None, output_path: str = None, base_config: Dict[str, Any] = None):
    base_config = base_config or get_config()
    schedule_path = schedule_path or base_config.get("schedule_file")
    output_path = output_path or base_config.get("schedule_output")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stage = load_stage(schedule_path)
    results = []

    for cycle_idx in range(stage["cycles"]):
        print(f"\n🔁 Cycle {cycle_idx+1}/{stage['cycles']}")

        for job_idx, job in enumerate(stage["jobs"]):
            print(f"▶ Job {job_idx+1}: {job['name']}")

            print(f"\n=== Running job {job_idx+1}/{len(stage["jobs"])}: {job.get('mode')} ===")
            start_ts = time.time()
            try:
                base_mode_cfg = get_mode_config(job["mode"])
                cfg, mode_cfg = build_job_config(base_config, base_mode_cfg, stage, job)

                model, info = train_embeddings(cfg, mode_cfg, return_info=True)

                print("Training history keys and sample values:")
                print({k: v[:3] for k, v in info.get("history", {}).items()})

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

                        num_examples = job.get("num_examples", 50)
                        eval_kwargs = mode_cfg.get("evaluation_function_kwargs", {})
                        eval_result = eval_fn(model, iter(val_gen), num_examples=num_examples, **eval_kwargs)
                    except Exception as e:
                        eval_result = {"error": str(e)}

                print("Evaluation result:", eval_result)

                result = {
                    "job": job,
                    "cycle": cycle_idx + 1,
                    "start_time": info["start_time"],
                    "end_time": info["end_time"],
                    "duration_seconds": info["end_time"] - info["start_time"],
                    "model_path": info.get("model_path"),
                    "encoder_path": info.get("encoder_path"),
                    "encoder_lr": info.get("encoder_lr"),
                    "head_lr": info.get("head_lr"),
                    # "evaluation": eval_result,
                    "summary": {
                        **_summarize_history(info.get("history")),
                    },                    
                    "history_keys": list(info["history"].keys()) if info["history"] else None,
                    "success": True,
                    "message": None
                }

                summary = {}
                summary.update(_summarize_history(info.get("history")))
                summary.update(_summarize_evaluation(eval_result))
                result["summary"] = summary

            except Exception as exc:
                summary = {}
                result = {
                    "job": job,
                    "cycle": cycle_idx + 1,
                    "start_time": start_ts,
                    "end_time": time.time(),
                    "duration_seconds": time.time() - start_ts,
                    "model_path": None,
                    "encoder_path": None,
                    "history_keys": None,
                    "success": False,
                    "message": str(exc),
                    "summary": {}
                }
                print("Job failed:", exc)

            def _format_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
                out = {}
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        out[k] = _format_metric(k, v)
                    else:
                        out[k] = v
                return out

            result["summary"] = _format_summary(summary)

            results.append(result)

        # Persist intermediate results so long runs don't lose progress
        with open(output_path, "w") as fout:
            json.dump({"results": results}, fout, indent=2, default=str)

    stage_summary = {
        "stage_name": stage.get("name", "unnamed_stage"),
        "total_cycles": stage["cycles"],
        "jobs": [job["name"] for job in stage["jobs"]],
        "results_count": len(results),
        "summary_per_job": {}
    }

    from collections import defaultdict
    job_aggregates = defaultdict(list)

    for r in results:
        job_aggregates[r["job"]["name"]].append(r["summary"])

    def aggregate_summaries(summaries):
        agg = {}
        if not summaries:
            return agg
        keys = summaries[0].keys()
        for k in keys:
            vals = [s[k] for s in summaries if k in s]
            if all(isinstance(v, (int, float)) for v in vals):
                agg[k] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": sum(vals) / len(vals),
                }
        return agg

    for job_name, sums in job_aggregates.items():
        stage_summary["summary_per_job"][job_name] = aggregate_summaries(sums)

    summary_path = os.path.splitext(output_path)[0] + "_stage_summary.json"
    with open(summary_path, "w") as fout:
        json.dump(stage_summary, fout, indent=2, default=str)

    print(f"\nStage complete. Stage summary saved to {summary_path}")

    return stage_summary

def _format_metric(key: str, value: float):
    spec = METRIC_FORMATS.get(key)
    if spec is None:
        return round(value, 4)  # sane default

    decimals = spec.get("decimals", 4)
    if spec.get("percent", False):
        return round(value * 100.0, decimals)

    return round(value, decimals)

if __name__ == "__main__":
    cfg = get_config()
    run_schedule(cfg.get("schedule_file"), cfg.get("schedule_output"), cfg)
