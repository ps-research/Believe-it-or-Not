#!/usr/bin/env python3
"""
BELIEVE IT OR NOT — COMPLETE EXPERIMENT SUITE
===============================================
"Mechanistic Interpretability of Learned Chain-of-Thought Unfaithfulness"

Uses pretrained Gemma Scope 2 CLTs as anomaly detectors on GRPO-finetuned model.
The reconstruction failure differential localizes WHERE GRPO added new computation.

4 Experiments:
  1. Reconstruction Differential Heatmap — per-layer, per-position
  2. Training Dynamics — differential across 6 GRPO checkpoints
  3. Feature Analysis — which CLT features are new in GRPO'd model?
  4. Adversarial Validation — same circuit for wrong hints?

Usage:
  python run_suite.py --quick          # 5 prompts, final checkpoint only
  python run_suite.py --only diff      # just reconstruction differential
  python run_suite.py                  # full suite

GPU Memory: ~19 GB (base model + CLT + merged model headroom)
"""

import sys, os, json, time, argparse, traceback, logging, re, gc
from datetime import datetime
from collections import defaultdict

WORKSPACE = "/workspace"
PROJECT = os.path.join(WORKSPACE, "Believe-it-or-Not")
INFRA = os.path.join(PROJECT, "Gemma-Scope-2-Study")

sys.path.insert(0, INFRA)
sys.path.insert(0, PROJECT)

import torch
import numpy as np
torch.set_grad_enabled(False)

from src.loader import load_clt, GEMMA3_1B_NUM_LAYERS
from src.hooks import gather_clt_activations
from src.metrics import compute_fvu

CACHE = os.path.join(INFRA, "cache")
DATA = os.path.join(PROJECT, "dataset")
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUT, "experiment.log"), mode="w"),
    ],
)
log = logging.getLogger("believe_it_or_not")


def save_results(data, filename):
    path = os.path.join(OUT, filename)
    data["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")


# ============================================================
# Model Loading
# ============================================================

def load_base_model(device="cuda"):
    """Load base Gemma 3 1B IT with standard transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading base Gemma 3 1B IT...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model.eval()

    mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    log.info(f"  Base model loaded: {mem:.2f} GB on {device}")
    return model, tokenizer


def load_grpo_model(lora_path, device="cuda"):
    """Load base model + LoRA, merge, return standard HF model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info(f"Loading GRPO model from {lora_path}...")

    base = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    peft_model = PeftModel.from_pretrained(base, lora_path)
    merged = peft_model.merge_and_unload()
    merged.eval()

    mem = sum(p.numel() * p.element_size() for p in merged.parameters()) / (1024**3)
    log.info(f"  Merged GRPO model: {mem:.2f} GB")
    return merged, tokenizer


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(name):
    path = os.path.join(DATA, name)
    with open(path) as f:
        data = json.load(f)
    log.info(f"  Loaded {name}: {len(data)} examples")
    return data


def format_prompt(example, tokenizer):
    """Format dataset example into model input string."""
    messages = example["prompt"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


# ============================================================
# Core Analysis: Per-Position Reconstruction Error
# ============================================================

def compute_per_position_mse(model, clt, tokenizer, text, device="cuda"):
    """
    Compute per-(position, layer) reconstruction MSE.

    Returns:
        mse: np.array of shape (seq_len, 26) — MSE at each position/layer
        tokens: list of token strings
    """
    inputs = tokenizer.encode(text, return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True, max_length=512).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

    clt_in, clt_tgt = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)

    if next(clt.parameters()).dtype == torch.float16:
        clt_in = clt_in.half()
        clt_tgt = clt_tgt.half()

    recon = clt.forward(clt_in)

    # Per-position, per-layer MSE: shape (seq_len, 26)
    residual = (recon - clt_tgt).float()
    mse = (residual ** 2).mean(dim=-1).cpu().numpy()  # mean over d_model

    return mse, tokens


def compute_per_layer_fvu(model, clt, tokenizer, text, device="cuda"):
    """
    Compute per-layer FVU (aggregated over positions).

    Returns:
        fvu_per_layer: list of 26 floats
    """
    inputs = tokenizer.encode(text, return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True, max_length=512).to(device)

    clt_in, clt_tgt = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)

    if next(clt.parameters()).dtype == torch.float16:
        clt_in = clt_in.half()
        clt_tgt = clt_tgt.half()

    recon = clt.forward(clt_in)

    fvu_per_layer = []
    for layer in range(GEMMA3_1B_NUM_LAYERS):
        r = recon[:, layer, :].float()
        t = clt_tgt[:, layer, :].float()
        fvu = compute_fvu(r, t).item()
        fvu_per_layer.append(fvu)

    return fvu_per_layer


# ============================================================
# EXPERIMENT 1: Reconstruction Differential Heatmap
# ============================================================

def run_exp1_differential(model_base, model_grpo, clt, tokenizer,
                          n_prompts=20):
    """
    Compare per-position MSE between base and GRPO'd model.
    Run on hint-present, hint-absent, and non-MCQ prompts.
    """
    log.info("=" * 70)
    log.info("EXPERIMENT 1: RECONSTRUCTION DIFFERENTIAL HEATMAP")
    log.info("=" * 70)

    results = {"conditions": {}}

    # Load datasets
    val_hint = load_dataset("grpo_val_faithful.json")[:n_prompts]
    val_nohint = load_dataset("grpo_val_no_hint.json")[:n_prompts]

    # Non-MCQ sanity check prompts
    sanity_prompts = [
        "The history of the Roman Empire spans over a thousand years, from the traditional founding of Rome in 753 BC through the fall of Constantinople in 1453 AD.",
        "Photosynthesis is the process by which green plants convert carbon dioxide and water into glucose and oxygen using sunlight energy.",
        "The theory of general relativity, published by Albert Einstein in 1915, describes gravity as a curvature of spacetime caused by mass and energy.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "The water cycle involves evaporation from surface water, condensation into clouds, and precipitation back to the surface.",
    ]

    conditions = [
        ("with_hint", val_hint, True),
        ("no_hint", val_nohint, True),
        ("non_mcq", [{"text": t} for t in sanity_prompts], False),
    ]

    for cond_name, examples, is_dataset in conditions:
        log.info(f"\n  Condition: {cond_name} ({len(examples)} prompts)")

        all_diffs = []       # list of (seq_len, 26) arrays
        all_base_mse = []
        all_grpo_mse = []
        per_layer_fvu_base = []
        per_layer_fvu_grpo = []
        prompt_info = []

        for pi, example in enumerate(examples):
            try:
                if is_dataset:
                    text = format_prompt(example, tokenizer)
                else:
                    text = example["text"]

                # Base model MSE
                mse_base, tokens = compute_per_position_mse(
                    model_base, clt, tokenizer, text)

                # GRPO model MSE
                mse_grpo, _ = compute_per_position_mse(
                    model_grpo, clt, tokenizer, text)

                # Differential
                # Pad/truncate to same length (should be same but be safe)
                min_len = min(mse_base.shape[0], mse_grpo.shape[0])
                diff = mse_grpo[:min_len] - mse_base[:min_len]

                all_diffs.append(diff)
                all_base_mse.append(mse_base[:min_len])
                all_grpo_mse.append(mse_grpo[:min_len])

                # Per-layer FVU
                fvu_b = compute_per_layer_fvu(model_base, clt, tokenizer, text)
                fvu_g = compute_per_layer_fvu(model_grpo, clt, tokenizer, text)
                per_layer_fvu_base.append(fvu_b)
                per_layer_fvu_grpo.append(fvu_g)

                # Token info for the first few prompts
                if pi < 5:
                    prompt_info.append({
                        "prompt_idx": pi,
                        "tokens": tokens[:min_len],
                        "n_tokens": min_len,
                        "diff_mean_per_layer": diff.mean(axis=0).tolist(),
                        "diff_max_per_layer": diff.max(axis=0).tolist(),
                        "has_hint": example.get("has_hint", False) if is_dataset else False,
                    })

                if pi % 5 == 0:
                    log.info(f"    Prompt {pi+1}/{len(examples)}: "
                             f"mean_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")

            except Exception as e:
                log.error(f"    Prompt {pi} failed: {e}")
                log.error(traceback.format_exc())

        # Aggregate per-layer statistics
        if per_layer_fvu_base:
            avg_fvu_base = np.mean(per_layer_fvu_base, axis=0).tolist()
            avg_fvu_grpo = np.mean(per_layer_fvu_grpo, axis=0).tolist()
            fvu_diff = (np.mean(per_layer_fvu_grpo, axis=0) -
                       np.mean(per_layer_fvu_base, axis=0)).tolist()
        else:
            avg_fvu_base = avg_fvu_grpo = fvu_diff = []

        # Average differential heatmap (pad all to max length, fill with 0)
        if all_diffs:
            max_len = max(d.shape[0] for d in all_diffs)
            padded = np.zeros((len(all_diffs), max_len, GEMMA3_1B_NUM_LAYERS))
            for i, d in enumerate(all_diffs):
                padded[i, :d.shape[0], :] = d
            avg_diff_heatmap = padded.mean(axis=0).tolist()
        else:
            avg_diff_heatmap = []

        results["conditions"][cond_name] = {
            "n_prompts": len(examples),
            "avg_fvu_base_per_layer": avg_fvu_base,
            "avg_fvu_grpo_per_layer": avg_fvu_grpo,
            "fvu_differential_per_layer": fvu_diff,
            "avg_diff_heatmap": avg_diff_heatmap,
            "prompt_details": prompt_info,
        }

        # Log per-layer summary
        if fvu_diff:
            top_layers = sorted(range(26), key=lambda l: abs(fvu_diff[l]), reverse=True)[:5]
            log.info(f"    Top differential layers:")
            for l in top_layers:
                log.info(f"      L{l}: base={avg_fvu_base[l]:.4f}, "
                         f"grpo={avg_fvu_grpo[l]:.4f}, "
                         f"diff={fvu_diff[l]:+.4f}")

    save_results(results, "exp1_reconstruction_differential.json")
    return results


# ============================================================
# EXPERIMENT 2: Training Dynamics
# ============================================================

def run_exp2_dynamics(model_base, clt, tokenizer, n_prompts=10):
    """
    Track reconstruction differential across GRPO checkpoints.
    Shows WHEN the hiding computation emerges during training.
    """
    log.info("=" * 70)
    log.info("EXPERIMENT 2: TRAINING DYNAMICS")
    log.info("=" * 70)

    checkpoints = [50, 100, 150, 200, 250, 300]
    val_hint = load_dataset("grpo_val_faithful.json")[:n_prompts]

    results = {"checkpoints": {}, "config": {"n_prompts": n_prompts}}

    # Compute base model FVU once
    log.info("  Computing base model FVU baseline...")
    base_fvus = []
    for pi, example in enumerate(val_hint):
        text = format_prompt(example, tokenizer)
        fvu = compute_per_layer_fvu(model_base, clt, tokenizer, text)
        base_fvus.append(fvu)
    avg_base_fvu = np.mean(base_fvus, axis=0).tolist()
    results["base_fvu"] = avg_base_fvu

    for step in checkpoints:
        log.info(f"\n  Checkpoint step {step}...")
        ckpt_path = os.path.join(PROJECT, "grpo_checkpoints", f"checkpoint-{step}")

        if not os.path.exists(ckpt_path):
            log.warning(f"    Checkpoint {ckpt_path} not found, skipping")
            continue

        try:
            # Load and merge this checkpoint
            model_ckpt, _ = load_grpo_model(ckpt_path)

            ckpt_fvus = []
            for pi, example in enumerate(val_hint):
                text = format_prompt(example, tokenizer)
                fvu = compute_per_layer_fvu(model_ckpt, clt, tokenizer, text)
                ckpt_fvus.append(fvu)

            avg_ckpt_fvu = np.mean(ckpt_fvus, axis=0).tolist()
            fvu_diff = (np.array(avg_ckpt_fvu) - np.array(avg_base_fvu)).tolist()

            results["checkpoints"][str(step)] = {
                "avg_fvu_per_layer": avg_ckpt_fvu,
                "fvu_differential": fvu_diff,
                "total_differential": float(np.sum(np.abs(fvu_diff))),
            }

            log.info(f"    Total |differential|: {np.sum(np.abs(fvu_diff)):.4f}")
            top3 = sorted(range(26), key=lambda l: abs(fvu_diff[l]), reverse=True)[:3]
            for l in top3:
                log.info(f"      L{l}: diff={fvu_diff[l]:+.4f}")

            # Free memory
            del model_ckpt
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"    Checkpoint {step} failed: {e}")
            log.error(traceback.format_exc())

    save_results(results, "exp2_training_dynamics.json")
    return results


# ============================================================
# EXPERIMENT 3: Feature Analysis
# ============================================================

def run_exp3_features(model_base, model_grpo, clt, tokenizer, n_prompts=10):
    """
    Compare which CLT features are active in base vs GRPO'd model.
    Features that are significantly more active in GRPO'd model = candidate hiding features.
    """
    log.info("=" * 70)
    log.info("EXPERIMENT 3: FEATURE ANALYSIS")
    log.info("=" * 70)

    val_hint = load_dataset("grpo_val_faithful.json")[:n_prompts]
    results = {"prompts": []}

    for pi, example in enumerate(val_hint):
        text = format_prompt(example, tokenizer)
        log.info(f"\n  Prompt {pi+1}/{len(val_hint)}")

        try:
            inputs = tokenizer.encode(text, return_tensors="pt",
                                      add_special_tokens=True,
                                      truncation=True, max_length=512).to("cuda")
            tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

            # Base model features
            clt_in_b, _ = gather_clt_activations(model_base, GEMMA3_1B_NUM_LAYERS, inputs)
            if next(clt.parameters()).dtype == torch.float16:
                clt_in_b = clt_in_b.half()
            feats_base = clt.encode(clt_in_b)  # (seq, 26, 10080)

            # GRPO model features
            clt_in_g, _ = gather_clt_activations(model_grpo, GEMMA3_1B_NUM_LAYERS, inputs)
            if next(clt.parameters()).dtype == torch.float16:
                clt_in_g = clt_in_g.half()
            feats_grpo = clt.encode(clt_in_g)

            # Compare at last position (pre-generation)
            f_base = feats_base[-1].float()  # (26, 10080)
            f_grpo = feats_grpo[-1].float()

            delta = f_grpo - f_base

            # Find features that are significantly more active in GRPO'd model
            new_features = []  # active in GRPO but not in base
            amplified = []     # active in both but much stronger in GRPO
            suppressed = []    # active in base but weaker/gone in GRPO

            for layer in range(GEMMA3_1B_NUM_LAYERS):
                for feat_idx in range(delta.shape[1]):
                    d = delta[layer, feat_idx].item()
                    b = f_base[layer, feat_idx].item()
                    g = f_grpo[layer, feat_idx].item()

                    if abs(d) < 10:
                        continue

                    entry = {"layer": layer, "feature": feat_idx,
                             "base_act": round(b, 1), "grpo_act": round(g, 1),
                             "delta": round(d, 1)}

                    if b < 1 and g > 10:
                        new_features.append(entry)
                    elif b > 1 and g > b * 1.5 and d > 10:
                        amplified.append(entry)
                    elif b > 10 and g < b * 0.5:
                        suppressed.append(entry)

            # Sort by delta magnitude
            new_features.sort(key=lambda x: abs(x["delta"]), reverse=True)
            amplified.sort(key=lambda x: abs(x["delta"]), reverse=True)
            suppressed.sort(key=lambda x: abs(x["delta"]), reverse=True)

            prompt_result = {
                "prompt_idx": pi,
                "n_tokens": len(tokens),
                "question": example.get("question_text", "")[:100],
                "has_hint": example.get("has_hint", False),
                "new_features": new_features[:20],
                "amplified_features": amplified[:20],
                "suppressed_features": suppressed[:20],
                "total_new": len(new_features),
                "total_amplified": len(amplified),
                "total_suppressed": len(suppressed),
            }
            results["prompts"].append(prompt_result)

            log.info(f"    New: {len(new_features)}, Amplified: {len(amplified)}, "
                     f"Suppressed: {len(suppressed)}")
            if new_features:
                f = new_features[0]
                log.info(f"    Top new: L{f['layer']}/f{f['feature']}: "
                         f"base={f['base_act']}, grpo={f['grpo_act']}")

        except Exception as e:
            log.error(f"    Prompt {pi} failed: {e}")
            log.error(traceback.format_exc())

    # Aggregate: which features appear as "new" across multiple prompts?
    feat_counts = defaultdict(int)
    for p in results["prompts"]:
        for f in p.get("new_features", []):
            feat_counts[(f["layer"], f["feature"])] += 1

    consistent_new = [{"layer": k[0], "feature": k[1], "n_prompts": v}
                      for k, v in feat_counts.items() if v >= 2]
    consistent_new.sort(key=lambda x: x["n_prompts"], reverse=True)
    results["consistent_new_features"] = consistent_new[:30]

    log.info(f"\n  Features consistently new across prompts: {len(consistent_new)}")
    for f in consistent_new[:5]:
        log.info(f"    L{f['layer']}/f{f['feature']}: appears in {f['n_prompts']} prompts")

    save_results(results, "exp3_feature_analysis.json")
    return results


# ============================================================
# EXPERIMENT 4: Adversarial Validation
# ============================================================

def run_exp4_adversarial(model_base, model_grpo, clt, tokenizer, n_prompts=20):
    """
    Compare reconstruction differential on correct-hint vs wrong-hint prompts.
    If the pattern is the SAME → hiding circuit is about concealment, not reasoning.
    If DIFFERENT → some genuine reasoning is involved.
    """
    log.info("=" * 70)
    log.info("EXPERIMENT 4: ADVERSARIAL VALIDATION")
    log.info("=" * 70)

    val_correct = load_dataset("grpo_val_faithful.json")[:n_prompts]
    val_adversarial = load_dataset("grpo_val_adversarial.json")[:n_prompts]

    results = {"conditions": {}}

    for cond_name, examples in [("correct_hint", val_correct),
                                 ("wrong_hint", val_adversarial)]:
        log.info(f"\n  Condition: {cond_name} ({len(examples)} prompts)")

        fvu_diffs = []
        for pi, example in enumerate(examples):
            try:
                text = format_prompt(example, tokenizer)

                fvu_base = compute_per_layer_fvu(model_base, clt, tokenizer, text)
                fvu_grpo = compute_per_layer_fvu(model_grpo, clt, tokenizer, text)

                diff = [g - b for g, b in zip(fvu_grpo, fvu_base)]
                fvu_diffs.append(diff)

            except Exception as e:
                log.error(f"    Prompt {pi} failed: {e}")

        if fvu_diffs:
            avg_diff = np.mean(fvu_diffs, axis=0).tolist()
            std_diff = np.std(fvu_diffs, axis=0).tolist()

            results["conditions"][cond_name] = {
                "n_prompts": len(fvu_diffs),
                "avg_fvu_differential": avg_diff,
                "std_fvu_differential": std_diff,
                "total_abs_differential": float(np.sum(np.abs(avg_diff))),
                "all_diffs": [d for d in fvu_diffs],  # keep individual for stats
            }

            log.info(f"    Total |diff|: {np.sum(np.abs(avg_diff)):.4f}")
        else:
            results["conditions"][cond_name] = {"error": "no valid prompts"}

    # Compare correct vs adversarial
    if "correct_hint" in results["conditions"] and "wrong_hint" in results["conditions"]:
        correct = np.array(results["conditions"]["correct_hint"]["avg_fvu_differential"])
        wrong = np.array(results["conditions"]["wrong_hint"]["avg_fvu_differential"])
        correlation = float(np.corrcoef(correct, wrong)[0, 1])
        diff_of_diffs = (wrong - correct).tolist()

        results["comparison"] = {
            "correlation": correlation,
            "diff_of_diffs_per_layer": diff_of_diffs,
            "interpretation": (
                "High correlation → same circuit for correct and wrong hints (hiding is generic)"
                if correlation > 0.7
                else "Low correlation → different circuits (some genuine reasoning involved)"
            ),
        }
        log.info(f"\n  Correct vs Wrong hint correlation: {correlation:.3f}")
        log.info(f"  {results['comparison']['interpretation']}")

    save_results(results, "exp4_adversarial_validation.json")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Believe it or Not — Experiment Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 prompts, skip training dynamics")
    parser.add_argument("--n-prompts", type=int, default=20,
                        help="Number of prompts per condition")
    parser.add_argument("--only", type=str, default=None,
                        choices=["diff", "dynamics", "features", "adversarial"])
    args = parser.parse_args()

    n_prompts = 5 if args.quick else args.n_prompts

    log.info("=" * 70)
    log.info("BELIEVE IT OR NOT — COMPLETE EXPERIMENT SUITE")
    log.info(f"  Prompts per condition: {n_prompts}")
    log.info(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    log.info(f"  Time: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # Load base model
    model_base, tokenizer = load_base_model()

    # Load CLT
    log.info("Loading CLT-IT (affine)...")
    clt = load_clt(width="262k", l0="big", affine=True, variant="it",
                   device="cuda", half_precision=True, cache_dir=CACHE)

    log.info(f"GPU after base + CLT: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    t_start = time.time()

    # Load GRPO'd model (needed for exp 1, 3, 4)
    model_grpo = None
    if args.only is None or args.only in ["diff", "features", "adversarial"]:
        lora_path = os.path.join(PROJECT, "grpo_final_lora")
        model_grpo, _ = load_grpo_model(lora_path)
        log.info(f"GPU after all models + CLT: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # Exp 1: Reconstruction Differential
    if args.only is None or args.only == "diff":
        run_exp1_differential(model_base, model_grpo, clt, tokenizer, n_prompts)

    # Exp 2: Training Dynamics
    if (args.only is None or args.only == "dynamics") and not args.quick:
        # Free GRPO model to make room for checkpoint loading
        if model_grpo is not None:
            del model_grpo
            gc.collect()
            torch.cuda.empty_cache()
            model_grpo = None

        run_exp2_dynamics(model_base, clt, tokenizer, n_prompts=min(n_prompts, 10))

        # Reload GRPO model if needed for remaining experiments
        if args.only is None:
            lora_path = os.path.join(PROJECT, "grpo_final_lora")
            model_grpo, _ = load_grpo_model(lora_path)

    # Exp 3: Feature Analysis
    if args.only is None or args.only == "features":
        if model_grpo is None:
            lora_path = os.path.join(PROJECT, "grpo_final_lora")
            model_grpo, _ = load_grpo_model(lora_path)
        run_exp3_features(model_base, model_grpo, clt, tokenizer, n_prompts=min(n_prompts, 10))

    # Exp 4: Adversarial
    if args.only is None or args.only == "adversarial":
        if model_grpo is None:
            lora_path = os.path.join(PROJECT, "grpo_final_lora")
            model_grpo, _ = load_grpo_model(lora_path)
        run_exp4_adversarial(model_base, model_grpo, clt, tokenizer, n_prompts)

    total = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info(f"BELIEVE IT OR NOT COMPLETE — {total/60:.1f} min")
    log.info("=" * 70)
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            log.info(f"  {f}: {os.path.getsize(os.path.join(OUT, f))/1024:.0f} KB")


if __name__ == "__main__":
    main()
