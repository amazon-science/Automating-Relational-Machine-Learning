"""LLM-based post-selection for HPO trial candidates."""

import json
import os
import re

import numpy as np
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LLM Client Abstraction
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper over Anthropic and Bedrock Claude APIs."""

    def __init__(self, backend: str = "anthropic", model: str = "claude-sonnet-4-5-20250929"):
        self.backend = backend
        self.model = model
        self._client = None

    def _init_anthropic(self):
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    def _init_bedrock(self):
        import boto3
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        if self.backend == "anthropic":
            return self._query_anthropic(system_prompt, user_prompt, max_tokens)
        elif self.backend == "bedrock":
            return self._query_bedrock(system_prompt, user_prompt, max_tokens)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def _query_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if self._client is None:
            self._init_anthropic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _query_bedrock(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if self._client is None:
            self._init_bedrock()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        })
        response = self._client.invoke_model(
            body=body,
            modelId=self.model,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert machine learning engineer specializing in hyperparameter \
optimization and model selection. Your task is to select the best trial from \
a set of candidate trials based on their validation performance and loss \
landscape properties.

You will be given structured data about each candidate and must return your \
selection as a JSON object."""


def _format_metric_value(val: Any, metric_name: str) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    return f"{float(val):.6f}"


def build_user_prompt(
    candidates: List[Tuple[Dict, Dict]],
    task_type: str,
    val_metric_fn,
) -> str:
    """Construct the user prompt describing the HPO selection problem.

    Args:
        candidates: List of (trial_dict, landscape_metrics_dict) tuples.
        task_type: One of "binary_classification", "multiclass_classification", "regression".
        val_metric_fn: Callable that extracts the primary validation score from a trial dict.
    """
    if task_type == "binary_classification":
        primary_metric = "ROC-AUC (higher is better)"
        direction = "higher"
    elif task_type == "multiclass_classification":
        primary_metric = "Accuracy (higher is better)"
        direction = "higher"
    elif task_type == "regression":
        primary_metric = "MAE (lower is better)"
        direction = "lower"
    else:
        primary_metric = "Unknown"
        direction = "higher"

    lines = [
        "## HPO Trial Selection Task",
        "",
        f"**Task type**: {task_type}",
        f"**Primary validation metric**: {primary_metric}",
        f"**Number of candidates**: {len(candidates)}",
        "",
        "### Background",
        "",
        "These candidates are the top-performing trials from a hyperparameter "
        "optimization run over neural network models for relational data. "
        "The goal is to select the single best trial that will generalize well "
        "to the test set.",
        "",
        "Selection should balance two factors:",
        f"1. **Validation performance** ({primary_metric}): "
        f"A {direction} value indicates better fit on the validation set.",
        "2. **Loss landscape smoothness**: Smoother landscapes (lower Lipschitz "
        "constant, lower sharpness, lower barrier) correlate with better "
        "generalization. A model in a flat minimum is more likely to generalize.",
        "",
        "### Landscape Metrics Explanation",
        "- **Lipschitz max**: Maximum gradient norm in the neighborhood. "
        "Lower means smoother landscape, better generalization.",
        "- **Sharpness lambda max**: Largest eigenvalue of the Hessian. "
        "Lower means flatter minimum, better generalization.",
        "- **Barrier overall max**: Maximum loss barrier when interpolating "
        "to nearby solutions. Lower means more robust minimum.",
        "",
        "### Candidates",
        "",
    ]

    for i, (trial, lm) in enumerate(candidates):
        val_score = val_metric_fn(trial)
        val_metric_dict = trial.get("val_metric", {}) or {}

        lines.append(f"#### Candidate {i}")
        lines.append(f"- Validation score: {_format_metric_value(val_score, 'val')}")

        if isinstance(val_metric_dict, dict):
            metric_parts = []
            for k, v in sorted(val_metric_dict.items()):
                metric_parts.append(f"{k}={_format_metric_value(v, k)}")
            if metric_parts:
                lines.append(f"- All validation metrics: {', '.join(metric_parts)}")

        lines.append(
            f"- Lipschitz max: {_format_metric_value(lm.get('lipschitz_max'), 'lipschitz')}"
        )
        lines.append(
            f"- Sharpness lambda max: {_format_metric_value(lm.get('sharpness_lambda_max'), 'sharpness')}"
        )
        lines.append(
            f"- Barrier overall max: {_format_metric_value(lm.get('barrier_overall_max'), 'barrier')}"
        )

        model_type = trial.get("model_type", "unknown")
        lines.append(f"- Model type: {model_type}")
        lines.append("")

    lines.extend([
        "### Instructions",
        "",
        "Select the single best candidate by its index (0-based). Consider both "
        "validation performance and landscape smoothness. A candidate with slightly "
        f"{'lower' if direction == 'higher' else 'higher'} validation score but "
        "significantly smoother landscape may generalize better.",
        "",
        "Return ONLY a JSON object with the following format:",
        '```json',
        '{"selected_index": <int>, "reasoning": "<brief explanation>"}',
        '```',
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str, num_candidates: int) -> Optional[int]:
    """Extract the selected index from the LLM response."""
    # Try JSON in code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            idx = int(data["selected_index"])
            if 0 <= idx < num_candidates:
                return idx
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    # Try raw JSON
    try:
        data = json.loads(response_text.strip())
        idx = int(data["selected_index"])
        if 0 <= idx < num_candidates:
            return idx
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Regex fallback
    idx_match = re.search(r'"selected_index"\s*:\s*(\d+)', response_text)
    if idx_match:
        idx = int(idx_match.group(1))
        if 0 <= idx < num_candidates:
            return idx

    logger.warning("Failed to parse LLM response for trial selection")
    return None


# ---------------------------------------------------------------------------
# Drop-in replacement for choose_best_trial_by_landscape
# ---------------------------------------------------------------------------

def choose_best_trial_by_llm(
    trial_results: List[Dict],
    task_type: str = "binary_classification",
    top_k: int = 10,
    metrics_provider=None,
    llm_backend: str = "anthropic",
    llm_model: str = "claude-sonnet-4-5-20250929",
) -> Optional[Dict]:
    """Select the best trial using an LLM, with voting fallback.

    Same return structure as ``choose_best_trial_by_landscape``.
    """
    from utils.hpo import (
        val_score_of_result,
        compute_landscape_metrics_for_trial,
        choose_best_trial_by_landscape,
    )

    if not trial_results:
        return None

    # Step 1: score and sort by validation, take top-k
    scored = []
    for r in trial_results:
        s = val_score_of_result(r, task_type)
        if s is None:
            continue
        scored.append((s, r))
    if not scored:
        return None

    reverse = "classification" in task_type
    scored.sort(key=lambda x: x[0], reverse=reverse)
    candidates = [r for _, r in scored[: max(1, top_k)]]

    def metrics_for(r):
        return metrics_provider(r) if metrics_provider is not None else compute_landscape_metrics_for_trial(r)

    metrics_list = []
    for r in candidates:
        m = metrics_for(r)
        if not m:
            continue
        metrics_list.append((r, m))

    if not metrics_list:
        return candidates[0] if candidates else None

    # Step 2: compute val_best and best_test_trial (same logic as voting)
    def _val_metric_for(result):
        vm = result.get("val_metric", {}) or {}
        if isinstance(vm, dict):
            if task_type == "binary_classification":
                return float(vm.get("roc_auc", -np.inf))
            if task_type == "multiclass_classification":
                return float(vm.get("accuracy", -np.inf))
            if "regression" in str(task_type):
                return -float(vm.get("mae", np.inf))
            if "classification" in str(task_type):
                return float(vm.get("roc_auc", -np.inf))
            return None
        return float(vm) if vm is not None else None

    val_scores = []
    for cand, _ in metrics_list:
        score = _val_metric_for(cand)
        if score is not None:
            val_scores.append((score, cand))
    val_best = max(val_scores, key=lambda t: t[0])[1] if val_scores else candidates[0]

    def _test_metric_values(result):
        tm = result.get("test_metric") if isinstance(result, dict) else None
        if not isinstance(tm, dict) or not tm:
            return None, None
        if task_type == "binary_classification":
            val = tm.get("roc_auc")
            return (val, val)
        if task_type == "multiclass_classification":
            val = tm.get("accuracy")
            return (val, val)
        raw = tm.get("mae")
        if raw is None:
            return None, None
        return (-raw, raw)

    best_test_trial = None
    best_test_metric = None
    best_test_score = -np.inf
    for trial in trial_results:
        score, raw_val = _test_metric_values(trial)
        if score is None:
            continue
        if score > best_test_score:
            best_test_score = score
            best_test_metric = raw_val
            best_test_trial = trial

    # Step 3: ask LLM
    llm_selected = None
    try:
        client = LLMClient(backend=llm_backend, model=llm_model)
        user_prompt = build_user_prompt(metrics_list, task_type, _val_metric_for)
        logger.info(f"Sending {len(metrics_list)} candidates to LLM ({llm_backend}/{llm_model})")
        response_text = client.query(SYSTEM_PROMPT, user_prompt)
        logger.info(f"LLM response: {response_text[:500]}")
        llm_idx = parse_llm_response(response_text, len(metrics_list))
        if llm_idx is not None:
            llm_selected = metrics_list[llm_idx][0]
            logger.info(f"LLM selected candidate {llm_idx} out of {len(metrics_list)}")
        else:
            logger.warning("LLM response parsing failed; falling back to voting")
    except Exception as exc:
        logger.warning(f"LLM selection failed ({exc}); falling back to voting")

    # Step 4: fallback to voting if LLM failed
    if llm_selected is None:
        fallback = choose_best_trial_by_landscape(
            trial_results, task_type=task_type, top_k=top_k,
            metrics_provider=metrics_provider,
        )
        return fallback

    # Step 5: build return dict matching voting structure
    weighted_scores = {id(r): 0.0 for r, _ in metrics_list}
    weighted_scores[id(llm_selected)] = 1.0

    return {
        "validation_top": val_best,
        "voted_top": llm_selected,
        "best_test_trial": best_test_trial,
        "best_test_metric": best_test_metric,
        "finalists": metrics_list,
        "weighted_scores": weighted_scores,
    }
