#!/usr/bin/env python3
"""Compare evaluation results between two model versions.

Produces a structured comparison report with APPROVE/REJECT/NEEDS_REVIEW recommendation.

Usage:
    python compare_models.py \
        --baseline workspace/eval_results/v0.0.1/eval_metrics.json \
        --candidate workspace/eval_results/v0.1.0-math/eval_metrics.json \
        --output workspace/eval_results/v0.1.0-math/report.yaml \
        --target-weakness math_reasoning
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Regression thresholds (from model-benchmarking skill spec)
HARD_REJECT_THRESHOLD = -0.02     # Any benchmark drops > 2%
SOFT_FLAG_THRESHOLD = -0.005      # Any benchmark drops > 0.5%
TARGET_IMPROVEMENT_MIN = 0.03     # Target weakness must improve >= 3%


def load_metrics(filepath):
    """Load evaluation metrics from JSON."""
    with open(filepath) as f:
        return json.load(f)


def compare(baseline, candidate, target_weakness=None):
    """Compare two sets of metrics and produce a report."""
    comparisons = []
    regressions = []
    improvements = []

    # Get common metric keys
    all_keys = set(baseline.keys()) | set(candidate.keys())
    metric_keys = [k for k in all_keys if isinstance(baseline.get(k, 0), (int, float))
                   and isinstance(candidate.get(k, 0), (int, float))]

    for key in sorted(metric_keys):
        base_val = baseline.get(key, 0)
        cand_val = candidate.get(key, 0)

        if base_val == 0:
            delta = cand_val
            delta_pct = None
        else:
            delta = cand_val - base_val
            delta_pct = round(delta / abs(base_val), 4)

        status = "neutral"
        if delta_pct is not None:
            if delta_pct > 0.005:
                status = "improved"
                improvements.append(key)
            elif delta_pct < -0.005:
                status = "regressed"
                regressions.append(key)

        comparisons.append({
            "metric": key,
            "baseline": round(base_val, 4) if isinstance(base_val, float) else base_val,
            "candidate": round(cand_val, 4) if isinstance(cand_val, float) else cand_val,
            "delta": round(delta, 4),
            "delta_pct": delta_pct,
            "status": status,
        })

    # Determine recommendation
    recommendation = "APPROVE"
    reasons = []

    # Check for hard regressions
    hard_regressions = [c for c in comparisons
                        if c["delta_pct"] is not None and c["delta_pct"] < HARD_REJECT_THRESHOLD]
    if hard_regressions:
        recommendation = "REJECT"
        for r in hard_regressions:
            reasons.append(f"Hard regression on {r['metric']}: {r['delta_pct']:.1%}")

    # Check for soft regressions
    soft_regressions = [c for c in comparisons
                        if c["delta_pct"] is not None
                        and HARD_REJECT_THRESHOLD <= c["delta_pct"] < SOFT_FLAG_THRESHOLD]
    if soft_regressions and recommendation != "REJECT":
        recommendation = "NEEDS_REVIEW"
        for r in soft_regressions:
            reasons.append(f"Soft regression on {r['metric']}: {r['delta_pct']:.1%}")

    # Check target improvement (if specified)
    if target_weakness:
        target_metrics = [c for c in comparisons if target_weakness.lower() in c["metric"].lower()]
        if target_metrics:
            best_improvement = max(c["delta_pct"] or 0 for c in target_metrics)
            if best_improvement < TARGET_IMPROVEMENT_MIN:
                if recommendation == "APPROVE":
                    recommendation = "NEEDS_REVIEW"
                reasons.append(
                    f"Target weakness '{target_weakness}' improvement "
                    f"({best_improvement:.1%}) below minimum ({TARGET_IMPROVEMENT_MIN:.0%})"
                )

    if not reasons:
        reasons.append("All metrics within acceptable range")

    return {
        "recommendation": recommendation,
        "reasons": reasons,
        "comparisons": comparisons,
        "summary": {
            "total_metrics": len(comparisons),
            "improved": len(improvements),
            "regressed": len(regressions),
            "neutral": len(comparisons) - len(improvements) - len(regressions),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument("--baseline", required=True, help="Baseline eval_metrics.json")
    parser.add_argument("--candidate", required=True, help="Candidate eval_metrics.json")
    parser.add_argument("--output", required=True, help="Output report YAML path")
    parser.add_argument("--target-weakness", default=None, help="Target weakness to check for improvement")
    args = parser.parse_args()

    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)

    report = compare(baseline, candidate, args.target_weakness)
    report["baseline_path"] = args.baseline
    report["candidate_path"] = args.candidate
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    report["target_weakness"] = args.target_weakness

    # Write report
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Recommendation: {report['recommendation']}")
    for reason in report["reasons"]:
        logger.info(f"  - {reason}")
    logger.info(f"Summary: {report['summary']}")

    print(f"__EVAL_REPORT__:{json.dumps({'recommendation': report['recommendation'], 'reasons': report['reasons'], 'summary': report['summary']})}")


if __name__ == "__main__":
    main()
