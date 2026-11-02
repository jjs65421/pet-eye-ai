"""Three-way decisions. Scores are not clinical probabilities or confidence."""
import numpy as np


def metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive = scores >= threshold
    tp = int(np.sum(positive & (labels == 1)))
    fp = int(np.sum(positive & (labels == 0)))
    tn = int(np.sum(~positive & (labels == 0)))
    fn = int(np.sum(~positive & (labels == 1)))
    return {"samples": len(labels), "positive_samples": tp + fn, "negative_samples": tn + fp,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None}


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    n1, n0 = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def select_thresholds(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if min(int(np.sum(labels == 0)), int(np.sum(labels == 1))) < 30:
        return None
    thresholds = np.unique(np.r_[0, scores, 1])
    negative = np.sort(scores[labels == 0])
    positive = np.sort(scores[labels == 1])
    specificity = np.searchsorted(negative, thresholds, side="left") / len(negative)
    sensitivity = 1 - np.searchsorted(positive, thresholds, side="left") / len(positive)
    high_candidates = np.flatnonzero((specificity >= .95) & (sensitivity >= .50))
    low_candidates = np.flatnonzero((sensitivity >= .95) & (specificity >= .10))
    if not len(high_candidates) or not len(low_candidates):
        return None
    high = float(thresholds[high_candidates[0]])
    low = float(thresholds[low_candidates[-1]])
    return {"low": min(low, high), "high": high}


def validate_cutoffs(labels, scores, cutoffs):
    if cutoffs is None:
        return {"approved": False, "reason": "calibration_did_not_meet_policy"}
    high = metrics(labels, scores, cutoffs["high"])
    low = metrics(labels, scores, cutoffs["low"])
    roc_auc = auc(labels, scores)
    approved = (min(high["positive_samples"], high["negative_samples"]) >= 30
                and roc_auc is not None and roc_auc >= .80
                and high["specificity"] >= .90 and high["sensitivity"] >= .50
                and low["sensitivity"] >= .90 and low["specificity"] >= .10)
    return {"approved": bool(approved), "roc_auc": roc_auc, "high_cutoff": high,
            "low_cutoff": low, "reason": "passed" if approved else "independent_test_did_not_meet_policy"}


def finding(disease, score, info):
    cutoffs = info.get("thresholds")
    approved = info.get("evaluation", {}).get("approved", False)
    if not approved or not cutoffs or not np.isfinite(score):
        decision = "inconclusive"
    elif score >= cutoffs["high"]:
        decision = "possible_abnormality"
    elif score < cutoffs["low"]:
        decision = "below_screening_threshold"
    else:
        decision = "inconclusive"
    return {"disease": disease, "score": round(float(score), 4),
            "decision": decision, "thresholds": cutoffs, "validated": bool(approved)}


def summarize(findings):
    flagged = [row["disease"] for row in findings if row["decision"] == "possible_abnormality"]
    epiphora = next((row for row in findings if row['disease'] == '유루증'), None)
    ranked = sorted(findings, key=lambda row: row['score'], reverse=True)
    return {"result": "hybrid_screening" if findings else "inconclusive",
            "flagged_diseases": flagged, "needs_visit": bool(ranked) or bool(flagged),
            "top_finding": ranked[0] if ranked else None,
            "epiphora_finding": epiphora,
            "ranked_findings": ranked,
            "findings": ranked,
            "score_meaning": "independent_model_score_not_disease_probability",
            "ranking_notice": "모델별 점수 척도가 달라 순위가 실제 질환 가능성 순서를 뜻하지 않습니다."}
