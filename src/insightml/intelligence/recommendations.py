"""Algorithm Recommender — rules engine mapping EDA findings to model families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Algorithm profiles
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmProfile:
    """Scoring profile for one algorithm family."""
    name: str
    handles_nonlinear: bool = False
    handles_high_dim: bool = False
    handles_missing: bool = False       # natively (no imputation needed)
    handles_categorical: bool = False   # natively
    is_interpretable: bool = False
    is_fast: bool = False
    min_samples: int = 30              # below this, model is unreliable
    base_score: int = 50


_PROFILES: list[AlgorithmProfile] = [
    AlgorithmProfile("LinearModels", handles_high_dim=True, is_interpretable=True,
                     is_fast=True, base_score=50),
    AlgorithmProfile("TreeEnsembles", handles_nonlinear=True, handles_categorical=True,
                     handles_missing=True, base_score=50),
    AlgorithmProfile("GradientBoosting", handles_nonlinear=True, handles_categorical=True,
                     handles_missing=True, base_score=55),
    AlgorithmProfile("SVM_Kernel", handles_nonlinear=True, handles_high_dim=True,
                     base_score=45),
    AlgorithmProfile("NeuralNetwork", handles_nonlinear=True, handles_high_dim=True,
                     base_score=40),
    AlgorithmProfile("KNearestNeighbors", handles_nonlinear=True, is_interpretable=True,
                     base_score=40, min_samples=50),
    AlgorithmProfile("NaiveBayes", handles_high_dim=True, is_fast=True,
                     is_interpretable=True, base_score=35),
]


# ---------------------------------------------------------------------------
# Recommendation result
# ---------------------------------------------------------------------------

@dataclass
class RecommendationResult:
    """Ranked list of algorithm family recommendations."""

    ranked: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)

    def top(self, n: int = 3) -> list[str]:
        """Return top-N algorithm family names."""
        return [r["algorithm"] for r in self.ranked[:n]]

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td>{r['rank']}</td><td>{r['algorithm']}</td>"
            f"<td>{r['score']}</td><td>{r['notes']}</td></tr>"
            for r in self.ranked
        )
        reasoning_html = "".join(f"<li>{r}</li>" for r in self.reasoning)
        return (
            "<h3>Algorithm Recommendations</h3>"
            "<table border='1' style='border-collapse:collapse;font-size:13px'>"
            "<tr><th>#</th><th>Algorithm</th><th>Score</th><th>Notes</th></tr>"
            f"{rows}</table>"
            f"<h4>Reasoning</h4><ul>{reasoning_html}</ul>"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_algorithms(
    n_samples: int,
    n_features: int,
    task: str,
    *,
    has_nonlinear: bool = False,
    has_high_cardinality: bool = False,
    has_missing: bool = False,
    is_small_dataset: bool = False,
    is_large_dataset: bool = False,
    minority_ratio: float = 1.0,
    readiness_score: float = 100.0,
) -> RecommendationResult:
    """Score and rank algorithm families based on data characteristics.

    Args:
        n_samples: Number of training samples.
        n_features: Number of features.
        task: ``"classification"`` or ``"regression"``.
        has_nonlinear: EDA detected non-linear feature-target relationships.
        has_high_cardinality: Dataset contains high-cardinality categorical columns.
        has_missing: Dataset has missing values.
        is_small_dataset: n_samples < 500.
        is_large_dataset: n_samples > 100_000.
        minority_ratio: min_class / max_class (classification); 1.0 = balanced.
        readiness_score: Output of :func:`~insightml.intelligence.readiness.compute_readiness`.

    Returns:
        :class:`RecommendationResult` with ranked algorithms.
    """
    reasoning: list[str] = []
    scored: list[tuple[AlgorithmProfile, int, list[str]]] = []

    for profile in _PROFILES:
        score = profile.base_score
        notes: list[str] = []

        # Non-linearity
        if has_nonlinear:
            if profile.handles_nonlinear:
                score += 15
                notes.append("+15 non-linear detected")
            else:
                score -= 20
                notes.append("-20 linear model, non-linear data")

        # High dimensionality
        if n_features > 50:
            if profile.handles_high_dim:
                score += 10
                notes.append("+10 handles high-dim")
            else:
                score -= 5
                notes.append("-5 struggles with high-dim")

        # Missing values
        if has_missing:
            if profile.handles_missing:
                score += 5
                notes.append("+5 native missing handling")

        # High cardinality
        if has_high_cardinality:
            if profile.handles_categorical:
                score += 5
                notes.append("+5 native categorical")

        # Small dataset
        if is_small_dataset:
            if profile.name in ("NeuralNetwork", "GradientBoosting"):
                score -= 15
                notes.append("-15 may overfit on small data")
            if profile.is_fast or profile.is_interpretable:
                score += 5
                notes.append("+5 fast/simple for small data")

        # Large dataset
        if is_large_dataset:
            if profile.name in ("SVM_Kernel", "KNearestNeighbors"):
                score -= 15
                notes.append("-15 slow on large data")
            if profile.name in ("GradientBoosting", "LinearModels"):
                score += 5
                notes.append("+5 scales well")

        # Imbalanced classes
        if minority_ratio < 0.3 and task == "classification":
            if profile.name in ("TreeEnsembles", "GradientBoosting"):
                score += 5
                notes.append("+5 supports class_weight")

        # Readiness penalty — if data is poor, prefer simpler models
        if readiness_score < 70:
            if profile.name in ("NeuralNetwork",):
                score -= 10
                notes.append("-10 needs clean data")
            if profile.is_interpretable:
                score += 5
                notes.append("+5 interpretable = easier debugging with poor data")

        # Sample size guard
        if n_samples < profile.min_samples:
            score -= 20
            notes.append(f"-20 too few samples (need {profile.min_samples}+)")

        scored.append((profile, max(0, score), notes))

    scored.sort(key=lambda x: x[1], reverse=True)

    ranked = [
        {
            "rank": i + 1,
            "algorithm": p.name,
            "score": s,
            "notes": "; ".join(n) if n else "default",
        }
        for i, (p, s, n) in enumerate(scored)
    ]

    # Summarise reasoning
    if has_nonlinear:
        reasoning.append("Non-linear relationships detected → tree-based / kernel models preferred.")
    if is_small_dataset:
        reasoning.append(f"Small dataset (n={n_samples}) → prefer simpler, regularised models.")
    if is_large_dataset:
        reasoning.append(f"Large dataset (n={n_samples}) → avoid O(n²) models (SVM, KNN).")
    if has_missing:
        reasoning.append("Missing values present → tree models with native missing support have an edge.")
    if minority_ratio < 0.3:
        reasoning.append(f"Severe class imbalance (ratio={minority_ratio:.2f}) → use class_weight.")
    if not reasoning:
        reasoning.append("No strong signals — all families are competitive; run iml.battle() to compare empirically.")

    return RecommendationResult(ranked=ranked, reasoning=reasoning)
