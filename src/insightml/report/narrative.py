"""Template-driven natural-language narrative summaries (no LLM dependency)."""

from __future__ import annotations


def executive_summary(
    task: str,
    target: str,
    n_samples: int,
    n_features: int,
    best_model: str | None,
    best_score: float | None,
    primary_metric: str,
    readiness_score: float | None = None,
    readiness_grade: str | None = None,
    n_leakage_warnings: int = 0,
    pareto_models: list[str] | None = None,
    significant_pairs: list[tuple[str, str, float]] | None = None,
) -> str:
    """Generate a concise executive summary paragraph.

    Args:
        task: "classification" or "regression".
        target: Target column name.
        n_samples: Number of training rows.
        n_features: Number of feature columns.
        best_model: Name of best-performing model.
        best_score: Best primary metric value.
        primary_metric: Metric name.
        readiness_score: Data readiness 0-100.
        readiness_grade: Letter grade A-F.
        n_leakage_warnings: Number of leakage warnings.
        pareto_models: Names of Pareto-optimal models.
        significant_pairs: (model_a, model_b, p_value) for significant pairs.

    Returns:
        Multi-sentence summary string.
    """
    lines: list[str] = []

    # Dataset
    lines.append(
        f"The dataset contains {n_samples:,} samples and {n_features} features "
        f"for a {task} task predicting '{target}'."
    )

    # Readiness
    if readiness_score is not None:
        lines.append(
            f"Data readiness score: {readiness_score:.0f}/100 (Grade {readiness_grade})."
        )
    if n_leakage_warnings > 0:
        lines.append(
            f"⚠ {n_leakage_warnings} potential leakage warning(s) detected — review before deployment."
        )

    # Best model
    if best_model and best_score is not None:
        lines.append(
            f"Best model: {best_model} achieved {primary_metric}={best_score:.4f} "
            f"via {5}-fold cross-validation."
        )

    # Pareto
    if pareto_models:
        lines.append(
            f"Pareto-optimal models (best accuracy-speed tradeoff): {', '.join(pareto_models)}."
        )

    # Statistical significance
    if significant_pairs:
        n_sig = len(significant_pairs)
        lines.append(
            f"{n_sig} model pair(s) are statistically significantly different (p<0.05)."
        )

    return " ".join(lines)


def model_narrative(
    model_name: str,
    metrics: dict[str, float],
    primary_metric: str,
    rank: int,
    n_models: int,
    is_pareto: bool = False,
    train_time: float = 0.0,
) -> str:
    """One-line narrative for a model in the leaderboard.

    Args:
        model_name: Name of the model.
        metrics: Dict of metric -> score.
        primary_metric: Key metric name.
        rank: 1-based rank.
        n_models: Total number of models.
        is_pareto: Whether this model is on the Pareto front.
        train_time: Training time in seconds.

    Returns:
        One-line narrative string.
    """
    score = metrics.get(primary_metric, 0.0)
    pareto_tag = " (Pareto optimal)" if is_pareto else ""
    time_str = f"{train_time:.1f}s"
    return (
        f"#{rank}/{n_models}: {model_name}{pareto_tag} — "
        f"{primary_metric}={score:.4f}, trained in {time_str}."
    )


def data_recommendations(
    readiness_score: float,
    leakage_columns: list[str],
    high_vif_columns: list[str],
    missing_pct: float,
    imbalance_severity: str | None = None,
) -> list[str]:
    """Generate actionable recommendations based on data quality findings.

    Returns:
        List of recommendation strings.
    """
    recs: list[str] = []

    if leakage_columns:
        recs.append(
            f"Leakage risk: Investigate columns {leakage_columns[:5]} — "
            "they correlate near-perfectly with the target."
        )

    if high_vif_columns:
        recs.append(
            f"Multicollinearity: {high_vif_columns[:5]} have VIF>=10. "
            "Consider removing or combining via PCA."
        )

    if missing_pct > 0.1:
        recs.append(
            f"Missing data: {missing_pct:.1%} of values are missing. "
            "Apply KNN or iterative imputation."
        )

    if imbalance_severity in ("moderate", "severe"):
        recs.append(
            f"Class imbalance ({imbalance_severity}): Use class_weight='balanced', "
            "SMOTE oversampling, or adjust decision threshold."
        )

    if readiness_score < 70:
        recs.append(
            f"Data readiness is low ({readiness_score:.0f}/100). "
            "Address the above issues before production deployment."
        )

    if not recs:
        recs.append("No major data quality issues detected. Dataset is ready for modelling.")

    return recs


def ensemble_recommendation(
    ensemble_candidates: list[tuple[str, str, float]],
    best_model: str | None,
    pareto_models: list[str],
) -> str:
    """Recommendation about ensembling or model selection.

    Returns:
        Recommendation string.
    """
    if not best_model:
        return "No models available for recommendation."

    if len(pareto_models) == 1:
        return (
            f"Only one model dominates the Pareto front: {pareto_models[0]}. "
            "It offers the best accuracy-speed tradeoff."
        )

    if ensemble_candidates:
        a, b, score = ensemble_candidates[0]
        return (
            f"Models {a} and {b} are highly complementary (complementarity={score:.2f}). "
            "An ensemble (voting or stacking) may outperform either alone."
        )

    return (
        f"{best_model} is the top performer. Consider it as the primary model, "
        "and verify results on a held-out test set before deployment."
    )
