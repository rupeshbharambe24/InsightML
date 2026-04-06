"""Cross-model error analysis — disagreement, hard samples, complementarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dissectml.battle.result import BattleResult
from dissectml.viz.theme import QUALITATIVE, make_figure


def analyze_errors(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
    X: pd.DataFrame | None = None,
    hard_sample_pct: float = 0.10,
) -> ErrorAnalysisResult:
    """Run full cross-model error analysis.

    Args:
        battle_result: BattleResult with OOF predictions.
        y_true: Ground-truth labels/values.
        X: Optional feature DataFrame for profiling hard vs easy samples.
        hard_sample_pct: Bottom fraction of samples (by model accuracy) = hard samples.

    Returns:
        :class:`ErrorAnalysisResult`
    """
    scores = [s for s in battle_result.successful if s.oof_predictions is not None]
    if not scores:
        return ErrorAnalysisResult(
            task=battle_result.task, models=[], disagreement=pd.DataFrame(),
            complementarity=pd.DataFrame(), hard_indices=np.array([]),
            hard_sample_profile=pd.DataFrame(),
        )

    y_arr = np.asarray(y_true)
    task = battle_result.task
    names = [s.name for s in scores]
    n = len(y_arr)

    # Build correct/error matrix: shape (n_samples, n_models)
    # For classification: 1 if correct, 0 if wrong
    # For regression: -abs_error (higher = better)
    correct = np.full((n, len(scores)), np.nan)
    for j, score in enumerate(scores):
        preds = score.oof_predictions
        mask = ~np.isnan(preds)
        if task == "classification":
            correct[mask, j] = (preds[mask].round() == y_arr[mask]).astype(float)
        else:
            correct[mask, j] = -np.abs(preds[mask].astype(float) - y_arr[mask].astype(float))

    # --- Disagreement matrix ---
    disagree = np.zeros((len(scores), len(scores)))
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            mask = ~(np.isnan(correct[:, i]) | np.isnan(correct[:, j]))
            if mask.sum() == 0:
                continue
            if task == "classification":
                rate = float(np.mean(correct[mask, i] != correct[mask, j]))
            else:
                # For regression, disagreement = correlation of sign of errors
                e_i = correct[mask, i]
                e_j = correct[mask, j]
                rate = 1 - float(abs(np.corrcoef(e_i, e_j)[0, 1]))
            disagree[i, j] = round(rate, 4)
            disagree[j, i] = round(rate, 4)

    disagree_df = pd.DataFrame(disagree, index=names, columns=names).round(4)

    # --- Complementarity matrix ---
    # C[A][B] = fraction where A right but B wrong (how much A adds over B)
    compl = np.zeros((len(scores), len(scores)))
    if task == "classification":
        for i in range(len(scores)):
            for j in range(len(scores)):
                if i == j:
                    continue
                mask = ~(np.isnan(correct[:, i]) | np.isnan(correct[:, j]))
                if mask.sum() == 0:
                    continue
                a_right_b_wrong = (correct[mask, i] == 1) & (correct[mask, j] == 0)
                compl[i, j] = round(float(np.mean(a_right_b_wrong)), 4)

    compl_df = pd.DataFrame(compl, index=names, columns=names).round(4)

    # --- Hard samples ---
    # For each sample, count how many models got it wrong
    if task == "classification":
        per_sample_wrong = np.nanmean(correct == 0, axis=1)
    else:
        # Normalise per-sample error across models
        mean_errors = -np.nanmean(correct, axis=1)  # positive = high error
        mn, mx = np.nanmin(mean_errors), np.nanmax(mean_errors)
        per_sample_wrong = (mean_errors - mn) / max(mx - mn, 1e-10)

    threshold = np.nanpercentile(per_sample_wrong, 100 * (1 - hard_sample_pct))
    hard_indices = np.where(per_sample_wrong >= threshold)[0]

    # --- Hard sample profile ---
    hard_profile = pd.DataFrame()
    if X is not None and len(hard_indices) > 0:
        hard_profile = _profile_hard_vs_easy(X, hard_indices, n)

    return ErrorAnalysisResult(
        task=task,
        models=names,
        disagreement=disagree_df,
        complementarity=compl_df,
        hard_indices=hard_indices,
        hard_sample_profile=hard_profile,
        per_sample_difficulty=per_sample_wrong,
    )


class ErrorAnalysisResult:
    """Results of cross-model error analysis."""

    def __init__(
        self,
        task: str,
        models: list[str],
        disagreement: pd.DataFrame,
        complementarity: pd.DataFrame,
        hard_indices: np.ndarray,
        hard_sample_profile: pd.DataFrame,
        per_sample_difficulty: np.ndarray | None = None,
    ) -> None:
        self.task = task
        self.models = models
        self.disagreement = disagreement
        self.complementarity = complementarity
        self.hard_indices = hard_indices
        self.hard_sample_profile = hard_sample_profile
        self.per_sample_difficulty = per_sample_difficulty

    def disagreement_figure(self) -> go.Figure:
        """Heatmap of pairwise model disagreement rates."""
        if self.disagreement.empty:
            return make_figure("No disagreement data")
        fig = make_figure(title="Model Disagreement Rate")
        fig.add_trace(go.Heatmap(
            z=self.disagreement.values,
            x=self.models, y=self.models,
            colorscale="Reds", zmin=0, zmax=1,
            text=self.disagreement.values,
            texttemplate="%{text:.2f}",
            colorbar={"title": "Disagreement"},
        ))
        fig.update_layout(height=max(350, len(self.models) * 50 + 100))
        return fig

    def complementarity_figure(self) -> go.Figure:
        """Heatmap of model complementarity (A correct, B wrong)."""
        if self.complementarity.empty:
            return make_figure("No complementarity data (regression not supported)")
        fig = make_figure(title="Model Complementarity (row correct, col wrong)")
        fig.add_trace(go.Heatmap(
            z=self.complementarity.values,
            x=self.models, y=self.models,
            colorscale="Blues", zmin=0,
            text=self.complementarity.values,
            texttemplate="%{text:.2f}",
            colorbar={"title": "Fraction"},
        ))
        fig.update_layout(height=max(350, len(self.models) * 50 + 100))
        return fig

    def hard_sample_figure(self, n_bins: int = 20) -> go.Figure:
        """Histogram of per-sample difficulty scores."""
        if self.per_sample_difficulty is None:
            return make_figure("No difficulty data")
        fig = make_figure(title="Sample Difficulty Distribution")
        fig.add_trace(go.Histogram(
            x=self.per_sample_difficulty,
            nbinsx=n_bins,
            marker_color=QUALITATIVE[0], opacity=0.75,
        ))
        fig.update_layout(
            xaxis_title="Difficulty (fraction of models that failed)",
            yaxis_title="Count",
            height=350,
        )
        return fig

    def ensemble_candidates(self, min_complementarity: float = 0.05) -> list[tuple[str, str, float]]:
        """Return pairs of models with high complementarity (good ensemble candidates).

        Returns:
            List of (model_A, model_B, complementarity_score) sorted desc.
        """
        if self.complementarity.empty:
            return []
        pairs = []
        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                if i >= j:
                    continue
                c = float(self.complementarity.loc[m1, m2])
                if c >= min_complementarity:
                    pairs.append((m1, m2, round(c, 4)))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def summary(self) -> str:
        n_hard = len(self.hard_indices)
        n_total = (
            len(self.per_sample_difficulty) if self.per_sample_difficulty is not None else 0
        )
        pairs = self.ensemble_candidates()
        return (
            f"ErrorAnalysis: {len(self.models)} models, "
            f"{n_hard}/{n_total} hard samples, "
            f"{len(pairs)} ensemble candidate pairs."
        )

    def _repr_html_(self) -> str:
        rows = ""
        if not self.disagreement.empty:
            rows += f"<h4>Disagreement Matrix</h4>{self.disagreement.to_html()}"
        candidates = self.ensemble_candidates()
        if candidates:
            cand_rows = "".join(
                f"<tr><td>{a}</td><td>{b}</td><td>{c:.4f}</td></tr>"
                for a, b, c in candidates[:5]
            )
            rows += (
                "<h4>Top Ensemble Candidates</h4>"
                "<table border='1'><tr><th>Model A</th><th>Model B</th><th>Complementarity</th></tr>"
                f"{cand_rows}</table>"
            )
        return f"<h3>Error Analysis</h3><p>{self.summary()}</p>{rows}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile_hard_vs_easy(
    X: pd.DataFrame,
    hard_indices: np.ndarray,
    n_total: int,
) -> pd.DataFrame:
    """Compare mean feature values of hard vs easy samples."""
    easy_indices = np.setdiff1d(np.arange(n_total), hard_indices)
    if len(easy_indices) == 0 or len(hard_indices) == 0:
        return pd.DataFrame()

    rows = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        hard_mean = float(X.iloc[hard_indices][col].mean())
        easy_mean = float(X.iloc[easy_indices][col].mean())
        diff = hard_mean - easy_mean
        rows.append({"feature": col, "hard_mean": round(hard_mean, 4),
                     "easy_mean": round(easy_mean, 4), "difference": round(diff, 4)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["abs_diff"] = df["difference"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"])
    return df.reset_index(drop=True)
