"""ClusterDiscovery — auto K-Means + DBSCAN with profiling and PCA/t-SNE viz."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from insightml._types import ColumnType
from insightml.core.validators import infer_column_type
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import QUALITATIVE, make_figure


class ClusterDiscovery(BaseAnalysisModule):
    """Unsupervised cluster discovery on numeric features.

    **Auto K-Means**:
    - StandardScaler → KMeans for K=2..max_k
    - Select K with highest silhouette score
    - Elbow plot and silhouette plot

    **Auto DBSCAN**:
    - Estimate eps from k-distance curve knee
    - min_samples = max(2*n_features, 5)

    **Profiles**: per-cluster mean/mode/size statistics.

    Access::

        eda.clusters.kmeans()          # auto K-Means result
        eda.clusters.dbscan()          # auto DBSCAN result
        eda.clusters.profiles()        # cluster profile DataFrame
        eda.clusters.scatter_2d()      # PCA 2D scatter plot
        eda.clusters.elbow_plot()      # inertia vs K
        eda.clusters.silhouette_plot() # silhouette vs K
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config

        num_cols = [
            c for c in df.columns
            if infer_column_type(df[c], config) == ColumnType.NUMERIC
            and c != self._target
        ]
        if len(num_cols) < 2:
            self._warn("Need at least 2 numeric features for cluster analysis.")
            self._results = {"num_cols": num_cols, "skipped": True}
            return

        X_raw = df[num_cols].dropna()
        if len(X_raw) < 10:
            self._warn("Not enough samples for cluster analysis.")
            self._results = {"num_cols": num_cols, "skipped": True}
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Auto K-Means
        max_k = min(config.max_k_clusters, len(X_raw) // 2)
        kmeans_result = _auto_kmeans(X_scaled, X_raw, max_k)

        # PCA for visualization
        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.tolist()

        # DBSCAN
        dbscan_result = _auto_dbscan(X_scaled, X_raw)

        # Cluster profiles from K-Means labels
        profiles_df = pd.DataFrame()
        if kmeans_result.get("labels") is not None:
            profiles_df = _cluster_profiles(
                X_raw.copy(), kmeans_result["labels"], df, num_cols, self._target
            )

        self._results = {
            "num_cols": num_cols,
            "skipped": False,
            "kmeans": kmeans_result,
            "dbscan": dbscan_result,
            "X_pca": X_pca,
            "pca_explained_var": explained_var,
            "X_raw_index": list(X_raw.index),
            "profiles": profiles_df,
        }

    def _build_figures(self) -> dict[str, go.Figure]:
        if self._results.get("skipped"):
            return {}

        figs: dict[str, go.Figure] = {}
        km = self._results["kmeans"]

        # Elbow plot
        if km.get("inertias"):
            figs["elbow"] = _elbow_fig(km["k_range"], km["inertias"])

        # Silhouette plot
        if km.get("silhouette_scores"):
            figs["silhouette"] = _silhouette_fig(
                km["k_range"], km["silhouette_scores"], km.get("optimal_k")
            )

        # PCA scatter
        X_pca = self._results["X_pca"]
        labels = km.get("labels")
        if X_pca is not None and labels is not None:
            figs["scatter_2d"] = _scatter_fig(
                X_pca, labels,
                self._results["pca_explained_var"],
            )

        return figs

    def summary(self) -> str:
        if self._results.get("skipped"):
            return "Cluster analysis skipped (insufficient numeric columns or samples)."
        km = self._results["kmeans"]
        db = self._results["dbscan"]
        k = km.get("optimal_k", "?")
        sil = km.get("best_silhouette", "?")
        n_noise = db.get("n_noise", "?")
        return (
            f"Auto K-Means: optimal K={k} (silhouette={sil:.3f}). "
            f"DBSCAN: {db.get('n_clusters', '?')} clusters, {n_noise} noise points."
            if isinstance(sil, float) else
            f"Auto K-Means: optimal K={k}."
        )

    # --- Public accessors ---

    def kmeans(self, k: int | None = None) -> dict[str, Any]:
        self._ensure_computed()
        if k is not None:
            return _run_kmeans_k(
                self._results.get("X_pca"), k  # type: ignore
            )
        return self._results.get("kmeans", {})

    def dbscan(self, eps: float | None = None,
               min_samples: int | None = None) -> dict[str, Any]:
        self._ensure_computed()
        return self._results.get("dbscan", {})

    def profiles(self) -> pd.DataFrame:
        self._ensure_computed()
        return self._results.get("profiles", pd.DataFrame())

    def elbow_plot(self) -> go.Figure:
        self._ensure_computed()
        return self._figures.get("elbow", make_figure("No elbow data"))

    def silhouette_plot(self) -> go.Figure:
        self._ensure_computed()
        return self._figures.get("silhouette", make_figure("No silhouette data"))

    def scatter_2d(self, method: str = "pca") -> go.Figure:
        self._ensure_computed()
        return self._figures.get("scatter_2d", make_figure("No scatter data"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_kmeans(
    X_scaled: np.ndarray, X_raw: pd.DataFrame, max_k: int
) -> dict[str, Any]:
    k_range = list(range(2, max(3, max_k + 1)))
    inertias, sil_scores = [], []

    for k in k_range:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            if len(set(labels)) > 1:
                sil_scores.append(float(silhouette_score(X_scaled, labels,
                                                          sample_size=min(2000, len(X_scaled)))))
            else:
                sil_scores.append(-1.0)
        except Exception:
            inertias.append(np.nan)
            sil_scores.append(-1.0)

    if not sil_scores or all(s == -1.0 for s in sil_scores):
        return {"k_range": k_range, "inertias": inertias, "silhouette_scores": sil_scores}

    best_idx = int(np.argmax(sil_scores))
    optimal_k = k_range[best_idx]

    km_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    labels = km_final.fit_predict(X_scaled)

    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "optimal_k": optimal_k,
        "best_silhouette": round(sil_scores[best_idx], 4),
        "labels": labels,
    }


def _auto_dbscan(X_scaled: np.ndarray, X_raw: pd.DataFrame) -> dict[str, Any]:
    try:
        from sklearn.neighbors import NearestNeighbors
        n_features = X_scaled.shape[1]
        min_samples = max(2 * n_features, 5)
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])
        # Find knee: max of second derivative
        d2 = np.diff(k_distances, n=2)
        knee_idx = int(np.argmax(d2)) + 2
        eps = float(k_distances[knee_idx])
        eps = max(eps, 0.1)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_noise = int((labels == -1).sum())

        return {
            "eps": round(eps, 4),
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels": labels,
        }
    except Exception as e:
        return {"error": str(e)}


def _cluster_profiles(
    X_raw: pd.DataFrame, labels: np.ndarray,
    df_full: pd.DataFrame, num_cols: list[str], target: str | None
) -> pd.DataFrame:
    X_raw = X_raw.copy()
    X_raw["_cluster"] = labels
    if target and target in df_full.columns:
        # Align index
        target_aligned = df_full[target].loc[X_raw.index]
        X_raw["_target"] = target_aligned.values

    rows = []
    for cluster_id in sorted(set(labels)):
        group = X_raw[X_raw["_cluster"] == cluster_id]
        row: dict[str, Any] = {
            "cluster": int(cluster_id),
            "size": len(group),
            "pct": round(100 * len(group) / len(X_raw), 1),
        }
        for col in num_cols:
            if col in group.columns:
                row[f"{col}_mean"] = round(float(group[col].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _elbow_fig(k_range: list, inertias: list) -> go.Figure:
    fig = make_figure(title="K-Means Elbow Plot")
    fig.add_trace(go.Scatter(
        x=k_range, y=inertias, mode="lines+markers",
        marker_color=QUALITATIVE[0], name="Inertia",
    ))
    fig.update_layout(xaxis_title="K", yaxis_title="Inertia", height=350)
    return fig


def _silhouette_fig(k_range: list, scores: list, optimal_k: int | None) -> go.Figure:
    fig = make_figure(title="Silhouette Score vs K")
    colors = [
        "#e45756" if k == optimal_k else QUALITATIVE[0]
        for k in k_range
    ]
    fig.add_trace(go.Bar(
        x=k_range, y=scores, marker_color=colors,
        name="Silhouette",
    ))
    fig.update_layout(xaxis_title="K", yaxis_title="Silhouette Score", height=350)
    return fig


def _scatter_fig(X_pca: np.ndarray, labels: np.ndarray,
                 explained_var: list) -> go.Figure:
    unique_labels = sorted(set(labels))
    xlab = f"PC1 ({explained_var[0]*100:.1f}%)" if explained_var else "PC1"
    ylab = f"PC2 ({explained_var[1]*100:.1f}%)" if len(explained_var) > 1 else "PC2"
    fig = make_figure(title="Cluster Scatter (PCA)")
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = f"Cluster {label}" if label != -1 else "Noise"
        color = "#999999" if label == -1 else QUALITATIVE[i % len(QUALITATIVE)]
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0].tolist(),
            y=X_pca[mask, 1].tolist() if X_pca.shape[1] > 1 else [0.0] * mask.sum(),
            mode="markers",
            name=name,
            marker={"color": color, "size": 5, "opacity": 0.6},
        ))
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, height=450)
    return fig


def _run_kmeans_k(X: np.ndarray | None, k: int) -> dict[str, Any]:
    if X is None:
        return {}
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else -1.0
        return {"k": k, "inertia": float(km.inertia_), "silhouette": round(sil, 4),
                "labels": labels}
    except Exception as e:
        return {"error": str(e)}
