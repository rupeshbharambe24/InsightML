"""Tests for ClusterDiscovery."""
import pandas as pd

from insightml.eda.clusters import ClusterDiscovery


def test_kmeans_finds_clusters(df_classification):
    cd = ClusterDiscovery(df_classification)
    result = cd.kmeans()
    assert "optimal_k" in result
    assert result["optimal_k"] >= 2


def test_profiles_dataframe(df_classification):
    cd = ClusterDiscovery(df_classification)
    profiles = cd.profiles()
    assert isinstance(profiles, pd.DataFrame)
    if not profiles.empty:
        assert "cluster" in profiles.columns
        assert "size" in profiles.columns


def test_scatter_fig_built(df_classification):
    import plotly.graph_objects as go
    cd = ClusterDiscovery(df_classification)
    fig = cd.scatter_2d()
    assert isinstance(fig, go.Figure)


def test_elbow_plot_built(df_classification):
    import plotly.graph_objects as go
    cd = ClusterDiscovery(df_classification)
    fig = cd.elbow_plot()
    assert isinstance(fig, go.Figure)


def test_insufficient_columns():
    df = pd.DataFrame({"only_one": [1, 2, 3, 4, 5]})
    cd = ClusterDiscovery(df)
    cd._ensure_computed()
    assert cd._results.get("skipped") is True


def test_dbscan_runs(df_regression):
    cd = ClusterDiscovery(df_regression)
    result = cd.dbscan()
    assert "n_clusters" in result or "error" in result
