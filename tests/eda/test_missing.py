"""Tests for MissingDataIntelligence."""
import pandas as pd

from dissectml.eda.missing import MissingDataIntelligence


def test_no_missing():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    m = MissingDataIntelligence(df)
    m._ensure_computed()
    assert m._results["total_missing"] == 0
    assert m._results["cols_with_missing"] == []


def test_counts_correct(df_with_missing):
    m = MissingDataIntelligence(df_with_missing)
    counts = m.counts()
    num1_row = counts[counts["column"] == "num1"].iloc[0]
    assert num1_row["missing_count"] == 15


def test_littles_test_returns_dict(df_with_missing):
    m = MissingDataIntelligence(df_with_missing)
    result = m.littles_test()
    assert "mechanism" in result


def test_recommendations_returned(df_with_missing):
    m = MissingDataIntelligence(df_with_missing)
    recs = m.recommendations()
    assert isinstance(recs, dict)
    for _col, rec_list in recs.items():
        assert isinstance(rec_list, list)
        assert len(rec_list) > 0


def test_classify_returns_missingness_types(df_with_missing):
    m = MissingDataIntelligence(df_with_missing)
    classification = m.classify()
    from dissectml._types import MissingnessType
    for _col, mtype in classification.items():
        assert isinstance(mtype, MissingnessType)


def test_figures_built(df_with_missing):
    m = MissingDataIntelligence(df_with_missing)
    figs = m.plot()
    assert "patterns" in figs
    assert "missing_bar" in figs
