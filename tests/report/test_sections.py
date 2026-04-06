"""Tests for report section builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dissectml.report.builder import AnalysisReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_report():
    """Report with no stage results — only required fields."""
    return AnalysisReport(task="classification", target="y", n_samples=100, n_features=5)


@pytest.fixture
def full_report():
    """Report with real EDA, Intelligence, Battle, and Compare results."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(1, 1, n),
        "x3": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })

    from dissectml.eda.result import EDAResult
    eda = EDAResult(df, target="target", task="classification")

    from dissectml.intelligence import analyze_intelligence
    intel = analyze_intelligence(df, target="target", task="classification", eda_result=eda)

    from dissectml.battle import battle
    battle_result = battle(df, target="target", task="classification", models=["LogisticRegression"], cv=3, n_jobs=1)

    X = df.drop(columns=["target"])
    y = df["target"]
    from dissectml.compare.comparator import ModelComparator
    compare = ModelComparator(battle_result, X=X, y=y)

    return AnalysisReport(
        task="classification", target="target",
        n_samples=n, n_features=3,
        eda=eda, intelligence=intel, models=battle_result, compare=compare,
    )


# ---------------------------------------------------------------------------
# summary_section
# ---------------------------------------------------------------------------

class TestSummarySection:
    def test_minimal_report_returns_dict(self, minimal_report):
        from dissectml.report.sections.summary_section import build_summary_section
        result = build_summary_section(minimal_report)
        assert isinstance(result, dict)

    def test_has_required_keys(self, minimal_report):
        from dissectml.report.sections.summary_section import build_summary_section
        result = build_summary_section(minimal_report)
        assert "title" in result
        assert "anchor" in result
        assert "content" in result

    def test_title_is_executive_summary(self, minimal_report):
        from dissectml.report.sections.summary_section import build_summary_section
        result = build_summary_section(minimal_report)
        assert result["title"] == "Executive Summary"

    def test_anchor_is_summary(self, minimal_report):
        from dissectml.report.sections.summary_section import build_summary_section
        result = build_summary_section(minimal_report)
        assert result["anchor"] == "summary"

    def test_content_is_html_string(self, minimal_report):
        from dissectml.report.sections.summary_section import build_summary_section
        result = build_summary_section(minimal_report)
        assert isinstance(result["content"], str)
        assert "<p" in result["content"]

    def test_full_report_includes_recommendations(self, full_report):
        from dissectml.report.sections.summary_section import build_summary_section
        result = build_summary_section(full_report)
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0


# ---------------------------------------------------------------------------
# eda_section
# ---------------------------------------------------------------------------

class TestEdaSection:
    def test_returns_none_when_no_eda(self, minimal_report):
        from dissectml.report.sections.eda_section import build_eda_section
        assert build_eda_section(minimal_report) is None

    def test_returns_dict_with_eda(self, full_report):
        from dissectml.report.sections.eda_section import build_eda_section
        result = build_eda_section(full_report)
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_report):
        from dissectml.report.sections.eda_section import build_eda_section
        result = build_eda_section(full_report)
        assert result["title"] == "EDA Findings"
        assert result["anchor"] == "eda"
        assert "content" in result

    def test_content_contains_html(self, full_report):
        from dissectml.report.sections.eda_section import build_eda_section
        result = build_eda_section(full_report)
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0


# ---------------------------------------------------------------------------
# intelligence_section
# ---------------------------------------------------------------------------

class TestIntelligenceSection:
    def test_returns_none_when_no_intel(self, minimal_report):
        from dissectml.report.sections.intelligence_section import build_intelligence_section
        assert build_intelligence_section(minimal_report) is None

    def test_returns_dict_with_intel(self, full_report):
        from dissectml.report.sections.intelligence_section import build_intelligence_section
        result = build_intelligence_section(full_report)
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_report):
        from dissectml.report.sections.intelligence_section import build_intelligence_section
        result = build_intelligence_section(full_report)
        assert result["title"] == "Pre-Model Intelligence"
        assert result["anchor"] == "intelligence"
        assert "content" in result


# ---------------------------------------------------------------------------
# battle_section
# ---------------------------------------------------------------------------

class TestBattleSection:
    def test_returns_none_when_no_models(self, minimal_report):
        from dissectml.report.sections.battle_section import build_battle_section
        assert build_battle_section(minimal_report) is None

    def test_returns_dict_with_models(self, full_report):
        from dissectml.report.sections.battle_section import build_battle_section
        result = build_battle_section(full_report)
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_report):
        from dissectml.report.sections.battle_section import build_battle_section
        result = build_battle_section(full_report)
        assert result["title"] == "Model Battle"
        assert result["anchor"] == "battle"
        assert "content" in result

    def test_content_has_leaderboard(self, full_report):
        from dissectml.report.sections.battle_section import build_battle_section
        result = build_battle_section(full_report)
        assert "table" in result["content"].lower()


# ---------------------------------------------------------------------------
# compare_section
# ---------------------------------------------------------------------------

class TestCompareSection:
    def test_returns_none_when_no_compare(self, minimal_report):
        from dissectml.report.sections.compare_section import build_compare_section
        assert build_compare_section(minimal_report) is None

    def test_returns_dict_with_compare(self, full_report):
        from dissectml.report.sections.compare_section import build_compare_section
        result = build_compare_section(full_report)
        assert isinstance(result, dict)

    def test_has_required_keys(self, full_report):
        from dissectml.report.sections.compare_section import build_compare_section
        result = build_compare_section(full_report)
        assert result["title"] == "Comparative Analysis"
        assert result["anchor"] == "compare"
        assert "content" in result
