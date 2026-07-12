"""Unit tests for the phase1_diagnose configuration logic.

The end-to-end pipeline requires DuckDB-resident data and isn't unit-tested
here; the diagnostic script writes through the existing metals.models.lgbm_vol
path which is already covered by tests/test_lgbm_vol.py.
"""

from __future__ import annotations

from scripts.phase1_diagnose import SUBSETS, configurations, feature_subset

SAMPLE_COLUMNS = [
    # returns + vol family
    "GC=F_ret_1d",
    "GC=F_ret_5d",
    "SI=F_ret_20d",
    "GC=F_rvol_20d",
    "SI=F_skew_20d",
    "PL=F_kurt_20d",
    "PA=F_maxdd_60d",
    # spreads family
    "Au_Ag_ratio",
    "Au_Ag_logchg_5d",
    "Pt_Pd_logchg_1d",
    "Au_Cu_z_252d",
    "Au_Oil_logchg_20d",
    # macro family
    "real_yield_10y",
    "real_yield_chg_5d",
    "breakeven_5y_chg_5d",
    "dxy_chg_5d",
    "dxy_pctile_252d",
    "vix_chg_5d",
    "vix_pctile_252d",
    "yield_curve_slope",
    "yield_curve_slope_chg_5d",
    "baa_spread_chg_5d",
    "gpr_chg_5d",
    "gpr_pctile_252d",
]


def test_feature_subset_matches_substrings():
    out = feature_subset(SAMPLE_COLUMNS, ("Au_Ag",))
    assert "Au_Ag_ratio" in out
    assert "Au_Ag_logchg_5d" in out
    assert "GC=F_ret_1d" not in out


def test_subsets_partition_cover_all_known_columns():
    """Every sample column should belong to at least one SUBSETS group."""
    all_substrs = sum(SUBSETS.values(), ())
    for c in SAMPLE_COLUMNS:
        assert any(sub in c for sub in all_substrs), c


def test_configurations_include_all_and_paired_variants():
    cfg = configurations(SAMPLE_COLUMNS)
    assert "all" in cfg
    # Each known group should produce both an ablation and a marginal config
    for grp in SUBSETS:
        assert f"no_{grp}" in cfg
        assert f"only_{grp}" in cfg


def test_no_and_only_are_complementary():
    cfg = configurations(SAMPLE_COLUMNS)
    for grp in SUBSETS:
        union = set(cfg[f"no_{grp}"]) | set(cfg[f"only_{grp}"])
        assert union == set(SAMPLE_COLUMNS), grp


def test_only_subset_is_nonempty_for_each_group():
    cfg = configurations(SAMPLE_COLUMNS)
    for grp in SUBSETS:
        assert cfg[f"only_{grp}"], f"empty only_{grp}"
