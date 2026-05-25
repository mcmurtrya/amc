"""Tests for the CFTC COT parser (pure-function side; no network)."""

from __future__ import annotations

import pandas as pd

from metals.data.cot import TUE_TO_FRI_OFFSET, parse_cot_dataframe


def _synthetic_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-style CFTC frame from a list of row dicts.

    Fills any column we don't override with a zero default so the parser's
    downstream rename + arithmetic doesn't trip over missing columns.
    """
    cols = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "Prod_Merc_Positions_Long_All",
        "Prod_Merc_Positions_Short_All",
        "Swap_Positions_Long_All",
        "Swap__Positions_Short_All",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
        "Other_Rept_Positions_Long_All",
        "Other_Rept_Positions_Short_All",
        "NonRept_Positions_Long_All",
        "NonRept_Positions_Short_All",
        "Open_Interest_All",
    ]
    full = []
    for r in rows:
        full.append({c: r.get(c, 0) for c in cols})
    return pd.DataFrame(full, columns=cols)


def test_friday_shift():
    """Tuesday positioning -> Friday release. Verify the offset is 3 days."""
    raw = _synthetic_raw([
        {
            "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
            "Report_Date_as_YYYY-MM-DD": "2024-05-21",  # a Tuesday
            "Open_Interest_All": 100,
        },
    ])
    out = parse_cot_dataframe(raw)
    assert len(out) == 1
    assert out.loc[0, "timestamp_utc"] == pd.Timestamp("2024-05-24")
    # Sanity: 2024-05-21 is a Tuesday; +3 days = 2024-05-24, a Friday.
    assert out.loc[0, "timestamp_utc"].weekday() == 4
    assert TUE_TO_FRI_OFFSET.days == 3


def test_excludes_emini_and_micro_variants():
    """Substring 'GOLD - COMMODITY EXCHANGE INC.' must not match
    'E-MINI GOLD' or 'MICRO GOLD' (which have it as a suffix)."""
    raw = _synthetic_raw([
        {
            "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
            "Report_Date_as_YYYY-MM-DD": "2024-01-02",
            "Open_Interest_All": 500_000,
        },
        {
            "Market_and_Exchange_Names": "E-MINI GOLD - COMMODITY EXCHANGE INC.",
            "Report_Date_as_YYYY-MM-DD": "2024-01-02",
            "Open_Interest_All": 30_000,
        },
        {
            "Market_and_Exchange_Names": "MICRO GOLD - COMMODITY EXCHANGE INC.",
            "Report_Date_as_YYYY-MM-DD": "2024-01-02",
            "Open_Interest_All": 5_000,
        },
    ])
    out = parse_cot_dataframe(raw)
    assert len(out) == 1
    assert out.loc[0, "open_interest"] == 500_000
    assert out.loc[0, "metal"] == "gold"


def test_commercial_combines_producer_and_swap():
    """commercial_long = Producer_Long + Swap_Long; same for short."""
    raw = _synthetic_raw([
        {
            "Market_and_Exchange_Names": "SILVER - COMMODITY EXCHANGE INC.",
            "Report_Date_as_YYYY-MM-DD": "2024-06-04",
            "Prod_Merc_Positions_Long_All": 10_000,
            "Prod_Merc_Positions_Short_All": 30_000,
            "Swap_Positions_Long_All": 5_000,
            "Swap__Positions_Short_All": 7_000,
        },
    ])
    out = parse_cot_dataframe(raw)
    assert out.loc[0, "commercial_long"] == 15_000
    assert out.loc[0, "commercial_short"] == 37_000


def test_picks_up_all_four_metals():
    raw = _synthetic_raw([
        {"Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
         "Report_Date_as_YYYY-MM-DD": "2024-01-02"},
        {"Market_and_Exchange_Names": "SILVER - COMMODITY EXCHANGE INC.",
         "Report_Date_as_YYYY-MM-DD": "2024-01-02"},
        {"Market_and_Exchange_Names": "PLATINUM - NEW YORK MERCANTILE EXCHANGE",
         "Report_Date_as_YYYY-MM-DD": "2024-01-02"},
        {"Market_and_Exchange_Names": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE",
         "Report_Date_as_YYYY-MM-DD": "2024-01-02"},
        {"Market_and_Exchange_Names": "WHEAT - CHICAGO BOARD OF TRADE",
         "Report_Date_as_YYYY-MM-DD": "2024-01-02"},
    ])
    out = parse_cot_dataframe(raw)
    assert set(out["metal"]) == {"gold", "silver", "platinum", "palladium"}
    assert len(out) == 4


def test_missing_swap_columns_treated_as_zero():
    """Older COT vintages sometimes omit the Swap split — must not crash."""
    cols_without_swap = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "Prod_Merc_Positions_Long_All",
        "Prod_Merc_Positions_Short_All",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
        "Other_Rept_Positions_Long_All",
        "Other_Rept_Positions_Short_All",
        "NonRept_Positions_Long_All",
        "NonRept_Positions_Short_All",
        "Open_Interest_All",
    ]
    raw = pd.DataFrame([{
        "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
        "Report_Date_as_YYYY-MM-DD": "2024-01-02",
        "Prod_Merc_Positions_Long_All": 100,
        "Prod_Merc_Positions_Short_All": 50,
    }], columns=cols_without_swap).fillna(0)
    out = parse_cot_dataframe(raw)
    assert out.loc[0, "commercial_long"] == 100
    assert out.loc[0, "commercial_short"] == 50
