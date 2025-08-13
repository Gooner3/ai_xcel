from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def merge_dataframes(main: pd.DataFrame, other: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    """Left-merge ``other`` into ``main`` using the given key columns."""
    return main.merge(other, left_on=left_on, right_on=right_on, how="left")


def lookup_dataframe(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Return rows from ``df`` where ``column`` equals ``value`` (as string)."""
    return df[df[column].astype(str) == str(value)]
