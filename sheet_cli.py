#!/usr/bin/env python3
"""Simple CLI tool to load a main sheet and reference/merge additional sheets."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI for viewing and merging multiple spreadsheets"
    )
    parser.add_argument(
        "main_sheet", help="Path to main spreadsheet (csv/xls/xlsx)"
    )
    args = parser.parse_args()

    main_path = Path(args.main_sheet)
    main_df = load_dataframe(main_path)
    print(
        f"Loaded main sheet: {main_path} ({len(main_df)} rows × {len(main_df.columns)} cols)"
    )

    others: dict[str, pd.DataFrame] = {}

    while True:
        try:
            cmd = input(
                "command> "
            ).strip()
        except EOFError:
            break
        if not cmd:
            continue
        parts = cmd.split()
        action = parts[0].lower()

        if action == "load" and len(parts) == 3:
            name, path_str = parts[1], parts[2]
            try:
                df = load_dataframe(Path(path_str))
                others[name] = df
                print(
                    f"Loaded '{name}' ({len(df)} rows × {len(df.columns)} cols)"
                )
            except Exception as e:
                print(f"Load error: {e}")

        elif action == "merge" and len(parts) >= 3:
            name = parts[1]
            df = others.get(name)
            if df is None:
                print(f"No sheet named '{name}'.")
                continue
            if len(parts) == 3:
                left_key = right_key = parts[2]
            else:
                left_key, right_key = parts[2], parts[3]
            if left_key not in main_df.columns or right_key not in df.columns:
                print("Column missing in one of the sheets.")
                continue
            main_df = main_df.merge(df, left_on=left_key, right_on=right_key, how="left")
            print(
                f"Merged '{name}' into main sheet on '{left_key}' -> '{right_key}'."
            )

        elif action == "lookup" and len(parts) >= 4:
            name, column, value = parts[1], parts[2], " ".join(parts[3:])
            df = others.get(name)
            if df is None:
                print(f"No sheet named '{name}'.")
                continue
            if column not in df.columns:
                print(f"Column '{column}' not found in '{name}'.")
                continue
            matches = df[df[column].astype(str) == value]
            if matches.empty:
                print("No matches found.")
            else:
                print(matches)

        elif action == "show":
            target = parts[1] if len(parts) > 1 else "main"
            df = main_df if target == "main" else others.get(target)
            if df is None:
                print(f"No sheet named '{target}'.")
            else:
                print(df.head())

        elif action == "save":
            out_path = Path(parts[1]) if len(parts) > 1 else main_path
            try:
                if out_path.suffix.lower() == ".csv":
                    main_df.to_csv(out_path, index=False)
                else:
                    main_df.to_excel(out_path, index=False)
                print(f"Saved main sheet to {out_path}")
            except Exception as e:
                print(f"Save error: {e}")

        elif action in {"exit", "quit", "q"}:
            break

        else:
            print(
                "Commands: load <name> <path>, merge <name> <main_col> [other_col], lookup <name> <column> <value>, show [main|name], save [path], exit"
            )


if __name__ == "__main__":
    main()
