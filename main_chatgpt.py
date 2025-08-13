#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

from sheet_utils import load_dataframe, merge_dataframes, lookup_dataframe


# =========================
#   JSON / InfoTree utils
# =========================

class NumpyEncoder(json.JSONEncoder):
    """Safe JSON encoder for numpy & pandas types."""
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if pd.isna(obj):
                return None
            if hasattr(obj, "item"):
                return obj.item()
        except Exception:
            pass
        return super().default(obj)


def safe_convert_scalar(obj):
    """Convert numpy/pandas scalars to Python builtins."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if pd.isna(obj):
        return None
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


# =========================
#     Agent Definition
# =========================

class SpreadsheetAgent:
    def __init__(
        self,
        sheet_path: Optional[str],
        llama_endpoint: str = "http://100.114.241.89/v1/chat/completions",
        model: str = "default",
        autosave: bool = True,
    ):
        self.prompts_dir = Path(".prompts")
        self.history_dir = Path(".history")
        self.prompts_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
        self.api_log_path = Path("api_prompts.log")

        self.llama_endpoint = llama_endpoint
        self.model = model
        self.autosave = autosave

        self.sheet_path = sheet_path
        self.df: Optional[pd.DataFrame] = None
        self.extra_dfs: Dict[str, pd.DataFrame] = {}
        self.changes_history: List[Dict[str, Any]] = []   # in-memory stack
        self._last_generated_blocks: List[str] = []
        self._last_request: str = ""

        self._init_minimal_prompts()

        if sheet_path:
            self.load_data(sheet_path)

    # -------------------------
    #       File I/O
    # -------------------------

    def load_data(self, file_path: str) -> bool:
        try:
            self.df = load_dataframe(file_path)
            print(f"✓ Loaded {file_path} ({len(self.df)} rows × {len(self.df.columns)} cols)")
            return True
        except Exception as e:
            print(f"[✗] Load error: {e}")
            return False


    def save_data(self, file_path: Optional[str] = None) -> bool:
        if self.df is None:
            print("[!] No data to save.")
            return False
        try:
            out = file_path or self.sheet_path or f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            if out.lower().endswith(".csv"):
                self.df.to_csv(out, index=False)
            elif out.lower().endswith((".xlsx", ".xls")):
                self.df.to_excel(out, index=False)
            else:
                # Default to CSV
                out = f"{out}.csv"
                self.df.to_csv(out, index=False)
            print(f"✓ Saved: {out}")
            return True
        except Exception as e:
            print(f"[✗] Save error: {e}")
            return False

    # -------------------------
    #     Prompts & LLM
    # -------------------------

    def _init_minimal_prompts(self):
        # Keep these extremely minimal; you’ll edit them later.
        defaults = {
            "system.txt": "Return ONLY executable pandas code that transforms a DataFrame named df.",
            "analyze.txt": (
                "DATA:\n{info_tree}\n\n"
                "USER:\n{user_request}\n\n"
                "Write pandas code to modify df. No prose. No backticks. No prints."
            ),
            # Optional user instructions prepended to every request
            "instructions.txt": "",
        }
        for name, content in defaults.items():
            p = self.prompts_dir / name
            if not p.exists():
                p.write_text(content, encoding="utf-8")

    def _read_prompt(self, name: str) -> str:
        p = self.prompts_dir / name
        if not p.exists():
            p.write_text("", encoding="utf-8")
        return p.read_text(encoding="utf-8").strip()

    def _log_api_prompt(self, payload: Dict[str, Any]) -> None:
        try:
            entry = {"timestamp": datetime.now().isoformat(), "payload": payload}
            with self.api_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
        except Exception as e:
            print(f"[!] Log error: {e}")

    def query_llama(self, user_request: str) -> str:
        """Send info tree + request to llama.cpp and return raw text."""
        system_prompt = self._read_prompt("system.txt")
        analyze_t = self._read_prompt("analyze.txt")

        info_tree = self.create_info_tree()
        info_json = json.dumps(info_tree, indent=2, cls=NumpyEncoder)

        prompt = analyze_t.format(info_tree=info_json, user_request=user_request)

        # Prepend any user-specified instructions from .prompts/instructions.txt
        extra_instructions = self._read_prompt("instructions.txt")
        if extra_instructions:
            prompt = f"{extra_instructions}\n\n{prompt}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1500,
        }

        self._log_api_prompt(payload)

        try:
            r = requests.post(self.llama_endpoint, json=payload, timeout=360)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[✗] LLaMA error: {e}")
            return ""

    # -------------------------
    #     Info Tree
    # -------------------------

    def create_info_tree(self) -> Dict[str, Any]:
        """Comprehensive, JSON-safe info structure."""
        if self.df is None:
            return {}
        df = self.df

        # Convert memory_usage & null_counts to safe types
        memory_dict = {k: safe_convert_scalar(v) for k, v in df.memory_usage(deep=True).items()}
        null_dict = {k: safe_convert_scalar(v) for k, v in df.isnull().sum().items()}

        info: Dict[str, Any] = {
            "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
            "columns": {},
            "sample_data": {},
            "memory_usage": memory_dict,
            "null_counts": null_dict,
        }

        for col in df.columns:
            s = df[col]
            col_info = {
                "dtype": str(s.dtype),
                "null_count": safe_convert_scalar(s.isnull().sum()),
                "unique_count": safe_convert_scalar(s.nunique(dropna=True)),
                "memory_bytes": safe_convert_scalar(s.memory_usage(deep=True)),
            }

            if pd.api.types.is_numeric_dtype(s):
                col_info.update({
                    "min": safe_convert_scalar(s.min() if len(s) else None),
                    "max": safe_convert_scalar(s.max() if len(s) else None),
                    "mean": safe_convert_scalar(s.mean() if len(s) else None),
                })
            elif pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    max_len = safe_convert_scalar(s.astype(str).str.len().max())
                except Exception:
                    max_len = None
                common_vals = {}
                try:
                    common_vals = {str(k): safe_convert_scalar(v) for k, v in s.value_counts(dropna=True).head(3).items()}
                except Exception:
                    pass
                col_info.update({
                    "max_length": max_len,
                    "common_values": common_vals,
                })

            info["columns"][str(col)] = col_info

            # first non-null sample
            try:
                sample_val = s.dropna().iloc[0] if s.dropna().size else None
            except Exception:
                sample_val = None
            info["sample_data"][str(col)] = safe_convert_scalar(sample_val)

        return info

    # -------------------------
    #   Code extraction/exec
    # -------------------------

    @staticmethod
    def _extract_code_blocks(raw: str) -> List[str]:
        """
        Accepts AI output and returns a list of code blocks.
        - Strips ``` fences if present.
        - Splits multiple blocks (by code fences or blank-line separators).
        """
        if not raw:
            return []

        # If fenced blocks present, extract them
        fenced = re.findall(r"```(?:python)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        blocks = [b.strip() for b in fenced if b.strip()]

        if not blocks:
            # No fences, split by blank lines where it looks like separate statements
            parts = re.split(r"\n\s*\n", raw.strip())
            blocks = [p.strip() for p in parts if p.strip()]

        # Final cleanup: keep only lines that plausibly relate to df/pandas
        cleaned_blocks = []
        for b in blocks:
            lines = []
            for line in b.splitlines():
                # Keep imports of pandas/numpy, df operations, assignments, function defs
                if (
                    "df" in line
                    or line.strip().startswith(("import ", "from "))
                    or line.strip().startswith(("def ", "class "))
                    or line.strip().startswith(("pd.", "np."))
                ):
                    lines.append(line)
            cleaned = "\n".join(lines).strip() or b
            cleaned_blocks.append(cleaned)

        # Remove duplicates while preserving order
        seen = set()
        unique_blocks = []
        for b in cleaned_blocks:
            if b not in seen:
                unique_blocks.append(b)
                seen.add(b)
        return unique_blocks

    def _exec_on_copy(self, code: str, df: pd.DataFrame) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """Execute code against a COPY for preview. Returns (ok, new_df, error)."""
        try:
            local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
            for name, other in self.extra_dfs.items():
                local_vars[name] = other.copy()
            exec(code, {'__builtins__': {}}, local_vars)
            new_df = local_vars.get('df', None)
            if new_df is None or not isinstance(new_df, pd.DataFrame):
                # If they mutated in-place and didn't reassign, use the mutated df
                new_df = local_vars.get('df', df)
            return True, new_df, None
        except Exception as e:
            return False, None, str(e)

    def _snapshot_for_undo(self, code: str, user_request: str):
        """Push current DataFrame into history (in-memory and on-disk)."""
        if self.df is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = self.history_dir / f"{stamp}.pkl"
        meta_path = self.history_dir / f"{stamp}.json"
        try:
            self.df.to_pickle(pkl_path)
            meta = {"timestamp": stamp, "code": code, "user_request": user_request}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.changes_history.append({"pkl": pkl_path, "meta": meta, "code": code})
        except Exception as e:
            print(f"[!] Could not snapshot history: {e}")

    def undo(self) -> bool:
        if not self.changes_history:
            print("[i] Nothing to undo.")
            return False
        last = self.changes_history.pop()
        try:
            restored = pd.read_pickle(last["pkl"])
            self.df = restored
            print(f"✓ Undo successful (from {last['pkl'].name})")
            if self.autosave:
                self.save_data(self.sheet_path)
            return True
        except Exception as e:
            print(f"[✗] Undo failed: {e}")
            return False

    # -------------------------
    #   Condensed diffs/preview
    # -------------------------

    @staticmethod
    def _condensed_diff(before: pd.DataFrame, after: pd.DataFrame, max_show: int = 8) -> str:
        """Return a human-readable condensed summary of changes."""
        out = []
        r0, c0 = before.shape
        r1, c1 = after.shape

        # Shape changes
        if (r0, c0) != (r1, c1):
            out.append(f"- shape: {r0}×{c0} → {r1}×{c1}")

        # Column adds/removes
        before_cols = set(map(str, before.columns))
        after_cols = set(map(str, after.columns))
        added = sorted(after_cols - before_cols)
        removed = sorted(before_cols - after_cols)
        if added:
            out.append(f"- added columns: {', '.join(added[:10])}" + (" ..." if len(added) > 10 else ""))
        if removed:
            out.append(f"- removed columns: {', '.join(removed[:10])}" + (" ..." if len(removed) > 10 else ""))

        # Dtype changes & modified columns sample
        common = [c for c in after.columns if c in before.columns]
        dtype_changes = []
        changed_cols = []

        for c in common:
            if str(before[c].dtype) != str(after[c].dtype):
                dtype_changes.append(f"{c}: {before[c].dtype}→{after[c].dtype}")
            # quick equality check (cheap heuristic)
            try:
                if not before[c].equals(after[c]):
                    changed_cols.append(str(c))
            except Exception:
                changed_cols.append(str(c))

        if dtype_changes:
            out.append("- dtype changes: " + "; ".join(dtype_changes[:8]) + (" ..." if len(dtype_changes) > 8 else ""))

        if changed_cols:
            out.append("- changed columns (sample): " + ", ".join(changed_cols[:10]) + (" ..." if len(changed_cols) > 10 else ""))

        # Show a tiny head of changed columns (if reasonable)
        cols_to_show = list((added + changed_cols)[:max_show])
        if cols_to_show:
            try:
                tiny = after[cols_to_show].head(5)
                out.append("\nPreview (first 5 rows of changed/added cols):")
                out.append(tabulate(tiny, headers="keys", tablefmt="github", showindex=False))
            except Exception:
                pass

        if not out:
            out.append("- no structural differences detected (may be value tweaks only)")
            tiny = after.head(5)
            out.append("\nPreview (first 5 rows):")
            out.append(tabulate(tiny, headers="keys", tablefmt="github", showindex=False))

        return "\n".join(out)

    def _default_preview(self, rows: int = 8) -> None:
        if self.df is None:
            print("[!] No data loaded.")
            return
        print("\n=== Preview (first {} rows) ===".format(rows))
        print(tabulate(self.df.head(rows), headers="keys", tablefmt="github", showindex=False))
        print(f"\nShape: {self.df.shape}, Columns: {len(self.df.columns)}")

    # -------------------------
    #     Interactive Loop
    # -------------------------

    def interactive(self):
        if self.df is None:
            print("[!] Load a sheet first.")
            return

        # Always show condensed preview first (your request)
        self._default_preview(rows=8)

        print("\nCommands:")
        print("  ask <instruction>   # AI generates code; previews condensed diff by default")
        print("  apply               # apply last generated code block(s)")
        print("  undo                # undo last applied change")
        print("  load <name> <path>  # load another sheet as DataFrame")
        print("  merge <name> <main_col> [other_col]  # merge a loaded sheet into main")
        print("  lookup <name> <column> <value>  # search a loaded sheet")
        print("  show [name|n]       # show main (default) or loaded sheet, or first n rows")
        print("  info                # detailed info tree (JSON)")
        print("  save [path]         # save to path (or overwrite original)")
        print("  reload              # reload original file from disk")
        print("  help                # show commands")
        print("  quit/exit           # leave\n")

        while True:
            try:
                cmd = input("> ").strip()
                if not cmd:
                    continue

                if cmd in ("quit", "exit", "q"):
                    print("Bye!")
                    break

                if cmd.startswith("show"):
                    parts = cmd.split()
                    if len(parts) == 1:
                        self._default_preview(rows=8)
                    elif len(parts) == 2 and parts[1].isdigit():
                        n = int(parts[1])
                        self._default_preview(rows=n)
                    else:
                        name = parts[1]
                        df2 = self.extra_dfs.get(name)
                        if df2 is None:
                            print(f"[!] No sheet named '{name}'.")
                        else:
                            print(f"\n=== {name} preview (first 8 rows) ===")
                            print(tabulate(df2.head(8), headers='keys', tablefmt='github', showindex=False))
                            print(f"\nShape: {df2.shape}, Columns: {len(df2.columns)}")
                    continue

                if cmd == "info":
                    it = self.create_info_tree()
                    print(json.dumps(it, indent=2, cls=NumpyEncoder))
                    continue

                if cmd.startswith("save"):
                    parts = cmd.split(maxsplit=1)
                    path = parts[1].strip() if len(parts) == 2 else self.sheet_path
                    if not path:
                        path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    self.save_data(path)
                    continue

                if cmd == "reload":
                    if self.sheet_path and Path(self.sheet_path).exists():
                        self.load_data(self.sheet_path)
                        self._default_preview(rows=8)
                    else:
                        print("[!] No original path available to reload.")
                    continue

                if cmd.startswith('load '):
                    parts = cmd.split(maxsplit=2)
                    if len(parts) != 3:
                        print('Usage: load <name> <path>')
                        continue
                    name, path_str = parts[1], parts[2]
                    try:
                        df2 = load_dataframe(path_str)
                        self.extra_dfs[name] = df2
                        print(f"✓ Loaded '{name}' ({len(df2)} rows × {len(df2.columns)} cols)")
                    except Exception as e:
                        print(f"[✗] Load error: {e}")
                    continue

                if cmd.startswith('merge '):
                    parts = cmd.split()
                    if len(parts) not in (3, 4):
                        print('Usage: merge <name> <main_col> [other_col]')
                        continue
                    name = parts[1]
                    other = self.extra_dfs.get(name)
                    if other is None:
                        print(f"[!] No sheet named '{name}'.")
                        continue
                    left_key = parts[2]
                    right_key = parts[3] if len(parts) == 4 else left_key
                    if left_key not in self.df.columns or right_key not in other.columns:
                        print('[!] Column missing in one of the sheets.')
                        continue
                    self.df = merge_dataframes(self.df, other, left_key, right_key)
                    print(f"✓ Merged '{name}' into main sheet on '{left_key}' ← '{right_key}'")
                    self._default_preview(rows=8)
                    continue

                if cmd.startswith('lookup '):
                    parts = cmd.split()
                    if len(parts) < 4:
                        print('Usage: lookup <name> <column> <value>')
                        continue
                    name, column = parts[1], parts[2]
                    value = ' '.join(parts[3:])
                    other = self.extra_dfs.get(name)
                    if other is None:
                        print(f"[!] No sheet named '{name}'.")
                        continue
                    if column not in other.columns:
                        print(f"[!] Column '{column}' not found in '{name}'.")
                        continue
                    matches = lookup_dataframe(other, column, value)
                    if matches.empty:
                        print('No matches found.')
                    else:
                        print(tabulate(matches, headers='keys', tablefmt='github', showindex=False))
                    continue
                if cmd == "undo":
                    self.undo()
                    continue

                if cmd == "apply":
                    if not self._last_generated_blocks:
                        print("[i] No pending code blocks. Use: ask <instruction>")
                        continue
                    self._apply_blocks(self._last_generated_blocks, self._last_request)
                    # clear after apply
                    self._last_generated_blocks = []
                    self._last_request = ""
                    continue

                if cmd.startswith("ask "):
                    request = cmd[4:].strip()
                    if not request:
                        print("[!] Provide an instruction after 'ask'.")
                        continue

                    self._last_request = request
                    print("🧠 Querying model...")
                    raw = self.query_llama(request)
                    if not raw:
                        print("[✗] No response from model.")
                        continue

                    blocks = self._extract_code_blocks(raw)
                    if not blocks:
                        print("[✗] No executable code blocks found.")
                        continue

                    self._last_generated_blocks = blocks

                    # Show condensed previews for *all* blocks by default
                    print(f"\nFound {len(blocks)} code block(s). Showing condensed previews:")
                    before_df = self.df.copy()
                    preview_ok = True
                    for i, code in enumerate(blocks, 1):
                        ok, preview_df, err = self._exec_on_copy(code, before_df)
                        print(f"\n--- Block {i} Preview ---")
                        if not ok or preview_df is None:
                            preview_ok = False
                            print(f"[✗] Preview failed: {err}")
                            continue
                        print(self._condensed_diff(before_df, preview_df))
                        # Chain previews (so subsequent blocks preview on top of previous)
                        before_df = preview_df

                    if preview_ok:
                        print("\nApply all blocks? [y/n] (you can also type 'apply' later)")
                        ans = input("> ").strip().lower()
                        if ans == "y":
                            self._apply_blocks(blocks, request)
                        else:
                            print("[i] Changes not applied. Use 'apply' to apply later.")
                    else:
                        print("[!] One or more previews failed. You can still try 'apply' but expect errors.")
                    continue

                if cmd == "help":
                    print("Commands:")
                    print("  ask <instruction>   # AI generates code; previews condensed diff by default")
                    print("  apply               # apply last generated code block(s)")
                    print("  undo                # undo last applied change")
                    print("  load <name> <path>  # load another sheet as DataFrame")
                    print("  merge <name> <main_col> [other_col]  # merge a loaded sheet into main")
                    print("  lookup <name> <column> <value>  # search a loaded sheet")
                    print("  show [name|n]       # show main (default) or loaded sheet, or first n rows")
                    print("  info                # detailed info tree (JSON)")
                    print("  save [path]         # save to path (or overwrite original)")
                    print("  reload              # reload original file from disk")
                    print("  quit/exit           # leave")
                    continue

                print("[!] Unknown command. Type 'help'.")

            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"[!] Error: {e}")

    # -------------------------
    #   Apply code blocks
    # -------------------------

    def _apply_blocks(self, blocks: List[str], user_request: str):
        if self.df is None:
            print("[!] No data loaded.")
            return
        working = self.df.copy()
        for idx, code in enumerate(blocks, 1):
            print(f"\n>>> Applying block {idx}/{len(blocks)}:")
            print(code)
            ok, new_df, err = self._exec_on_copy(code, working)
            if not ok or new_df is None:
                print(f"[✗] Execution failed on block {idx}: {err}")
                print("[i] Aborting remaining blocks.")
                return
            # show condensed diff for each apply (brief)
            print(self._condensed_diff(working, new_df, max_show=6))
            working = new_df

        # Snapshot and commit
        self._snapshot_for_undo("\n\n".join(blocks), user_request)
        self.df = working
        print("\n✓ All blocks applied.")
        if self.autosave:
            self.save_data(self.sheet_path)
        # show default preview after changes
        self._default_preview(rows=8)


# =========================
#          Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Minimal AI CLI Spreadsheet Agent")
    ap.add_argument("sheet", nargs="?", help="Path to CSV/XLSX file")
    ap.add_argument("--endpoint", "-e", default="http://fedora-1:7860/v1/chat/completions", help="llama.cpp chat completions endpoint")
    ap.add_argument("--model", "-m", default="default", help="Model name loaded in llama.cpp")
    ap.add_argument("--no-autosave", action="store_true", help="Disable autosave after apply/undo")
    args = ap.parse_args()

    agent = SpreadsheetAgent(
        sheet_path=args.sheet,
        llama_endpoint=args.endpoint,
        model=args.model,
        autosave=not args.no_autosave,
    )

    if not args.sheet:
        print("Usage: python agent.py <sheet.csv|sheet.xlsx> [--endpoint URL] [--model NAME]")
        sys.exit(1)

    if agent.df is None:
        sys.exit(1)

    agent.interactive()


if __name__ == "__main__":
    main()

