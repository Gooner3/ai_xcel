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

# Optional GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


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

        self.llama_endpoint = llama_endpoint
        self.model = model
        self.autosave = autosave

        self.sheet_path = sheet_path
        self.df: Optional[pd.DataFrame] = None
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
            if file_path.lower().endswith(".csv"):
                self.df = pd.read_csv(file_path)
            elif file_path.lower().endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(file_path)
            else:
                print(f"[!] Unsupported file format: {file_path}")
                return False
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

    def query_llama(self, user_request: str) -> str:
        """Send info tree + request to llama.cpp and return raw text."""
        system_prompt = self._read_prompt("system.txt")
        analyze_t = self._read_prompt("analyze.txt")

        info_tree = self.create_info_tree()
        info_json = json.dumps(info_tree, indent=2, cls=NumpyEncoder)

        prompt = analyze_t.format(info_tree=info_json, user_request=user_request)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1500,
        }

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
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            exec(code, {"__builtins__": {}}, local_vars)
            new_df = local_vars.get("df", None)
            if new_df is None or not isinstance(new_df, pd.DataFrame):
                # If they mutated in-place and didn't reassign, use the mutated df
                new_df = local_vars.get("df", df)
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
        print("  show [n]            # show first n rows (default 8)")
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
                    n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 8
                    self._default_preview(rows=n)
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
                    print("  show [n]            # show first n rows (default 8)")
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
#   Simple merge GUI
# =========================


class SimpleMergeGUI:
    """Minimal Tkinter interface for merging and referencing sheets."""

    def __init__(self, main_path: str):
        self.main_path = Path(main_path)
        self.main_df = self._load_dataframe(self.main_path)
        self.other_df: Optional[pd.DataFrame] = None

        self.root = tk.Tk()
        self.root.title("Sheet Merger")

        self.text = tk.Text(self.root, width=100, height=20)
        self.text.pack(fill="both", expand=True)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Load Secondary", command=self.load_secondary).pack(side="left")
        tk.Button(btn_frame, text="Merge", command=self.merge).pack(side="left")
        tk.Button(btn_frame, text="Lookup", command=self.lookup).pack(side="left")
        tk.Button(btn_frame, text="View Secondary", command=self.view_secondary).pack(side="left")
        tk.Button(btn_frame, text="Save", command=self.save).pack(side="left")

        self.update_preview()

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file type: {path}")

    def load_secondary(self):
        path = filedialog.askopenfilename(
            title="Select secondary sheet",
            filetypes=[("Spreadsheets", "*.csv *.xls *.xlsx")],
        )
        if not path:
            return
        try:
            self.other_df = self._load_dataframe(Path(path))
            messagebox.showinfo(
                "Loaded",
                f"Loaded secondary sheet ({len(self.other_df)} rows × {len(self.other_df.columns)} cols)",
            )
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def merge(self):
        if self.other_df is None:
            messagebox.showwarning("Merge", "Load a secondary sheet first.")
            return
        left = simpledialog.askstring("Merge", "Column in main sheet:")
        if not left:
            return
        right = simpledialog.askstring(
            "Merge", "Column in secondary sheet (leave blank for same column):"
        )
        right = right or left
        if left not in self.main_df.columns or right not in self.other_df.columns:
            messagebox.showerror("Merge", "Column not found in one of the sheets.")
            return
        self.main_df = self.main_df.merge(
            self.other_df, left_on=left, right_on=right, how="left"
        )
        messagebox.showinfo("Merge", f"Merged on {left} ← {right}.")
        self.update_preview()

    def lookup(self):
        if self.other_df is None:
            messagebox.showwarning("Lookup", "Load a secondary sheet first.")
            return
        column = simpledialog.askstring("Lookup", "Column in secondary sheet:")
        value = simpledialog.askstring("Lookup", "Value to match:")
        if not column or not value:
            return
        if column not in self.other_df.columns:
            messagebox.showerror("Lookup", "Column not found.")
            return
        matches = self.other_df[self.other_df[column].astype(str) == value]
        if matches.empty:
            messagebox.showinfo("Lookup", "No matches found.")
        else:
            messagebox.showinfo("Lookup", matches.head().to_string())

    def view_secondary(self):
        if self.other_df is None:
            messagebox.showwarning("View", "No secondary sheet loaded.")
            return
        messagebox.showinfo("Secondary preview", self.other_df.head().to_string())

    def save(self):
        path = filedialog.asksaveasfilename(
            initialfile=self.main_path.name,
            defaultextension=self.main_path.suffix,
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("Excel", "*.xls")],
        )
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".csv":
                self.main_df.to_csv(path, index=False)
            else:
                self.main_df.to_excel(path, index=False)
            messagebox.showinfo("Save", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def update_preview(self):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, self.main_df.head().to_string())

    def run(self):
        self.root.mainloop()

# =========================
#          Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Minimal AI CLI Spreadsheet Agent")
    ap.add_argument("sheet", nargs="?", help="Path to CSV/XLSX file")
    ap.add_argument("--endpoint", "-e", default="http://fedora-1:7860/v1/chat/completions", help="llama.cpp chat completions endpoint")
    ap.add_argument("--model", "-m", default="default", help="Model name loaded in llama.cpp")
    ap.add_argument("--no-autosave", action="store_true", help="Disable autosave after apply/undo")
    ap.add_argument("--gui", action="store_true", help="Launch simple Tkinter GUI for merging/lookup")
    args = ap.parse_args()

    if args.gui:
        if not args.sheet:
            ap.error("sheet required for GUI mode")
        SimpleMergeGUI(args.sheet).run()
        return

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

