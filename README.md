# ai_xcel

AI coding agent.

## CLI usage

Run `python main_chatgpt.py path/to/main.csv` to load the primary sheet. At the `>` prompt you can:

- `ask <instruction>` – have the AI generate pandas code that previews before applying.
- `apply` – apply the last generated code block(s).
- `undo` – undo the most recent change.
- `load <name> <path>` – load another CSV/XLS(X) as a DataFrame for reference.
- `merge <name> <main_col> [other_col]` – merge a loaded sheet into the main sheet on key columns.
- `lookup <name> <column> <value>` – view matching rows from a loaded sheet without merging.
- `show [name|n]` – display the main sheet (default), a loaded sheet by name, or show the first `n` rows of the main sheet.
- `save [path]` – save the main sheet; secondary sheets are never written.
- `reload` – reload the original main sheet from disk.
- `quit` – exit the program.

Secondary sheets are kept in memory only so you can reference or merge their data while keeping the primary sheet under your control.

Every prompt sent to the model is appended to `api_prompts.log` in JSON lines format for later review.

Place any custom guidance for the AI in `.prompts/instructions.txt`; its contents are prepended to every request sent to the model.
