# ai_xcel

AI coding agent.

## CLI for referencing sheets

Run `python sheet_cli.py path/to/main.csv` to load a main sheet. At the `command>` prompt:

- `load <name> <path>` loads another sheet.
- `merge <name> <main_col> [other_col]` joins columns from another sheet into the main sheet.
- `lookup <name> <column> <value>` shows matching rows from a loaded sheet without merging.
- `show [main|name]` prints the first few rows of the main sheet or a loaded sheet.
- `save [path]` saves the main sheet (defaults to the original path).
- `exit` quits the program.
