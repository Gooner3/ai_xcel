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

## Simple GUI for merging sheets

Run `python main_chatgpt.py path/to/main.csv --gui` to open a small Tkinter window.
The interface keeps the first sheet as the main DataFrame and lets you:

- **Load Secondary** – import another sheet for reference.
- **Merge** – join columns from the secondary sheet into the main sheet.
- **Lookup** – quickly view matching rows from the secondary sheet without merging.
- **View Secondary** – preview the loaded secondary sheet.
- **Save** – export the updated main sheet to CSV or Excel.
