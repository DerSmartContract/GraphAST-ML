import ast
import os
from pathlib import Path

def parse_python_files(input_dir: str):
    """
    Liest alle .py-Dateien in input_dir ein, erzeugt einen AST
    und gibt diesen als Liste von (Pfad, AST) zurück.
    """
    ast_trees = []
    files_found = 0

    for file_path in Path(input_dir).rglob("*.py"):
        files_found += 1
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                source_code = f.read()
                tree = ast.parse(source_code, filename=str(file_path))
                ast_trees.append((file_path, tree))
                print(f"✔️ AST erfolgreich erstellt für {file_path}")
            except SyntaxError as e:
                print(f"❌ SyntaxError in Datei {file_path}: {e}")

    print(f"📂 Gefundene Python-Dateien: {files_found}")
    return ast_trees