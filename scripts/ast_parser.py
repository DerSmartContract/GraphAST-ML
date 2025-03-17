import ast
import astpretty
import os
from pathlib import Path

def parse_python_files(input_dir: str):
    """
    Liest alle .py-Dateien in input_dir ein, erzeugt einen AST
    und gibt eine Liste (pfad, ast_root) zurück.
    """
    ast_trees = []
    for file_path in Path(input_dir).rglob("*.py"):
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
            try:
                tree = ast.parse(source_code, filename=str(file_path))
                ast_trees.append((file_path, tree))
            except SyntaxError as e:
                # Falls eine Datei nicht parsebar ist, überspringen
                print(f"[WARN] SyntaxError in {file_path}: {e}")
    return ast_trees

if __name__ == "__main__":
    # Beispiel: 
    # AST für alle Dateien in data/raw/ erzeugen und ausgeben
    input_directory = "data/raw"
    trees = parse_python_files(input_directory)
    for path, tree in trees:
        print(f"\n--- AST für: {path} ---")
        astpretty.pprint(tree, indent=2)