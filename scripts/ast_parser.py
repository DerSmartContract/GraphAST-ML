import ast
import os
from pathlib import Path

def parse_python_files(input_dir: str):
    """Liest alle .py-Dateien ein, erzeugt ASTs und gibt eine Liste zurück."""
    ast_trees = []
    for file_path in Path(input_dir).rglob("*.py"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                source_code = f.read()
                tree = ast.parse(source_code, filename=str(file_path))
                ast_trees.append((file_path, tree))
            except SyntaxError as e:
                print(f"⚠️ SyntaxError in Datei {file_path}: {e}")
    
    print(f"📂 Gefundene Python-Dateien: {len(ast_trees)}")
    return ast_trees

if __name__ == "__main__":
    trees = parse_python_files("data/raw")
    for path, _ in trees:
        print(f"✔️ AST erfolgreich erstellt für {path}")