import ast
import os
from pathlib import Path

def parse_python_files(input_path: str):
    """
    Liest eine Datei oder alle .py-Dateien in input_path ein,
    erzeugt einen AST und gibt eine Liste von (Pfad, AST) zurück.
    """
    ast_trees = []
    files_found = 0

    path_obj = Path(input_path)

    # Falls eine einzelne Datei angegeben wurde
    if path_obj.is_file() and path_obj.suffix == ".py":
        files_found = 1
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                source_code = f.read()
                tree = ast.parse(source_code, filename=str(path_obj))
                ast_trees.append((path_obj, tree))
                print(f"✔️ AST erfolgreich erstellt für {path_obj}")
        except SyntaxError as e:
            print(f"❌ SyntaxError in Datei {path_obj}: {e}")

    # Falls ein Verzeichnis angegeben wurde
    elif path_obj.is_dir():
        for file_path in path_obj.rglob("*.py"):
            files_found += 1
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                    tree = ast.parse(source_code, filename=str(file_path))
                    ast_trees.append((file_path, tree))
                    print(f"✔️ AST erfolgreich erstellt für {file_path}")
            except SyntaxError as e:
                print(f"❌ SyntaxError in Datei {file_path}: {e}")

    else:
        print(f"❌ Fehler: '{input_path}' ist keine gültige Datei oder Verzeichnis.")

    print(f"📂 Gefundene Python-Dateien: {files_found}")
    return ast_trees