import pytest
from scripts.ast_parser import parse_python_files
import os

def test_ast_parser():
    # Test: Lege im Ordner "tests/data_samples/" mindestens 1 kleines .py-Snippet an
    test_dir = "tests/data_samples"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "example.py")

    # Erstelle ein Test-Pythonfile
    with open(test_file_path, "w") as f:
        f.write("def foo(x):\n    return x+1\n")

    # Parser aufrufen
    trees = parse_python_files(test_dir)
    assert len(trees) == 1, "Es sollte genau 1 AST zurückgegeben werden"
    # 1. Element im Ergebnis: (path, ast)
    path, ast_root = trees[0]
    assert ast_root is not None, "AST-Root sollte nicht None sein"