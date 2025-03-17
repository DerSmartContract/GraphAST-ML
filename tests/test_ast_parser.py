import pytest
from scripts.ast_parser import parse_python_files

def test_ast_parser():
    test_dir = "tests/data_samples"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "example.py")

    with open(test_file_path, "w") as f:
        f.write("def foo(x): return x * 2")

    trees = parse_python_files(test_dir)
    assert len(trees) > 0, "Es sollte mindestens ein AST geparst werden."