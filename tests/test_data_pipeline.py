import pytest
from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset

def test_data_pipeline():
    test_dir = "tests/data_samples"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "example2.py")

    with open(test_file_path, "w") as f:
        f.write("def bar(y): return y * 2")

    trees = parse_python_files(test_dir)
    dataset = build_dataset(trees)
    
    assert len(dataset) > 0, "Dataset sollte mindestens ein Data-Objekt enthalten"