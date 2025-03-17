import pytest
from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset
import os

def test_data_pipeline():
    test_dir = "tests/data_samples"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "example2.py")

    # Kleines Beispielskript
    with open(test_file_path, "w") as f:
        f.write("def bar(y):\n    return y*2\n")

    trees = parse_python_files(test_dir)
    dataset = build_dataset(trees)
    assert len(dataset) == 1, "Genau ein Data-Objekt erwartet"
    data_obj = dataset[0]

    # Prüfen, ob wir mindestens 1 Knoten haben
    assert data_obj.x.size(0) > 0, "Mindestens 1 AST-Knoten erwartet"