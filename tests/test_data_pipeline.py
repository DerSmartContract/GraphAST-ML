import os
import shutil
from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset

def test_data_pipeline():
    test_dir = "tests/data_samples"

    # Lösche den Test-Ordner, falls er noch existiert (um doppelte Dateien zu vermeiden)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "example2.py")

    # Kleines Beispielskript
    with open(test_file_path, "w") as f:
        f.write("def bar(y):\n    return y*2\n")

    # Parse den AST
    trees = parse_python_files(test_dir)
    
    # Debugging: Wie viele Bäume wurden erzeugt?
    print(f"Anzahl der geparsten AST-Bäume: {len(trees)}")
    print(f"Geparste Dateien: {[t[0] for t in trees]}")

    # Konvertiere die ASTs in Graphen
    dataset = build_dataset(trees)

    # Debugging: Anzahl der Data-Objekte aus der Pipeline
    print(f"Anzahl der erzeugten Data-Objekte: {len(dataset)}")

    # Erwartung: Genau ein Data-Objekt für die einzige Datei
    assert len(dataset) == 1, f"Genau ein Data-Objekt erwartet, aber {len(dataset)} erhalten!"

    # Cleanup nach Testlauf
    shutil.rmtree(test_dir)