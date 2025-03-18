import os
import torch
import argparse
from pathlib import Path
import torch.nn.functional as F
from scripts.model_training import SimpleGCN
from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset

def load_model(model_path, input_dim=1, hidden_dim=16, output_dim=2):
    """Lädt das gespeicherte Modell für Vorhersagen."""
    model = SimpleGCN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AST-Inferenz mit GCN")
    parser.add_argument("--input_file", type=str, help="Pfad zu einer Python-Datei")
    parser.add_argument("--input_folder", type=str, help="Pfad zu einem Ordner mit Python-Dateien")
    args = parser.parse_args()

    model_path = "saved_models/trained_model.pth"

    # Modell laden
    try:
        model = load_model(model_path)
        print(f"✅ Modell '{model_path}' erfolgreich geladen!")
    except FileNotFoundError:
        print(f"❌ Modell '{model_path}' nicht gefunden! Bitte erst trainieren mit:")
        print("   python -m scripts.model_training")
        exit()

    # Eingabe-Datei oder Ordner bestimmen
    if args.input_file and Path(args.input_file).is_file():
        input_path = args.input_file
    elif args.input_folder and Path(args.input_folder).is_dir():
        input_path = args.input_folder
    else:
        print("❌ Fehler: Bitte eine gültige Datei oder ein Verzeichnis angeben!")
        exit()

    print(f"🔍 Parsing neuer Code-Dateien aus: {input_path} ...")
    trees = parse_python_files(input_path)

    if not trees:
        print("⚠️ Keine Dateien gefunden!")
        exit()

    dataset = build_dataset(trees)
    
    if not dataset:
        print("⚠️ Keine Daten zur Vorhersage verfügbar!")
        exit()

    print("🧠 Modell macht Vorhersagen ...")
    for i, data in enumerate(dataset):
        with torch.no_grad():
            out = model(data)
            probabilities = F.softmax(out, dim=-1).tolist()
            predicted_classes = out.argmax(dim=-1).tolist()

            print(f"🔹 Vorhersagen für Datei {i+1} (Shape: {out.shape}):")
            for idx, (pred, prob) in enumerate(zip(predicted_classes, probabilities)):
                print(f"   🔸 Node {idx}: Klasse {pred} (Confidence: {max(prob):.2f})")