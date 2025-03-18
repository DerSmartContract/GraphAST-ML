import torch
from scripts.model_training import SimpleGCN
from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset

def load_model(model_path, input_dim=1, hidden_dim=16, output_dim=2):
    """Lädt das gespeicherte Modell für Vorhersagen."""
    model = SimpleGCN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == "__main__":
    model_path = "saved_models/trained_model.pth"
    try:
        model = load_model(model_path)
        print(f"✅ Modell '{model_path}' erfolgreich geladen!")
    except FileNotFoundError:
        print(f"❌ Modell '{model_path}' nicht gefunden!")
        exit()

    print("🔍 Parsing neuer Code-Dateien ...")
    trees = parse_python_files("data/raw")

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
            predicted_classes = out.argmax(dim=-1).tolist()
            print(f"🔹 Vorhersagen für Datei {i+1} (Shape: {out.shape}): {predicted_classes}")