import torch
from torch_geometric.loader import DataLoader

from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset
from scripts.model_training import SimpleGCN

def load_model(model_path, input_dim=1, hidden_dim=16, output_dim=2):
    model = SimpleGCN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_inference(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    for data_obj in loader:
        with torch.no_grad():
            out = model(data_obj)
            graph_embedding = out.mean(dim=0, keepdim=True)
            # Hier nur Argmax als Beispiel
            pred_class = torch.argmax(graph_embedding, dim=1).item()
            results.append(pred_class)
    return results

if __name__ == "__main__":
    # Beispielhafter Ablauf: Nimm an, wir haben model.pt (trainiert)
    model_path = "model.pt"  # Pfad anpassen an dein Setup
    model = load_model(model_path)

    # ASTs parsen
    new_code_dir = "data/raw"  # Oder beliebiger anderer Ordner
    ast_list = parse_python_files(new_code_dir)
    dataset = build_dataset(ast_list)

    # Inferenz durchführen
    preds = run_inference(model, dataset)
    for i, pred in enumerate(preds):
        print(f"Datei {ast_list[i][0]} -> Prognose-Klasse: {pred}")