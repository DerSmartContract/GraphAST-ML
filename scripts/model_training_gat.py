import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from scripts.data_pipeline import build_dataset
from scripts.ast_parser import parse_python_files
import os

class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

def train_gat_model(dataset, epochs=20, batch_size=2):
    if not dataset:
        raise ValueError("❌ Keine Daten zum Trainieren gefunden!")

    print(f"🚀 Starte GAT-Modelltraining ...")
    print(f"Dataset-Größe: {len(dataset)}")

    # Dummy-Labels generieren
    for d in dataset:
        d.y = torch.tensor([0], dtype=torch.long)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GATModel(input_dim=1, hidden_dim=32, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            graph_embedding = out.mean(dim=0, keepdim=True)
            loss = loss_fn(graph_embedding, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Modell speichern
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/trained_gat_model.pth")
    print("💾 GAT-Modell gespeichert unter 'saved_models/trained_gat_model.pth'")

    return model

if __name__ == "__main__":
    print(f"Torch Version: {torch.__version__}")
    print("🔍 Parsing Quellcode-Dateien ...")
    trees = parse_python_files("data/raw")
    
    if not trees:
        print("⚠️ Keine Quellcode-Dateien gefunden!")
        exit()

    print(f"📂 Gefundene Python-Dateien: {len(trees)}")

    dataset = build_dataset(trees)
    if not dataset:
        print("⚠️ Keine AST-Datenpunkte generiert!")
        exit()

    print(f"📊 Erstellt {len(dataset)} Graph-Datenpunkte")

    trained_model = train_gat_model(dataset, epochs=10, batch_size=2)
    print("✅ GAT-Modelltraining abgeschlossen!")
