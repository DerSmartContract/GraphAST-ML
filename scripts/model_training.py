import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from scripts.data_pipeline import build_dataset
from scripts.ast_parser import parse_python_files

# Debugging: Prüfen, ob Torch und PyG richtig installiert sind
print(f"Torch Version: {torch.__version__}")

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_model(dataset, epochs=10, batch_size=1):
    # 🔥 Debugging: Prüfen, ob Dataset nicht leer ist
    print(f"Dataset-Größe: {len(dataset)}")
    assert len(dataset) > 0, "Fehler: Dataset ist leer! Stelle sicher, dass parse_python_files() Daten liefert."

    # Dummy-Labels generieren (zwei Klassen)
    for d in dataset:
        d.y = torch.tensor([0], dtype=torch.long)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleGCN(input_dim=1, hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            graph_embedding = out.mean(dim=0, keepdim=True)  # shape [1, 2]
            loss = loss_fn(graph_embedding, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

if __name__ == "__main__":
    print("🔍 Parsing Quellcode-Dateien ...")
    trees = parse_python_files("data/raw")

    print(f"📂 Gefundene Python-Dateien: {len(trees)}")
    for path, _ in trees:
        print(f"✔️ AST erfolgreich erstellt für {path}")

    print("📊 Konvertiere ASTs in Graphen ...")
    dataset = build_dataset(trees)

    print("🚀 Starte Modelltraining ...")
    trained_model = train_model(dataset, epochs=5, batch_size=1)
    print("✅ Modelltraining abgeschlossen!")