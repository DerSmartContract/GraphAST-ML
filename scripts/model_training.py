import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from scripts.data_pipeline import build_dataset
from scripts.ast_parser import parse_python_files
import os

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
    if not dataset:
        raise ValueError("❌ Keine Daten zum Trainieren gefunden!")

    print(f"🚀 Starte Modelltraining ...")
    print(f"Dataset-Größe: {len(dataset)}")

    # Dynamisch den Input-Dimension berechnen
    input_dim = dataset[0].x.shape[1]

    # Labels generieren (Dummy-Zuweisung)
    for d in dataset:
        d.y = torch.tensor([0], dtype=torch.long)  # Dummy-Label für jetzt

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleGCN(input_dim=input_dim, hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            graph_embedding = out.mean(dim=0, keepdim=True)
            
            # Sicherstellen, dass die Shape für Loss passt
            target = batch.y.view(-1)  
            loss = loss_fn(graph_embedding, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(f"Epoch {epoch+1}: Loss {total_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Trainingslog speichern
    os.makedirs("logs", exist_ok=True)
    with open("logs/training_log.txt", "w") as log_file:
        log_file.write("\n".join(losses))

    # Modell speichern
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/trained_model.pth")
    print("💾 Modell gespeichert unter 'saved_models/trained_model.pth'")

    return model

if __name__ == "__main__":
    print(f"Torch Version: {torch.__version__}")
    print("🔍 Parsing Quellcode-Dateien ...")

    if not os.path.exists("data/raw"):
        print("❌ Der Pfad 'data/raw' existiert nicht! Bitte sicherstellen, dass Daten vorhanden sind.")
        exit()

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

    trained_model = train_model(dataset, epochs=5, batch_size=1)
    print("✅ Modelltraining abgeschlossen!")