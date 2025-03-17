import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import build_dataset

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

def train_model(dataset, epochs=5, batch_size=1):
    """
    Trainiert ein simples GCN-Modell auf den gegebenen Data-Objekten.
    Derzeit werden Dummy-Labels (0) verwendet, 
    da wir keine echten Labels definiert haben.
    """
    # Dummy-Labels generieren (zwei Klassen: 0 oder 1 -> wir nehmen einfach 0)
    for data_obj in dataset:
        data_obj.y = torch.tensor([0], dtype=torch.long)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleGCN(input_dim=1, hidden_dim=16, output_dim=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            
            # Ausgabedimension: [#Knoten, 2]
            # Minimaler Ansatz: average pooling (mean) aller Knoten als Graph-Output
            graph_embedding = out.mean(dim=0, keepdim=True)  # shape [1, 2]
            
            # batch.y hat shape [1], z.B. [0]
            loss = loss_fn(graph_embedding, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model

if __name__ == "__main__":
    # End-to-End-Beispiel
    input_dir = "data/raw"
    ast_list = parse_python_files(input_dir)
    dataset = build_dataset(ast_list)
    trained_model = train_model(dataset, epochs=5, batch_size=1)
    print("Modelltraining abgeschlossen.")