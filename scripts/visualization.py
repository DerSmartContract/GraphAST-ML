import ast
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from scripts.ast_parser import parse_python_files
from scripts.data_pipeline import ast_to_networkx, networkx_to_torch_geometric

def draw_ast_graph(ast_tree, output_file=None):
    """Erstellt eine AST-Visualisierung als Baumdiagramm."""
    graph = ast_to_networkx(ast_tree)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=42)
    labels = {node: graph.nodes[node]['ast_type'] for node in graph.nodes}
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray", node_size=1000, font_size=8)

    if output_file:
        plt.savefig(output_file)
        print(f"📸 AST-Graph gespeichert als {output_file}")
    else:
        plt.show()

def draw_torch_graph(tg_data, output_file=None):
    """Erstellt eine Visualisierung des Graph-Datenmodells."""
    graph = nx.Graph()
    
    # Knoten hinzufügen
    for i in range(tg_data.x.shape[0]):
        graph.add_node(i)

    # Kanten hinzufügen
    edge_index = tg_data.edge_index.numpy().T
    for edge in edge_index:
        graph.add_edge(edge[0], edge[1])

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color="lightcoral", edge_color="black", node_size=800, font_size=8)

    if output_file:
        plt.savefig(output_file)
        print(f"📸 PyTorch-Geometric Graph gespeichert als {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisierung von ASTs und Graphen.")
    parser.add_argument("--input_file", type=str, required=True, help="Pfad zur Python-Datei zur Visualisierung")
    parser.add_argument("--output", type=str, default=None, help="Optional: Speicherpfad für das Bild")
    args = parser.parse_args()

    print(f"🔍 Parsing Datei: {args.input_file}")
    trees = parse_python_files(args.input_file)

    if not trees:
        print("⚠️ Keine ASTs gefunden!")
        exit()

    for i, (file_path, tree) in enumerate(trees):
        print(f"📂 Visualisierung von {file_path} ...")

        draw_ast_graph(tree, output_file=f"{args.output}_ast.png" if args.output else None)
        
        tg_data = networkx_to_torch_geometric(ast_to_networkx(tree))
        draw_torch_graph(tg_data, output_file=f"{args.output}_torch.png" if args.output else None)
    
    print("✅ Visualisierung abgeschlossen!")