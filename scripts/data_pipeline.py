import ast
import networkx as nx
import torch
from torch_geometric.data import Data

def ast_to_networkx(ast_node, graph=None, parent=None):
    """
    Rekursive Funktion, um einen AST in ein networkx-DiGraph umzuwandeln.
    """
    if graph is None:
        graph = nx.DiGraph()

    node_id = len(graph.nodes)
    graph.add_node(node_id, ast_type=type(ast_node).__name__)

    # Falls es einen Elternknoten gibt, Kante hinzufügen
    if parent is not None:
        graph.add_edge(parent, node_id)

    # Durch die Felder des AST-Knotens iterieren
    for field_name in getattr(ast_node, "_fields", []):
        child = getattr(ast_node, field_name, None)
        if isinstance(child, list):
            for subchild in child:
                if isinstance(subchild, ast.AST):
                    ast_to_networkx(subchild, graph, parent=node_id)
        elif isinstance(child, ast.AST):
            ast_to_networkx(child, graph, parent=node_id)

    return graph

def networkx_to_torch_geometric(nx_graph):
    """
    Wandelt ein networkx-DiGraph in ein PyTorch-Geometric Data-Objekt um.
    """
    # Knotenfeatures
    node_attributes = []
    for _, node_data in nx_graph.nodes(data=True):
        # Bsp.: Nur der Knotentyp als Feature, via Hash reingequetscht
        ast_type = node_data.get("ast_type", "Unknown")
        # Minimal: x dimension = 1
        node_attributes.append([hash(ast_type) % 1000])

    x = torch.tensor(node_attributes, dtype=torch.float)

    # Kantenliste (Gerichteter Graph)
    edges = list(nx_graph.edges())
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Keine Kanten -> Leeres Tensor
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data

def build_dataset(ast_trees):
    """
    Nimmt eine Liste (pfad, AST) entgegen und baut
    daraus eine Liste von PyTorch-Geometric Data-Objekten.
    """
    dataset = []
    for path, tree in ast_trees:
        nx_graph = ast_to_networkx(tree)
        tg_data = networkx_to_torch_geometric(nx_graph)
        dataset.append(tg_data)
    return dataset