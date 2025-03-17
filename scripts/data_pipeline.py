import networkx as nx
from torch_geometric.data import Data
import torch
import ast

def ast_to_networkx(ast_node, graph=None, parent=None):
    """Konvertiert AST in ein NetworkX-Graphobjekt."""
    if graph is None:
        graph = nx.DiGraph()

    node_id = len(graph.nodes)
    graph.add_node(node_id, ast_type=type(ast_node).__name__)

    if parent is not None:
        graph.add_edge(parent, node_id)

    for field_name in ast_node._fields or []:
        child = getattr(ast_node, field_name, None)
        if isinstance(child, list):
            for subchild in child:
                if isinstance(subchild, ast.AST):
                    ast_to_networkx(subchild, graph, node_id)
        elif isinstance(child, ast.AST):
            ast_to_networkx(child, graph, node_id)

    return graph

def networkx_to_torch_geometric(nx_graph):
    """Konvertiert NetworkX-Graph in PyTorch-Geometric Data-Objekt."""
    node_attributes = [[hash(data["ast_type"]) % 1000] for _, data in nx_graph.nodes(data=True)]
    x = torch.tensor(node_attributes, dtype=torch.float)
    edges = list(nx_graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def build_dataset(ast_trees):
    """Konvertiert ASTs in Graph-Format für ML-Modell."""
    dataset = []
    for path, tree in ast_trees:
        nx_g = ast_to_networkx(tree)
        tg_data = networkx_to_torch_geometric(nx_g)
        dataset.append(tg_data)

    print(f"📊 Erstellt {len(dataset)} Graph-Datenpunkte")
    return dataset