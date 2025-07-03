import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from typing import List, Optional
import numpy as np
from src.mol_graph import smiles_to_mol


def mol_to_graph_data_obj(mol: Chem.Mol) -> Optional[Data]:
    """
    Convert an RDKit Mol object to a PyTorch Geometric Data object.
    Node features: atomic number, aromaticity
    Edge features: bond type (as integer)
    """
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic())
        ])
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([int(bond.GetBondTypeAsDouble())])
        edge_attr.append([int(bond.GetBondTypeAsDouble())])
    if not atom_features:
        return None
    x = torch.tensor(atom_features, dtype=torch.float)
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class SmilesDataset(Dataset):
    def __init__(self, smiles_list: List[str], targets: Optional[List[float]] = None):
        super().__init__()
        self.smiles_list = smiles_list
        self.targets = targets

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        mol = smiles_to_mol(self.smiles_list[idx])
        data = mol_to_graph_data_obj(mol)
        if data is None:
            data = Data()
        if self.targets is not None:
            data.y = torch.tensor([self.targets[idx]], dtype=torch.float)
        return data


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=32, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


def example_usage():
    from src.data import load_smiles_from_csv
    smiles_list = load_smiles_from_csv('data/molecules.csv')
    # Fake targets for demonstration
    targets = np.random.rand(len(smiles_list)).tolist()
    dataset = SmilesDataset(smiles_list, targets)
    data = dataset.get(0)
    model = GCN(num_node_features=data.x.shape[1])
    out = model(data.x, data.edge_index, torch.zeros(data.x.shape[0], dtype=torch.long))
    print(f"Model output: {out}") 