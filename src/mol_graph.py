from rdkit import Chem
import networkx as nx
from typing import Any, Dict


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol object.
    Args:
        smiles (str): SMILES string.
    Returns:
        Chem.Mol: RDKit molecule object, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol


def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """
    Convert an RDKit Mol object to a NetworkX graph with atom and bond features.
    Args:
        mol (Chem.Mol): RDKit molecule object.
    Returns:
        nx.Graph: NetworkX graph with atom and bond features.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   symbol=atom.GetSymbol(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   bond_type=str(bond.GetBondType()),
                   is_conjugated=bond.GetIsConjugated())
    return G


def example_usage():
    smiles = 'CCO'  # Ethanol
    mol = smiles_to_mol(smiles)
    if mol:
        G = mol_to_nx(mol)
        print(f"Nodes: {G.nodes(data=True)}")
        print(f"Edges: {G.edges(data=True)}")
    else:
        print("Invalid SMILES.") 