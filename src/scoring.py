import numpy as np
from typing import List, Tuple, Dict

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to [0, 1].
    """
    arr = np.array(scores)
    if arr.max() == arr.min():
        return [0.5] * len(scores)
    return ((arr - arr.min()) / (arr.max() - arr.min())).tolist()


def rank_molecules(smiles_list: List[str], gnn_scores: List[float], docking_scores: List[float], alpha: float = 0.5) -> List[Tuple[str, float]]:
    """
    Rank molecules by a weighted sum of normalized GNN and docking scores.
    Args:
        smiles_list (List[str]): List of SMILES strings.
        gnn_scores (List[float]): GNN-predicted property scores (higher is better).
        docking_scores (List[float]): Docking scores (lower is better).
        alpha (float): Weight for GNN score (1-alpha for docking score).
    Returns:
        List[Tuple[str, float]]: List of (SMILES, combined_score), sorted descending.
    """
    gnn_norm = normalize_scores(gnn_scores)
    docking_norm = normalize_scores([-s for s in docking_scores])  # Lower docking is better
    combined = [alpha * g + (1 - alpha) * d for g, d in zip(gnn_norm, docking_norm)]
    ranked = sorted(zip(smiles_list, combined), key=lambda x: x[1], reverse=True)
    return ranked


def example_usage():
    smiles_list = ['CCO', 'CC(=O)O', 'C1=CC=CC=C1']
    gnn_scores = [0.8, 0.5, 0.9]  # Example GNN predictions
    docking_scores = [-7.2, -6.5, -8.1]  # Example docking scores (lower is better)
    ranked = rank_molecules(smiles_list, gnn_scores, docking_scores, alpha=0.6)
    print("Ranking:")
    for smi, score in ranked:
        print(f"{smi}: {score:.3f}") 