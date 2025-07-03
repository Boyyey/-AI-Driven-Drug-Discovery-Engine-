def test_data_ingestion():
    print("\n=== Data Ingestion ===")
    from src.data import load_smiles_from_csv
    smiles_list = load_smiles_from_csv('data/molecules.csv')
    print("Loaded SMILES:", smiles_list)
    return smiles_list


def test_mol_graph(smiles):
    print("\n=== Molecular Graph Conversion ===")
    from src.mol_graph import smiles_to_mol, mol_to_nx
    mol = smiles_to_mol(smiles)
    if mol:
        G = mol_to_nx(mol)
        print("Nodes:", G.nodes(data=True))
        print("Edges:", G.edges(data=True))
    else:
        print("Invalid SMILES.")


def test_gnn(smiles_list):
    print("\n=== GNN for Property Prediction ===")
    from src.gnn import SmilesDataset, GCN
    import numpy as np
    import torch
    targets = np.random.rand(len(smiles_list)).tolist()
    dataset = SmilesDataset(smiles_list, targets)
    data = dataset.get(0)
    model = GCN(num_node_features=data.x.shape[1])
    out = model(data.x, data.edge_index, torch.zeros(data.x.shape[0], dtype=torch.long))
    print("GNN model output:", out)
    return targets


def test_generative():
    print("\n=== Generative Model ===")
    from src.generative import SmilesVAE, sample_smiles
    model = SmilesVAE()
    samples = sample_smiles(model, num_samples=3)
    print("Sampled valid SMILES:", samples)
    return samples


def test_docking(smiles):
    print("\n=== Docking Simulation Interface ===")
    try:
        from src.docking import smiles_to_pdbqt, run_vina
        import tempfile, os
        protein_pdbqt = 'protein.pdbqt'  # Replace with your actual file
        with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as ligand_f:
            ligand_pdbqt = ligand_f.name
        if smiles_to_pdbqt(smiles, ligand_pdbqt):
            score = run_vina(ligand_pdbqt, protein_pdbqt, center='0 0 0', size='20 20 20')
            print('Docking score:', score)
        else:
            print('Failed to prepare ligand.')
        if os.path.exists(ligand_pdbqt):
            os.remove(ligand_pdbqt)
    except Exception as e:
        print("Docking test skipped (requirements not met):", e)


def test_scoring(smiles_list, targets):
    print("\n=== Scoring/Ranking ===")
    from src.scoring import rank_molecules
    # Fake docking scores for demo
    docking_scores = [-7.2, -6.5, -8.1][:len(smiles_list)]
    ranked = rank_molecules(smiles_list, targets, docking_scores, alpha=0.6)
    print("Ranking:")
    for smi, score in ranked:
        print(f"{smi}: {score:.3f}")


def main():
    smiles_list = test_data_ingestion()
    test_mol_graph(smiles_list[0])
    targets = test_gnn(smiles_list)
    test_generative()
    test_docking(smiles_list[0])
    test_scoring(smiles_list, targets)


if __name__ == "__main__":
    main() 