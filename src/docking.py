import subprocess
import tempfile
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional


def smiles_to_pdbqt(smiles: str, pdbqt_path: str) -> bool:
    """
    Convert a SMILES string to a PDBQT file using RDKit and OpenBabel (if available).
    Args:
        smiles (str): Ligand SMILES string.
        pdbqt_path (str): Output path for PDBQT file.
    Returns:
        bool: True if successful, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    tmp_sdf = pdbqt_path.replace('.pdbqt', '.sdf')
    Chem.MolToMolFile(mol, tmp_sdf)
    # Convert SDF to PDBQT using OpenBabel (must be installed)
    try:
        subprocess.run(['obabel', tmp_sdf, '-O', pdbqt_path], check=True)
        os.remove(tmp_sdf)
        return True
    except Exception as e:
        print(f"OpenBabel conversion failed: {e}")
        return False


def run_vina(ligand_pdbqt: str, protein_pdbqt: str, center: str, size: str, vina_path: str = 'vina') -> Optional[float]:
    """
    Run AutoDock Vina docking simulation and parse the best score.
    Args:
        ligand_pdbqt (str): Path to ligand PDBQT file.
        protein_pdbqt (str): Path to protein PDBQT file.
        center (str): Center of the search box (e.g., '0 0 0').
        size (str): Size of the search box (e.g., '20 20 20').
        vina_path (str): Path to Vina executable.
    Returns:
        Optional[float]: Best docking score (lower is better), or None if failed.
    """
    with tempfile.NamedTemporaryFile(delete=False) as out_f:
        out_path = out_f.name
    try:
        result = subprocess.run([
            vina_path,
            '--receptor', protein_pdbqt,
            '--ligand', ligand_pdbqt,
            '--center_x', center.split()[0],
            '--center_y', center.split()[1],
            '--center_z', center.split()[2],
            '--size_x', size.split()[0],
            '--size_y', size.split()[1],
            '--size_z', size.split()[2],
            '--out', out_path
        ], capture_output=True, text=True, check=True)
        # Parse score from output
        for line in result.stdout.splitlines():
            if line.strip().startswith('1 '):
                parts = line.split()
                if len(parts) > 1:
                    return float(parts[1])
    except Exception as e:
        print(f"Vina docking failed: {e}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    return None


def example_usage():
    smiles = 'CCO'  # Example ligand
    protein_pdbqt = 'protein.pdbqt'  # Example protein file
    with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as ligand_f:
        ligand_pdbqt = ligand_f.name
    if smiles_to_pdbqt(smiles, ligand_pdbqt):
        score = run_vina(ligand_pdbqt, protein_pdbqt, center='0 0 0', size='20 20 20')
        print(f'Docking score: {score}')
    else:
        print('Failed to prepare ligand.')
    if os.path.exists(ligand_pdbqt):
        os.remove(ligand_pdbqt) 