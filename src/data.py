import pandas as pd
from typing import List


def load_smiles_from_csv(csv_path: str, smiles_column: str = 'smiles') -> List[str]:
    """
    Load SMILES strings from a CSV file.
    Args:
        csv_path (str): Path to the CSV file.
        smiles_column (str): Name of the column containing SMILES strings.
    Returns:
        List[str]: List of SMILES strings.
    """
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in CSV.")
    return df[smiles_column].dropna().astype(str).tolist()


def example_usage():
    # Example usage (replace 'data/molecules.csv' with your file)
    smiles_list = load_smiles_from_csv('data/molecules.csv')
    print(smiles_list[:5]) 