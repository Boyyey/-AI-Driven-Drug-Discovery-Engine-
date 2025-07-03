import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
import os

st.set_page_config(page_title="Drug Discovery Engine", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
st.sidebar.title("🧬 Drug Discovery Engine")
st.sidebar.markdown("---")

uploaded_csv = st.sidebar.file_uploader("Upload Molecule CSV", type=["csv"])
uploaded_protein = st.sidebar.file_uploader("Upload Protein (PDBQT)", type=["pdbqt"])

st.sidebar.markdown("---")
alpha = st.sidebar.slider("GNN vs Docking Weight (alpha)", 0.0, 1.0, 0.6, 0.05)
run_pipeline = st.sidebar.button("🚀 Run Pipeline")

# --- Helper functions ---
def mol_to_image(mol):
    img = Draw.MolToImage(mol, size=(150, 150), kekulize=True)
    buf = BytesIO()
    img.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"<img src='data:image/png;base64,{data}' style='background: #222; border-radius: 8px;'/>"

def get_protein_path(uploaded_protein):
    if uploaded_protein is not None:
        protein_path = os.path.join('data', uploaded_protein.name)
        with open(protein_path, 'wb') as f:
            f.write(uploaded_protein.read())
        return protein_path
    else:
        return 'data/protein.pdbqt'

# --- Main Area ---
st.title("AI-Driven Drug Discovery Engine")
st.markdown("""
A modern, AI-powered platform for predicting and ranking drug candidates using molecular graphs, GNNs, generative models, and docking simulations.
""")

if 'results' not in st.session_state:
    st.session_state['results'] = None

if run_pipeline and uploaded_csv is not None:
    # --- Load molecules ---
    df = pd.read_csv(uploaded_csv)
    smiles_list = df['smiles'].dropna().astype(str).tolist()
    from src.gnn import SmilesDataset, GCN
    import torch
    # --- GNN predictions ---
    dataset = SmilesDataset(smiles_list)
    data = [dataset.get(i) for i in range(len(smiles_list))]
    model = GCN(num_node_features=data[0].x.shape[1])
    gnn_scores = []
    for d in data:
        out = model(d.x, d.edge_index, torch.zeros(d.x.shape[0], dtype=torch.long))
        gnn_scores.append(float(out.item()))
    # --- Generative model (sample, not train) ---
    from src.generative import SmilesVAE, sample_smiles
    vae = SmilesVAE()
    generated = sample_smiles(vae, num_samples=3)
    # --- Docking (mock, for demo) ---
    docking_scores = list(np.random.uniform(-9, -6, len(smiles_list)))
    # --- Ranking ---
    from src.scoring import rank_molecules
    ranked = rank_molecules(smiles_list, gnn_scores, docking_scores, alpha=alpha)
    # --- Store results ---
    st.session_state['results'] = {
        'smiles_list': smiles_list,
        'gnn_scores': gnn_scores,
        'generated': generated,
        'docking_scores': docking_scores,
        'ranked': ranked
    }

results = st.session_state['results']

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loaded Molecules")
    if results:
        for smi in results['smiles_list']:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                st.markdown(mol_to_image(mol), unsafe_allow_html=True)
                st.caption(smi)
    else:
        st.info("Upload a CSV to view molecules.")

with col2:
    st.subheader("GNN Predictions")
    if results:
        st.dataframe(pd.DataFrame({
            'SMILES': results['smiles_list'],
            'GNN Score': results['gnn_scores']
        }))
    else:
        st.info("GNN property predictions will appear here.")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Generated Molecules")
    if results:
        for smi in results['generated']:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                st.markdown(mol_to_image(mol), unsafe_allow_html=True)
                st.caption(smi)
    else:
        st.info("Generated molecules will appear here.")

with col4:
    st.subheader("Docking Scores")
    if results:
        st.dataframe(pd.DataFrame({
            'SMILES': results['smiles_list'],
            'Docking Score': results['docking_scores']
        }))
    else:
        st.info("Docking scores will appear here.")

st.markdown("---")

st.subheader("Ranked Results")
if results:
    st.dataframe(pd.DataFrame(results['ranked'], columns=['SMILES', 'Combined Score']))
else:
    st.info("Run the pipeline to see ranked molecules.")

# --- Dark Theme Note ---
st.markdown("<style>body { background-color: #18191A; color: #E4E6EB; }</style>", unsafe_allow_html=True) 