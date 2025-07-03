import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from typing import List
import random

# Character set for SMILES (simplified for demo)
SMILES_CHARS = list("#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXZ[\\]abcdefghilmnoprstuy")
CHAR_TO_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(SMILES_CHARS)}
MAX_LEN = 32  # Max SMILES length for demo


def smiles_to_tensor(smiles: str) -> torch.Tensor:
    arr = [CHAR_TO_IDX.get(c, 0) for c in smiles[:MAX_LEN]]
    arr += [0] * (MAX_LEN - len(arr))
    return torch.tensor(arr, dtype=torch.long)


def tensor_to_smiles(tensor: torch.Tensor) -> str:
    chars = [IDX_TO_CHAR.get(int(i), '') for i in tensor]
    return ''.join(chars).strip()


class SmilesVAE(nn.Module):
    def __init__(self, vocab_size=len(SMILES_CHARS), emb_dim=32, latent_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, 64)  # Added for decoder init
        self.decoder = nn.LSTM(emb_dim, 64, batch_first=True)
        self.fc_out = nn.Linear(64, vocab_size)

    def encode(self, x):
        x = self.emb(x)
        _, (h, _) = self.encoder(x)
        h = h[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len=MAX_LEN):
        # Start with zeros as input
        inputs = torch.zeros((z.size(0), seq_len, self.emb.embedding_dim), device=z.device)
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)  # shape: [1, batch, 64]
        c0 = torch.zeros_like(h0)
        outputs, _ = self.decoder(inputs, (h0, c0))
        logits = self.fc_out(outputs)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x.size(1))
        return logits, mu, logvar


def sample_smiles(model: SmilesVAE, num_samples=5) -> List[str]:
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.fc_mu.out_features)
        logits = model.decode(z, seq_len=MAX_LEN)
        sampled = torch.argmax(logits, dim=-1)
        smiles_list = [tensor_to_smiles(s) for s in sampled]
        # Validate with RDKit
        valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s)]
        return valid_smiles


def example_usage():
    model = SmilesVAE()
    samples = sample_smiles(model, num_samples=3)
    print("Sampled valid SMILES:", samples) 