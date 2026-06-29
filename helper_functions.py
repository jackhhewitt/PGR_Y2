import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
import torch._dynamo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric import loader

from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize


def mol_to_morgan_fp(smiles, nBits=2048, radius=3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_fps_and_labels(df, smiles_col="Structure", label_col="Class", nBits=2048):
    fps = []
    labels = []
    for smi, lbl in zip(df[smiles_col], df[label_col]):
        fp = mol_to_morgan_fp(smi, nBits=nBits)
        if fp is not None:
            fps.append(fp)
            labels.append(int(lbl))
    if not fps:
        raise ValueError("No valid fingerprints computed — check SMILES or column names.")
    return np.asarray(fps, dtype=np.float32), np.asarray(labels, dtype=np.float32)

def average_losses_across_folds(fold_losses):
    max_len = max(len(f) for f in fold_losses)
    arr = np.array([np.pad(f, (0, max_len - len(f)), constant_values=np.nan) for f in fold_losses])
    return np.nanmean(arr, axis=0)

def pad_to_same_length(list_of_lists, pad_value=np.nan):
    """Pad shorter lists with NaN to make all equal length."""
    max_len = max(len(lst) for lst in list_of_lists)
    return np.array([np.pad(lst, (0, max_len - len(lst)), constant_values=pad_value)
                     for lst in list_of_lists])

def average_folds(losses):
    """Average across folds (each trial has multiple folds)."""
    padded = pad_to_same_length([np.array(f) for f in losses if len(f) > 0])
    return np.nanmean(padded, axis=0)

def enrichment_factor(y_true, y_prob, top_percent=0.01):
    n_samples = len(y_true)
    n_top = max(1, int(n_samples * top_percent))
    idx_sorted = np.argsort(-y_prob)
    y_top = y_true[idx_sorted[:n_top]]
    n_actives_total = y_true.sum()
    ef = (y_top.sum() / n_top) / (n_actives_total / n_samples)
    return ef
        
def build_hybrid_graph_dataset(df, descriptor_cols, smiles_col="Structure", label_col="Class"):
    """
    Build a list of PyG Data objects with both graph (from SMILES) and descriptor features.
    
    Args:
        df (pd.DataFrame): dataframe containing SMILES, labels, and descriptors
        descriptor_cols (list): list of column names to use as descriptors
        smiles_col (str): column name for SMILES strings
        label_col (str): column name for class labels

    Returns:
        List[torch_geometric.data.Data]: list of Data objects with x, edge_index, y, descriptors
    """
    data_list = []

    for _, row in df.iterrows():
        # Convert SMILES to graph
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue  # skip invalid SMILES

        # Node features
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append([
                atom.GetAtomicNum(), 
                atom.GetDegree(),
                atom.GetFormalCharge(),
                float(atom.GetHybridization()), 
                float(atom.GetIsAromatic())
            ])
        if len(atom_feats) == 0:
            continue  # skip empty molecules

        x = torch.tensor(atom_feats, dtype=torch.float)

        # Edge index
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Label
        y = torch.tensor([float(row[label_col])], dtype=torch.float)

        # Descriptors
        desc_values = row[descriptor_cols].values.astype(float)  # ensure float
        desc = torch.tensor(desc_values, dtype=torch.float)
        if len(desc.shape) == 1:
            desc = desc.unsqueeze(0)  # shape [1, desc_dim] for single graph

        # Build Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.descriptors = desc  # attach descriptor tensor
        data_list.append(data)

    return data_list

def mol_to_graph(smiles, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic()
        ])
    x = torch.tensor(atom_feats, dtype=torch.float)
    
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i,j])
        edge_index.append([j,i])
    if len(edge_index) == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    y_tensor = torch.tensor([int(y)], dtype=torch.float) if y is not None else None
    return Data(x=x, edge_index=edge_index, y=y_tensor)

def build_dataset(df, smiles_col='Structure', label_col='Class'):
    data_list = []
    for _, row in df.iterrows():
        d = mol_to_graph(row[smiles_col], row[label_col])
        if d is not None:
            data_list.append(d)
    return data_list

def enable_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def mc_dropout_mlp(model, X_tensor, n_mc=100):
    
    model.eval()
    enable_dropout(model)

    preds = []

    with torch.no_grad():
        for _ in range(n_mc):
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds.append(probs.cpu().numpy())

    preds = np.stack(preds, axis=0)

    return {
        "mean": preds.mean(axis=0),
        "std": preds.std(axis=0),
        "all": preds
    }

def calibrate_temperature(logits, labels):
    """
    Finds the optimal temperature T by minimizing Negative Log Likelihood 
    on the validation set.
    """
    # Ensure inputs are tensors
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    
    def objective(t):
        t = t[0]
        # Temperature scaling for binary classification
        # We scale the logits before the sigmoid
        scaled_probs = torch.sigmoid(logits / t)
        # Calculate Binary Cross Entropy (NLL for binary)
        return F.binary_cross_entropy(scaled_probs, labels).item()

    # Optimize T starting at 1.0 (bounds prevent T from being 0 or too large)
    res = minimize(objective, x0=[1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    return res.x[0]

