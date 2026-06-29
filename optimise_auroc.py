import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

import torch._dynamo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem.Scaffolds import MurckoScaffold

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, matthews_corrcoef

from collections import Counter
import os
import random
import optuna
import torch.nn.functional as F
from helper_functions import compute_fps_and_labels, enrichment_factor
from architectures import MLPClassifier_2
from loss_functions import FocalLoss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def optimise_auroc(trial, X, y):

    # Hyperparameters
    nbits = 2048
    hidden_size = trial.suggest_int("hidden_size", 64, 1024, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.8)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = 2 ** trial.suggest_int("batch_size_exp", 4, 9)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    max_epochs = 100
    patience = 5

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    fold_metrics = {k: [] for k in ["AUROC", "AUPRC", "MCC", "EF1", "EF5", "EF10"]}

    all_train_losses = []
    all_val_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        # Model, optimizer, loss
        input_size = X_train_tensor.shape[1]
        model = MLPClassifier_2(input_size=input_size, hidden_size=hidden_size, dropout=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        pos_frac = y_train_tensor.mean().item()
        criterion = FocalLoss(alpha=pos_frac, gamma=2.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop with early stopping
        best_auc = -1.0
        wait = 0
        fold_train_losses = []
        fold_val_losses = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:     # loss computation and backpropagation
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            fold_train_losses.append(total_loss / len(train_dataset))

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_probs = torch.sigmoid(val_outputs).squeeze(-1).cpu().numpy()
                val_loss = criterion(val_outputs, y_val_tensor).item()
                fold_val_losses.append(val_loss)
                y_val_np = y_val_tensor.squeeze(-1).cpu().numpy()

                auc_val = roc_auc_score(y_val_np, val_probs)  # validation AUROC for pruning

            scheduler.step(val_loss)

            step_id = fold_idx * max_epochs + epoch
            trial.report(auc_val, step_id)
            if trial.should_prune():
                raise optuna.TrialPruned()     # early stopping/pruning based on AUROC improvement

            if auc_val > best_auc:
                best_auc = auc_val
                best_model_state = model.state_dict()
                best_probs_val = val_probs.copy()
                best_y_val = y_val_np.copy()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)

        # ROC-optimal threshold
        fpr, tpr, thresh = roc_curve(best_y_val, best_probs_val)
        optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
        threshold_roc = thresh[optimal_idx]
        y_pred = (best_probs_val >= threshold_roc).astype(int)

        # Metrics
        fold_metrics["AUROC"].append(roc_auc_score(best_y_val, best_probs_val))
        fold_metrics["AUPRC"].append(average_precision_score(best_y_val, best_probs_val))
        fold_metrics["MCC"].append(matthews_corrcoef(best_y_val, y_pred))
        fold_metrics["EF1"].append(enrichment_factor(best_y_val, best_probs_val, 0.01))
        fold_metrics["EF5"].append(enrichment_factor(best_y_val, best_probs_val, 0.05))
        fold_metrics["EF10"].append(enrichment_factor(best_y_val, best_probs_val, 0.10))

    # Aggregate CV metrics
    mean_metrics = {k: float(np.nanmean(v)) for k, v in fold_metrics.items()}
    std_metrics = {k: float(np.nanstd(v)) for k, v in fold_metrics.items()}

    # Save metrics in trial attributes
    for k, v in fold_metrics.items():
        trial.set_user_attr(f"cv_folds_{k}", v)
        trial.set_user_attr(f"cv_mean_{k}", mean_metrics[k])
        trial.set_user_attr(f"cv_std_{k}", std_metrics[k])

    trial.set_user_attr("all_train_losses", all_train_losses)
    trial.set_user_attr("all_val_losses", all_val_losses)

    # Return threshold-independent metric for Optuna optimization
    return mean_metrics["AUROC"]

# ================
#  EXECUTION CODE
# ================

# Prepare dataset
train_data = pd.read_csv("./data/train_data.csv")  # Replace with actual data path, ensure Smiles column is called 'Structure' and Class column is called 'Class'
                                                   # otherwise change helper function

X, y = compute_fps_and_labels(train_data, nBits=2048)
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

# Run Optuna Study
study = optuna.create_study(direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=25),
    sampler=optuna.samplers.TPESampler(seed=SEED))

N_TRIALS = 100

try:
    study.optimize(optimise_auroc, n_trials=N_TRIALS)
except KeyboardInterrupt:
    print("Optimization interrupted by user.")

# Best Trial
best_trial = study.best_trial
print("\nBest trial:")
print(f"Best MCC: {best_trial.value:.4f}")
print(f"Params: {best_trial.params}")

for metric in ["AUROC", "AUPRC", "MCC", "EF1", "EF5", "EF10"]:
    mean_val = best_trial.user_attrs.get(f"cv_mean_{metric}")
    print(f"mean_{metric}: {mean_val:.4f}")


### for optimising AUROC with the labelswapped data, use FocalLossWithSigmoidMargin and edit the code above to include:

"""    
def optimise_auroc(trial, X, y):

    ...

        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)

        y_true01 = (best_y_val + 1) / 2

        # ROC-optimal threshold
        fpr, tpr, thresh = roc_curve(y_true01, best_probs_val)
        optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
        threshold_roc = thresh[optimal_idx]
        y_pred = (best_probs_val >= threshold_roc).astype(int)

        # Metrics
        fold_metrics["AUROC"].append(roc_auc_score(y_true01, best_probs_val))
        fold_metrics["AUPRC"].append(average_precision_score(y_true01, best_probs_val))
        fold_metrics["MCC"].append(matthews_corrcoef(y_true01, y_pred))
        fold_metrics["EF1"].append(enrichment_factor(y_true01, best_probs_val, 0.01))
        fold_metrics["EF5"].append(enrichment_factor(y_true01, best_probs_val, 0.05))
        fold_metrics["EF10"].append(enrichment_factor(y_true01, best_probs_val, 0.10))
    
    ...

"""

### Ensure all retraining and evaluation blocks include the y_true01 = (value + 1) / 2 conversion