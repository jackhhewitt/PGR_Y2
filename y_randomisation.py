import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
import torch._dynamo
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, matthews_corrcoef
import os
import optuna
import pandas as pd
import random

from architectures import MLPClassifier_2
from loss_functions import FocalLoss
from helper_functions import enrichment_factor, compute_fps_and_labels

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

def hpo(trial, X, y, device):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = 2 ** trial.suggest_int("batch_size_exp", 6, 10)
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
        criterion = FocalLoss(alpha=1-pos_frac, gamma=2.0)
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
            for xb, yb in train_loader:
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

                auc_val = roc_auc_score(y_val_np, val_probs)

            scheduler.step(val_loss)

            step_id = fold_idx * max_epochs + epoch
            trial.report(auc_val, step_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

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

def training(best_params, X, y, device, save_model_path="mlp_final.pt"):

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)

    fold_metrics = {k: [] for k in ["AUROC", "AUPRC", "MCC", "EF1", "EF5", "EF10"]}
    all_train_losses = []
    all_val_losses = []
    roc_thresholds = []

    batch_size = 2 ** best_params["batch_size_exp"]
    max_epochs = 100
    patience = 5

    # 5x5 CV
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        model = MLPClassifier_2(
            input_size=X_train.shape[1],
            hidden_size=best_params["hidden_size"],
            dropout=best_params["dropout"]).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"])

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True)

        pos_frac = y_train_t.mean().item()
        criterion = FocalLoss(alpha=1 - pos_frac, gamma=2.0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5)

        wait = 0
        best_auc = -1.0
        best_state = None

        fold_train_losses = []
        fold_val_losses = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            fold_train_losses.append(total_loss / len(train_loader.dataset))

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_probs = torch.sigmoid(val_logits).squeeze().cpu().numpy()
                val_labels = y_val_t.squeeze().cpu().numpy()
                val_loss = criterion(val_logits, y_val_t).item()

            fold_val_losses.append(val_loss)
            scheduler.step(val_loss)

            auc_val = roc_auc_score(val_labels, val_probs)

            if auc_val > best_auc:
                best_auc = auc_val
                best_probs = val_probs.copy()
                best_labels = val_labels.copy()
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)

        model.load_state_dict(best_state)

        # ROC-optimal threshold
        fpr, tpr, thresh = roc_curve(best_labels, best_probs)
        thr = thresh[np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))]
        roc_thresholds.append(thr)

        y_pred = (best_probs >= thr).astype(int)

        fold_metrics["AUROC"].append(roc_auc_score(best_labels, best_probs))
        fold_metrics["AUPRC"].append(average_precision_score(best_labels, best_probs))
        fold_metrics["MCC"].append(matthews_corrcoef(best_labels, y_pred))
        fold_metrics["EF1"].append(enrichment_factor(best_labels, best_probs, 0.01))
        fold_metrics["EF5"].append(enrichment_factor(best_labels, best_probs, 0.05))
        fold_metrics["EF10"].append(enrichment_factor(best_labels, best_probs, 0.10))

        print(
            f"Fold {fold_idx:02d}/25 | "
            f"AUROC={fold_metrics['AUROC'][-1]:.4f} | "
            f"AUPRC={fold_metrics['AUPRC'][-1]:.4f} | "
            f"MCC={fold_metrics['MCC'][-1]:.4f}")

    # final training on full data
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    final_model = MLPClassifier_2(
        input_size=X.shape[1],
        hidden_size=best_params["hidden_size"],
        dropout=best_params["dropout"]).to(device)

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"])

    criterion = FocalLoss(alpha=1 - y_tensor.mean().item(), gamma=2.0)

    train_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True)

    for epoch in range(max_epochs):
        final_model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            optimizer.step()

    # mean threshold across CV folds
    final_threshold = float(np.mean(roc_thresholds))

    torch.save(
        {
            "model_state": final_model.state_dict(),
            "params": best_params,
            "threshold_roc": final_threshold,
            "fold_metrics": fold_metrics,
        },
        save_model_path
    )

    print("\nFinal MLP model saved to:", os.path.abspath(save_model_path))
    print(f"Mean ROC-optimal threshold: {final_threshold:.4f}")

    return fold_metrics, final_model, final_threshold, all_train_losses, all_val_losses

def evaluate(model, threshold_roc, X_test, y_test, device):

    model.eval()
    y_test_np = np.asarray(y_test)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs_test = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

    y_pred_test = (probs_test >= threshold_roc).astype(int)

    # Compute all metrics
    metrics = {
        "AUROC": roc_auc_score(y_test_np, probs_test),
        "AUPRC": average_precision_score(y_test_np, probs_test),
        "MCC":   matthews_corrcoef(y_test_np, y_pred_test),
        "EF1":   enrichment_factor(y_test_np, probs_test, 0.01),
        "EF5":   enrichment_factor(y_test_np, probs_test, 0.05),
        "EF10":  enrichment_factor(y_test_np, probs_test, 0.10),
        "probs_test" : probs_test}
    
    return metrics

class YRandomizationManager:
    
    def __init__(self, hpo_func, train_func, eval_func, device):

        self.hpo_func = hpo_func
        self.train_func = train_func
        self.eval_func = eval_func
        self.device = device
        self.summary_rows = []
        self.detailed_rows = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.all_tprs = []

    def run_study(self, X_train, y_train, X_test, y_test, iterations=50):

        for i in range(1, iterations + 1):  # Start at 1 to skip baseline
            # Shuffle labels for this iteration
            y_shuffled = y_train.copy()
            np.random.shuffle(y_shuffled)

            print(f"\n>>> Starting Y-Randomization Run {i}/{iterations}...")

            # 1. Hyperparameter optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: self.hpo_func(t, X_train, y_shuffled, self.device), n_trials=100)
            
            # 2. Training (5x5 CV + final model training)
            # Returns: fold_metrics, final_model, threshold, train_losses, val_losses
            cv_metrics, model, threshold, train_losses, val_losses = self.train_func(
                study.best_params, X_train, y_shuffled, self.device)
            
            # 3. Evaluation (Test Set)
            test_metrics = self.eval_func(model, threshold, X_test, y_test, self.device)

            fpr, tpr, _ = roc_curve(y_test, test_metrics['probs_test'])

            interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Ensure it starts at 0
            self.all_tprs.append(interp_tpr)

            # File 1: Summary Metrics (1 row per run)
            summary_entry = {
                "iteration": i,
                "test_auroc": test_metrics["AUROC"],
                "test_mcc": test_metrics["MCC"],
                "test_auprc": test_metrics["AUPRC"],
                "test_ef1": test_metrics["EF1"],
                "test_ef5": test_metrics["EF5"],
                "test_ef10": test_metrics["EF10"],
                "cv_mean_auroc": np.mean(cv_metrics["AUROC"]),
                "cv_mean_mcc": np.mean(cv_metrics["MCC"]),
                "cv_mean_auprc": np.mean(cv_metrics["AUPRC"]),
                "optuna_best_params": str(study.best_params)}
            
            self.summary_rows.append(summary_entry)
            pd.DataFrame(self.summary_rows).to_csv("y_rand_summary_metrics.csv", index=False)

            # File 2: Detailed Logs (Folds and Losses)
            for fold_idx in range(len(cv_metrics["AUROC"])):
                detail_entry = {
                    "iteration": i,
                    "fold_id": fold_idx + 1,
                    "fold_auroc": cv_metrics["AUROC"][fold_idx],
                    "fold_mcc": cv_metrics["MCC"][fold_idx],
                    # Mean loss across all epochs for this specific fold
                    "avg_train_loss": np.mean(train_losses[fold_idx]),
                    "avg_val_loss": np.mean(val_losses[fold_idx])
                }
                self.detailed_rows.append(detail_entry)
            
            pd.DataFrame(self.detailed_rows).to_csv("y_rand_detailed_logs.csv", index=False)
            
            print(f"Run {i} logged successfully.")

        return "Study complete. 50 iterations recorded"
    

# ==============
# EXECUTION CODE
# ==============

train = pd.read_csv(r"./data/train_data.csv")
test = pd.read_csv(r"./data/test_data.csv")

train_sub = train.copy()

X_all, y_all = compute_fps_and_labels(train_sub)
X_test, y_test = compute_fps_and_labels(test)

manager = YRandomizationManager(hpo_func=hpo,
                                train_func=training,
                                eval_func=evaluate,
                                device=device)

print(f'Starting Y-rand study (50 runs)')
status = manager.run_study(X_all, y_all, X_test, y_test, iterations=50)
print(status)