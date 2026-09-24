import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import json, time

SEEDS = [42, 123, 2024, 7, 999]
EPSILONS = [0.01, 0.05, 0.10, 0.15, 0.20]
EPOCHS = 15
BATCH_SIZE = 512
LR = 1e-3
ALPHA_RATIO = 0.02 / 0.15  # alpha = (2/15)*epsilon; alpha=0.02 when epsilon=0.15
PGD_ITERS = 10

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]
train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df = pd.read_csv("KDDTest+.txt", names=columns)
train_df.drop(columns=["difficulty"], inplace=True)
test_df.drop(columns=["difficulty"], inplace=True)
train_df["binary_label"] = (train_df["label"] != "normal").astype(int)
test_df["binary_label"] = (test_df["label"] != "normal").astype(int)

categorical_cols = ["protocol_type", "service", "flag"]
# Fit the categorical encoder on TRAIN only. Unknown test categories are
# mapped to -1 rather than using information from the test set.
cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_df[categorical_cols] = cat_encoder.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = cat_encoder.transform(test_df[categorical_cols])

feature_cols = [c for c in train_df.columns if c not in ["label", "binary_label"]]
# Identify categorical column indices in the feature vector (to FREEZE during perturbation)
CAT_INDICES = [feature_cols.index(c) for c in categorical_cols]
print("Indices des variables catégorielles (gelées pendant FGSM/PGD) :", CAT_INDICES, "sur", len(feature_cols), "caractéristiques")

X_train_raw = train_df[feature_cols].values.astype(np.float32)
y_train_raw = train_df["binary_label"].values.astype(np.int64)
X_test_raw = test_df[feature_cols].values.astype(np.float32)
y_test_raw = test_df["binary_label"].values.astype(np.int64)

scaler = StandardScaler()
X_train_raw = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test_raw = scaler.transform(X_test_raw).astype(np.float32)
n_features = X_train_raw.shape[1]

# Build perturbation mask: 1.0 for continuous features, 0.0 for categorical (frozen)
PERT_MASK = torch.ones(n_features)
for idx in CAT_INDICES:
    PERT_MASK[idx] = 0.0


class DNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)


def fgsm_attack(model, X, y, eps):
    X_adv = X.clone().detach().requires_grad_(True)
    out = model(X_adv)
    loss = nn.CrossEntropyLoss()(out, y)
    grad = torch.autograd.grad(loss, X_adv)[0]
    perturbation = eps * grad.sign() * PERT_MASK  # freeze categorical columns
    return (X_adv.detach() + perturbation).detach()


def pgd_attack(model, X, y, eps, alpha, iters):
    X_orig = X.clone().detach()
    X_adv = X.clone().detach()
    for _ in range(iters):
        X_adv.requires_grad_(True)
        out = model(X_adv)
        loss = nn.CrossEntropyLoss()(out, y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        step = alpha * grad.sign() * PERT_MASK  # freeze categorical columns
        X_adv = X_adv.detach() + step
        perturbation = torch.clamp(X_adv - X_orig, min=-eps, max=eps) * PERT_MASK
        X_adv = (X_orig + perturbation).detach()
    return X_adv


def train_model(model, X, y, epochs, lr, adv_training=False, eps=0.05):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            if adv_training:
                xb_adv = fgsm_attack(model, xb, yb, eps)
                xb = torch.cat([xb, xb_adv], dim=0)
                yb = torch.cat([yb, yb], dim=0)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    acc = (preds == y).float().mean().item()
    tp = ((preds == 1) & (y == 1)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    model.train()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


t0 = time.time()
# ============================================================
# PHASE 1 : baseline models trained once per seed (eps-independent training)
# then evaluated (clean + FGSM + PGD) at EACH epsilon value
# ============================================================
baseline_results = {eps: {"clean": [], "fgsm": [], "pgd": []} for eps in EPSILONS}
raw_records = []

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    X_train_t = torch.tensor(X_train_raw); y_train_t = torch.tensor(y_train_raw)
    X_test_t = torch.tensor(X_test_raw); y_test_t = torch.tensor(y_test_raw)
    model = train_model(DNN(n_features), X_train_t, y_train_t, EPOCHS, LR)
    clean_res = evaluate(model, X_test_t, y_test_t)
    for eps in EPSILONS:
        alpha = ALPHA_RATIO * eps
        fgsm_res = evaluate(model, fgsm_attack(model, X_test_t, y_test_t, eps), y_test_t)
        pgd_res = evaluate(model, pgd_attack(model, X_test_t, y_test_t, eps, alpha, PGD_ITERS), y_test_t)
        baseline_results[eps]["clean"].append(clean_res)
        baseline_results[eps]["fgsm"].append(fgsm_res)
        baseline_results[eps]["pgd"].append(pgd_res)
        raw_records.extend([
            {"seed": seed, "epsilon": eps, "configuration": "baseline", "attack": "clean", **clean_res},
            {"seed": seed, "epsilon": eps, "configuration": "baseline", "attack": "fgsm", **fgsm_res},
            {"seed": seed, "epsilon": eps, "configuration": "baseline", "attack": "pgd", **pgd_res},
        ])
    print(f"[baseline] seed={seed} clean_acc={clean_res['accuracy']:.4f}  ({time.time()-t0:.0f}s elapsed)")

# ============================================================
# PHASE 2 : defended models — one training PER (seed, epsilon) combination
# ============================================================
defended_results = {eps: {"clean": [], "fgsm": [], "pgd": []} for eps in EPSILONS}

for eps in EPSILONS:
    alpha = ALPHA_RATIO * eps
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        X_train_t = torch.tensor(X_train_raw); y_train_t = torch.tensor(y_train_raw)
        X_test_t = torch.tensor(X_test_raw); y_test_t = torch.tensor(y_test_raw)
        model = train_model(DNN(n_features), X_train_t, y_train_t, EPOCHS, LR, adv_training=True, eps=eps)
        clean_res = evaluate(model, X_test_t, y_test_t)
        fgsm_res = evaluate(model, fgsm_attack(model, X_test_t, y_test_t, eps), y_test_t)
        pgd_res = evaluate(model, pgd_attack(model, X_test_t, y_test_t, eps, alpha, PGD_ITERS), y_test_t)
        defended_results[eps]["clean"].append(clean_res)
        defended_results[eps]["fgsm"].append(fgsm_res)
        defended_results[eps]["pgd"].append(pgd_res)
        raw_records.extend([
            {"seed": seed, "epsilon": eps, "configuration": "defended", "attack": "clean", **clean_res},
            {"seed": seed, "epsilon": eps, "configuration": "defended", "attack": "fgsm", **fgsm_res},
            {"seed": seed, "epsilon": eps, "configuration": "defended", "attack": "pgd", **pgd_res},
        ])
        print(f"[defended] eps={eps} seed={seed} clean_acc={clean_res['accuracy']:.4f} fgsm_acc={fgsm_res['accuracy']:.4f}  ({time.time()-t0:.0f}s elapsed)")

# ============================================================
# Aggregate : mean +/- std for every (eps, config, metric)
# ============================================================
def agg(results_list):
    out = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        vals = np.array([r[metric] for r in results_list])
        out[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    return out

summary = {"epsilons": EPSILONS, "seeds": SEEDS, "categorical_frozen": True, "cat_indices": CAT_INDICES, "results": {}}
for eps in EPSILONS:
    summary["results"][str(eps)] = {
        "baseline_clean": agg(baseline_results[eps]["clean"]),
        "baseline_fgsm": agg(baseline_results[eps]["fgsm"]),
        "baseline_pgd": agg(baseline_results[eps]["pgd"]),
        "defended_clean": agg(defended_results[eps]["clean"]),
        "defended_fgsm": agg(defended_results[eps]["fgsm"]),
        "defended_pgd": agg(defended_results[eps]["pgd"]),
    }

with open("results_sensitivity.json", "w") as f:
    json.dump(summary, f, indent=2)

with open("results_sensitivity_raw.json", "w") as f:
    json.dump(raw_records, f, indent=2)

print(f"\nTerminé en {time.time()-t0:.0f}s. Résumé (eps=0.15) :")
r15 = summary["results"]["0.15"]
for k, v in r15.items():
    print(f"  {k}: acc={v['accuracy']['mean']*100:.2f}% +/- {v['accuracy']['std']*100:.2f}%")
