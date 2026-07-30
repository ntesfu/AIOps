"""CaptainCook4D error-recognition screen (architecture-validation experiment).

Loads per-segment VideoMAEv2 features and runs, under a participant-disjoint
(person) split, three detectors that isolate *why* IndustReal error detection
failed:

* ``linear`` / ``mlp`` — supervised normal-vs-error classifiers.  If these
  generalize to held-out persons, IndustReal's failure was data scarcity
  (19 errors), not the architecture.
* ``normal_proto`` — the StateVerify "expected-normal" idea: per step-type
  centroid of *correct-only* training segments; error score = cosine distance
  from the expected-normal feature.  Tests learn-normal-flag-deviation when
  normal data is abundant.

Reports threshold-free AUC/AP (primary) plus F1/precision/recall at a
validation-tuned threshold, and per-error-category recall.  Reference band:
CaptainCook4D paper's best vision-only Omnivore baseline ~55 F1 / ~65-75 AUC.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from aiops.data.captaincook4d import ERROR_CATEGORIES, load_recording_split


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks."""
    y = np.asarray(y)
    pos_n, neg_n = int((y == 1).sum()), int((y == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts  # 0-based start of each tie group (sorted)
    avg_rank = start + (counts + 1) / 2.0  # mean 1-based rank within group
    ranks = avg_rank[inv]
    return float((ranks[y == 1].sum() - pos_n * (pos_n + 1) / 2) / (pos_n * neg_n))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec_hits = y == 1
    if y.sum() == 0:
        return float("nan")
    return float(prec[rec_hits].sum() / y.sum())


def prf_at(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = np.unique(s)
    if len(cands) > 512:
        cands = np.quantile(s, np.linspace(0, 1, 512))
    best_f1, best_t = -1.0, 0.5
    for t in cands:
        _, _, f1 = prf_at(y, s, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def load_features(features_dir: str) -> dict[str, np.ndarray]:
    files = [f for f in sorted(glob.glob(f"{features_dir}/*.npz"))]
    keys = ["seg_mean", "labels", "categories", "step_id",
            "recording_id", "person_id", "activity_id"]
    acc: dict[str, list] = {k: [] for k in keys}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in keys:
            acc[k].append(d[k])
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def train_torch_head(Xtr, ytr, Xva, yva, hidden, epochs=200, lr=1e-3, wd=1e-4, seed=7):
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
    d = Xtr.shape[1]
    if hidden > 0:
        net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 1),
        ).to(dev)
    else:
        net = torch.nn.Linear(d, 1).to(dev)
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                         dtype=torch.float32, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        out = net(Xtr_t).squeeze(-1)
        loss_fn(out, ytr_t).backward()
        opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                va = torch.sigmoid(net(Xva_t).squeeze(-1)).cpu().numpy()
            a = roc_auc(yva, va)
            if a > best_auc:
                best_auc = a
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state:
        net.load_state_dict(best_state)
    net.eval()

    def score(X):
        with torch.no_grad():
            return torch.sigmoid(
                net(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(-1)
            ).cpu().numpy()

    return score


def normal_proto_scores(Xtr, ytr, steptr, Xte, stepte):
    """Cosine distance from the correct-only centroid of the same step-type."""
    def unit(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    Xtr_u, Xte_u = unit(Xtr), unit(Xte)
    normal = ytr == 0
    global_proto = Xtr_u[normal].mean(0)
    global_proto /= np.linalg.norm(global_proto) + 1e-8
    protos = {}
    for st in np.unique(steptr):
        m = normal & (steptr == st)
        if m.sum() >= 5:
            p = Xtr_u[m].mean(0)
            protos[st] = p / (np.linalg.norm(p) + 1e-8)
    scores = np.empty(len(Xte))
    for i in range(len(Xte)):
        p = protos.get(stepte[i], global_proto)
        scores[i] = 1.0 - float(Xte_u[i] @ p)  # higher = more anomalous
    return scores


def category_recall(y, s, thr, cats, cat_names):
    pred = (s >= thr).astype(int)
    out = {}
    for j, name in enumerate(cat_names):
        idx = cats[:, j] == 1
        if idx.sum() == 0:
            continue
        out[name] = round(float(pred[idx].mean()), 3), int(idx.sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--split-by", default="person")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    D = load_features(args.features_dir)
    X = D["seg_mean"].astype(np.float32)
    y = D["labels"].astype(int)
    rec = D["recording_id"].astype(str)
    step = D["step_id"].astype(int)
    cats = D["categories"].astype(int)
    split = load_recording_split(args.annotations_root, args.split_by)
    have = set(rec)
    tr_ids = split["train"] & have
    va_ids = split["val"] & have
    te_ids = split["test"] & have
    tr = np.isin(rec, list(tr_ids))
    va = np.isin(rec, list(va_ids))
    te = np.isin(rec, list(te_ids))
    print(f"loaded {len(y)} segs from {len(have)} recordings "
          f"({rec.shape[0]} rows); split={args.split_by}")
    print(f"  train segs={tr.sum()} err={y[tr].sum()} | "
          f"val segs={va.sum()} err={y[va].sum()} | "
          f"test segs={te.sum()} err={y[te].sum()}")

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xz = (X - mu) / sd

    results = {}
    models = {
        "linear": lambda: train_torch_head(Xz[tr], y[tr], Xz[va], y[va], 0, seed=args.seed),
        "mlp": lambda: train_torch_head(Xz[tr], y[tr], Xz[va], y[va], 512, seed=args.seed),
    }
    for name, make in models.items():
        sc = make()
        s_va, s_te = sc(Xz[va]), sc(Xz[te])
        thr = best_threshold(y[va], s_va)
        pr, rc, f1 = prf_at(y[te], s_te, thr)
        results[name] = {
            "test_auc": round(roc_auc(y[te], s_te), 4),
            "test_ap": round(average_precision(y[te], s_te), 4),
            "test_f1": round(f1, 4), "test_precision": round(pr, 4),
            "test_recall": round(rc, 4), "val_thr": round(float(thr), 4),
            "category_recall": category_recall(y[te], s_te, thr, cats[te], ERROR_CATEGORIES),
        }

    # normal-prototype (expected-normal) detector
    s_te = normal_proto_scores(Xz[tr], y[tr], step[tr], Xz[te], step[te])
    s_va = normal_proto_scores(Xz[tr], y[tr], step[tr], Xz[va], step[va])
    thr = best_threshold(y[va], s_va)
    pr, rc, f1 = prf_at(y[te], s_te, thr)
    results["normal_proto"] = {
        "test_auc": round(roc_auc(y[te], s_te), 4),
        "test_ap": round(average_precision(y[te], s_te), 4),
        "test_f1": round(f1, 4), "test_precision": round(pr, 4),
        "test_recall": round(rc, 4),
        "category_recall": category_recall(y[te], s_te, thr, cats[te], ERROR_CATEGORIES),
    }
    results["_meta"] = {
        "prevalence_test": round(float(y[te].mean()), 4),
        "n_recordings": len(have), "feature_dim": int(X.shape[1]),
        "reference_omnivore_paper": {"step_split_f1": 55.4, "step_split_auc": 75.7,
                                     "recording_split_f1": 56.0, "recording_split_auc": 65.3},
    }
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
