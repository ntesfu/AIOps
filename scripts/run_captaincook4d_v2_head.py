"""CaptainCook4D V2 head: attention-pool transformer over per-clip features.

The paper's stronger head (V2) processes the sequence of sub-segment features per
step with a transformer, instead of mean-pooling (V1). We stored per-clip features,
so this trains a [CLS]+transformer head over the variable-length clip sequence per
segment. Reports held-out AUC/AP/F1 vs the V1 mean-pool linear baseline.
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np
from aiops.data.captaincook4d import ERROR_CATEGORIES, load_recording_split

MAXLEN = 16


def roc_auc(y, s):
    y = np.asarray(y); p = int((y == 1).sum()); n = int((y == 0).sum())
    if p == 0 or n == 0: return float("nan")
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    ar = (np.cumsum(c) - c) + (c + 1) / 2.0
    r = ar[inv]
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); y = np.asarray(y)[o]
    if y.sum() == 0: return float("nan")
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float(prec[y == 1].sum() / y.sum())


def prf(y, s, t):
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def best_thr(y, s):
    cand = np.unique(s)
    if len(cand) > 400: cand = np.quantile(s, np.linspace(0, 1, 400))
    return max(cand, key=lambda t: prf(y, s, t)[2])


def load(features_dir):
    segs = []
    for f in sorted(glob.glob(features_dir + "/*.npz")):
        d = np.load(f, allow_pickle=True)
        cf = d["clip_feats"].astype(np.float32); cc = d["clip_counts"]
        b = np.concatenate([[0], np.cumsum(cc)])
        for i in range(len(cc)):
            segs.append(dict(clips=cf[b[i]:b[i + 1]], label=int(d["labels"][i]),
                             rec=str(d["recording_id"][i]), cat=d["categories"][i]))
    return segs


def pad(seg_list, mu, sd):
    B = len(seg_list)
    X = np.zeros((B, MAXLEN, seg_list[0]["clips"].shape[1]), np.float32)
    M = np.zeros((B, MAXLEN), bool)
    for i, s in enumerate(seg_list):
        c = (s["clips"] - mu) / sd
        L = min(len(c), MAXLEN)
        X[i, :L] = c[:L]; M[i, :L] = True
    return X, M


def train_v2(tr, va, te, seed=7, epochs=120):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    allc = np.concatenate([s["clips"] for s in tr])
    mu, sd = allc.mean(0), allc.std(0) + 1e-6
    Xtr, Mtr = pad(tr, mu, sd); Xva, Mva = pad(va, mu, sd); Xte, Mte = pad(te, mu, sd)
    ytr = np.array([s["label"] for s in tr]); yva = np.array([s["label"] for s in va]); yte = np.array([s["label"] for s in te])

    class Head(nn.Module):
        def __init__(self, d_in=768, d=256, heads=4, layers=2):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls, std=0.02)
            el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.2, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(el, layers)
            self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 1)

        def forward(self, x, m):
            B = x.shape[0]
            h = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], 1)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~m], 1)
            h = self.enc(h, src_key_padding_mask=kpm)
            return self.head(self.norm(h[:, 0])).squeeze(-1)

    net = Head().to(dev)
    pw = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, device=dev); Mtr_t = torch.tensor(Mtr, device=dev); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva, device=dev); Mva_t = torch.tensor(Mva, device=dev)
    Xte_t = torch.tensor(Xte, device=dev); Mte_t = torch.tensor(Mte, device=dev)
    n = len(ytr); bs = 128; best_auc, best = -1, None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xtr_t[idx], Mtr_t[idx])
            lossf(out, ytr_t[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va_s = torch.sigmoid(net(Xva_t, Mva_t)).cpu().numpy()
        a = roc_auc(yva, va_s)
        if a > best_auc: best_auc = a; best = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best); net.eval()
    with torch.no_grad():
        va_s = torch.sigmoid(net(Xva_t, Mva_t)).cpu().numpy()
        te_s = torch.sigmoid(net(Xte_t, Mte_t)).cpu().numpy()
    t = best_thr(yva, va_s); pr, rc, f1 = prf(yte, te_s, t)
    cats = np.stack([s["cat"] for s in te])
    catrec = {}
    pred = (te_s >= t).astype(int)
    for j, nm in enumerate(ERROR_CATEGORIES):
        m = cats[:, j] == 1
        if m.sum(): catrec[nm] = [round(float(pred[m].mean()), 3), int(m.sum())]
    return dict(test_auc=round(roc_auc(yte, te_s), 4), test_ap=round(ap(yte, te_s), 4),
                test_f1=round(f1, 4), test_precision=round(pr, 4), test_recall=round(rc, 4),
                val_auc=round(best_auc, 4), category_recall=catrec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--split-by", default="person")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    segs = load(args.features_dir)
    S = load_recording_split(args.annotations_root, args.split_by)
    have = set(s["rec"] for s in segs)
    tr = [s for s in segs if s["rec"] in (S["train"] & have)]
    va = [s for s in segs if s["rec"] in (S["val"] & have)]
    te = [s for s in segs if s["rec"] in (S["test"] & have)]
    print(f"split={args.split_by} tr={len(tr)} va={len(va)} te={len(te)} "
          f"(te err={sum(s['label'] for s in te)})")
    res = train_v2(tr, va, te, seed=args.seed)
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
