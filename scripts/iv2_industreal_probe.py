"""InternVideo2-B separability probe on IndustReal completion events.

Directly comparable to the evidence sweep: at each completion event, take an
8-frame trailing clip ending at the event frame, encode with InternVideo2-B,
and measure operator-disjoint leave-one-operator-out AUC (correct vs incorrect),
plus fusion with the cached Swin motion feature.
"""
import sys
sys.path.insert(0, "/media/lm-ciss/LM_4TB/aiops/AIOps-stategraph-industreal/scripts")
import numpy as np
from pathlib import Path
from aiops.data.stategraph_cache import read_cache_index
from internvideo2_b import InternVideo2Encoder

DATA_ROOT = "/home/aiops/Desktop/ego_psr_repro/industreal"
CACHE = "/media/lm-ciss/LM_4TB/aiops/AIOps-stategraph-industreal/data/processed/industreal_stateverify_v2/index.json"
CKPT = ("/home/aiops/.cache/huggingface/hub/models--OpenGVLab--InternVideo2_distillation_models/"
        "snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac/stage1/B14/B14_ft_k710_f8/pytorch_model.bin")


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos) / (len(pos) * len(neg)))


def loo_auc(X, y, ops):
    X = np.asarray(X, float); y = np.asarray(y); ops = np.asarray(ops)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sc = np.full(len(y), np.nan)
    for o in np.unique(ops):
        te = ops == o; tr = ~te
        if y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
            continue
        c1 = Xn[tr][y[tr] == 1].mean(0); c0 = Xn[tr][y[tr] == 0].mean(0)
        c1 /= np.linalg.norm(c1) + 1e-8; c0 /= np.linalg.norm(c0) + 1e-8
        sc[te] = Xn[te] @ c1 - Xn[te] @ c0
    v = ~np.isnan(sc)
    return auc(y[v], sc[v]), int(v.sum()), int(y[v].sum())


def main():
    from decord import VideoReader
    import torch
    enc = InternVideo2Encoder(CKPT, device="cuda", dtype=torch.float32)
    print("IV2 load:", {k: len(v) for k, v in enc.load_report.items()}, flush=True)
    meta, recs = read_cache_index(CACHE)
    feats, mots, labels, ops = [], [], [], []
    for cr in recs:
        with np.load(cr.path) as a:
            out = a["component_outcome"]; pf = a["prediction_frame_indices"]; motion = a["motion"]
        rows, comps = np.where((out == 0) | (out == 1))
        seen = {}
        for r, c in zip(rows, comps):
            seen.setdefault((int(pf[r]), int(out[r, c])), int(r))
        events = sorted((f, l, row) for (f, l), row in seen.items())
        if not events:
            continue
        vpath = Path(DATA_ROOT) / f"{cr.recording_id}.mp4"
        if not vpath.exists():
            print("  MISSING", cr.recording_id, flush=True); continue
        vr = VideoReader(str(vpath)); n = len(vr); op = int(cr.recording_id.split("_")[0])
        clips = []
        for frame, lab, row in events:
            idx = np.clip(frame + (np.arange(8) - 7) * 4, 0, n - 1)
            clips.append([f for f in vr.get_batch(idx.tolist()).asnumpy()])
            mots.append(motion[row]); labels.append(lab); ops.append(op)
        for s in range(0, len(clips), 8):
            feats.append(enc.encode(clips[s:s + 8]))
        print(f"  {cr.recording_id}: {len(events)} events", flush=True)
    feats = np.concatenate(feats, 0); mots = np.asarray(mots); labels = np.asarray(labels); ops = np.asarray(ops)
    np.savez("/tmp/iv2_industreal_feats.npz", feats=feats, motion=mots, labels=labels, ops=ops)
    print(f"\nEVENTS={len(labels)} incorrect={int((labels==1).sum())} ops={len(set(ops))}")

    def unit(X):
        X = np.asarray(X, float); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    fusion = np.concatenate([unit(mots), unit(feats)], axis=1)
    print("\n=== operator-disjoint LOO-AUC (correct vs incorrect completions) ===")
    for name, X in [("InternVideo2-B (8-frame)", feats), ("Swin motion (cache)", mots),
                    ("FUSION motion + IV2", fusion)]:
        a, nn, npos = loo_auc(X, labels, ops)
        print(f"  {name:28}: AUC={a:.3f}   (scored {nn}, {npos} incorrect)")
    print("\nreference sweep: Swin 0.669 | VideoMAE 0.43-0.52 | GroundingDINO 0.64 | SAM3 0.28 | fusions <=0.65")


if __name__ == "__main__":
    main()
