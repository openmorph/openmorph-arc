# SPDX-License-Identifier: MIT-0
"""Score the options of `docs/plans/morph-llm/weights.md` on the common test (CPU, free).

    python eval.py items.jsonl out/

Top-1 per state (the item with the highest score must be a true colour), per question, for: chance, the authored
priors, W-0 (zero-shot logit), W-A (readouts on the hidden states, held out by game), W-B (the held-out LoRA
folds) and every W-C file found (`out/wc_*.jsonl`).

W-A, per held-out game: the layer is chosen by an inner leave-one-game-out ridge on the TRAINING games only; the
hidden state is standardised and projected on the training items' top 16 principal directions; two readouts are
trained on the training games' items: a ridge regression (the reference) and the morph's own grown readout
(`openmorph.nn.ValueHead`, a `Field`, positives oversampled to balance). Held-out items are scored by both.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

K = 16


def top1(items, score):
    """Top-1 accuracy over states: {q: (correct, states)}."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        if it["item_id"] in score:
            by[it["q"]][it["state_id"]].append(it)
    out = {}
    for q, states in by.items():
        ok = sum(max(v, key=lambda it: score[it["item_id"]])["label"] for v in states.values())
        out[q] = (ok, len(states))
    return out


def chance(items):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        by[it["q"]][it["state_id"]].append(it["label"])
    return {q: (sum(sum(v) / len(v) for v in s.values()), len(s)) for q, s in by.items()}


def ridge_fit(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)


def ridge_pred(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def project(Htr, Hte, k=K):
    """Standardise on the training rows, project both on the training rows' top k principal directions. With fewer
    rows than dimensions the directions come from the n x n Gram matrix (the same subspace as the SVD, up to sign,
    which no readout here depends on), about 10x faster."""
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    A, B = (Htr - mu) / sd, (Hte - mu) / sd
    if A.shape[0] < A.shape[1]:
        vals, U = np.linalg.eigh(A @ A.T)
        top = np.argsort(vals)[::-1][:k]
        P = A.T @ (U[:, top] / np.sqrt(np.maximum(vals[top], 1e-12)))
    else:
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        P = vt[:k].T
    Ztr, Zte = A @ P, B @ P
    s = Ztr.std(0) + 1e-6
    return Ztr / s, Zte / s


def logo_ridge(H, y, games, q_idx):
    """Inner leave-one-game-out top-1 proxy (mean held-out correlation sign agreement is too weak; use AUC-ish:
    the share of held-out positives scored above the held-out mean)."""
    gs = sorted(set(games))
    hits = n = 0
    for g in gs:
        tr, te = games != g, games == g
        if te.sum() == 0 or y[tr].std() == 0:
            continue
        Ztr, Zte = project(H[tr], H[te])
        w = ridge_fit(Ztr, y[tr])
        p = ridge_pred(w, Zte)
        pos = y[te] > 0
        if pos.any():
            hits += float((p[pos] > p.mean()).mean())
            n += 1
    return hits / max(1, n)


class RlsHead:
    """The morph's ridge: a `Field` whose readout unit learns by recursive least squares (`Config::readout_rls`,
    lambda 1 = every event weighted equally, penalty 1/P0), one `learn_step` per training item. With `backbone` > 0
    the readout reads a grown random-feature backbone (`grow_backbone`) of the inputs instead of the inputs."""

    def __init__(self, k, prior=100.0, backbone=0, seed=0):
        import json as _json
        from openmorph import Morph

        self.k = k
        self.m = Morph(k)
        self.m.configure(_json.dumps({"readout_rls": 1.0, "rls_prior": float(prior)}), False)
        self.m.rng_seed(seed)
        self.idx = np.arange(k)
        reads = list(self.m.grow_backbone(list(range(k)), backbone, 1.0)) if backbone else list(range(k))
        self.out = self.m.n_cells() if hasattr(self.m, "n_cells") else None
        self.out = max(reads + [k - 1]) + 1
        self.m.add_unit(reads, [0.0] * len(reads), 0.0, self.out)

    def learn(self, z, target):
        self.m.learn_step(self.idx, np.asarray(z, dtype=np.float64), self.out, float(target), 1.0, 256)

    def predict(self, z):
        self.m.stimulate(self.idx, np.asarray(z, dtype=np.float64), 256)
        return float(self.m.cell(self.out))


def make_readouts(spec):
    """name -> (train(Ztr, ytr) -> predict(z)) for the morph readouts compared in step 1."""
    from openmorph.nn import ValueHead

    def valuehead(hidden, epochs):
        def train(Z, y):
            head = ValueHead(Z.shape[1], hidden=hidden, budget=64, seed=0)
            pos, neg = np.where(y > 0)[0], np.where(y < 0)[0]
            rng = np.random.RandomState(0)
            order = np.concatenate([neg, rng.choice(pos, len(neg), replace=True)]) if len(pos) else neg
            for _ in range(epochs):
                rng.shuffle(order)
                for i in order:
                    head.learn(Z[i], 0.9 * y[i], 0.05)
            return head.predict
        return train

    def rls(prior, backbone):
        def train(Z, y):
            head = RlsHead(Z.shape[1], prior=prior, backbone=backbone)
            pos, neg = np.where(y > 0)[0], np.where(y < 0)[0]
            rng = np.random.RandomState(0)
            order = np.concatenate([neg, rng.choice(pos, len(neg), replace=True)]) if len(pos) else neg
            rng.shuffle(order)
            for i in order:
                head.learn(Z[i], 0.9 * y[i])
            return head.predict
        return train

    all_ = {"morph ValueHead h4": valuehead(4, 8), "morph ValueHead h0": valuehead(0, 8),
            "morph RLS linear P0=1": rls(1.0, 0), "morph RLS linear P0=100": rls(100.0, 0),
            "morph RLS backbone64 P0=100": rls(100.0, 64)}
    return {k: v for k, v in all_.items() if k in spec}


def wa(items, H, layers, use_morph=True, readouts=("morph ValueHead h4",)):
    ridge_s, chosen = {}, collections.Counter()
    heads = make_readouts(readouts) if use_morph else {}
    morph_s = {name: {} for name in heads}
    for q in ("self", "goal"):
        idx = np.array([i for i, it in enumerate(items) if it["q"] == q])
        sub = [items[i] for i in idx]
        y = np.array([1.0 if it["label"] else -1.0 for it in sub])
        games = np.array([it["game"] for it in sub])
        for g in sorted(set(games)):
            tr, te = games != g, games == g
            # layer chosen on the training games only
            best = max(range(len(layers)), key=lambda l: logo_ridge(H[idx][tr][:, l, :], y[tr], games[tr], None))
            chosen[(q, layers[best])] += 1
            Ztr, Zte = project(H[idx][tr][:, best, :], H[idx][te][:, best, :])
            w = ridge_fit(Ztr, y[tr])
            for it, v in zip([sub[i] for i in np.where(te)[0]], ridge_pred(w, Zte)):
                ridge_s[it["item_id"]] = float(v)
            for name, train in heads.items():
                predict = train(Ztr, y[tr])
                for it, z in zip([sub[i] for i in np.where(te)[0]], Zte):
                    morph_s[name][it["item_id"]] = predict(z)
    return ridge_s, morph_s, chosen


def load_scores(path):
    return {r["item_id"]: r["score"] for r in map(json.loads, open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--no-morph", action="store_true")
    ap.add_argument("--readouts", default="morph ValueHead h4", help="comma-separated morph readouts to compare")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items)]
    rows = []
    fmt = lambda r: f"{100 * r[0] / r[1]:5.1f}% ({r[0]:.0f}/{r[1]})" if r else "     -"
    rows.append(("chance", chance(items)))
    rows.append(("authored prior", top1(items, {it["item_id"]: it["prior"] for it in items})))
    zs_p = os.path.join(args.out, "zs.jsonl")
    if os.path.exists(zs_p):
        rows.append(("W-0 zero-shot 8B (Yes-No logit)", top1(items, load_scores(zs_p))))
    hp = os.path.join(args.out, "hidden.npy")
    if os.path.exists(hp):
        meta = json.load(open(os.path.join(args.out, "hidden_ids.json")))
        pos = {i: k for k, i in enumerate(meta["ids"])}
        H = np.load(hp).astype(np.float32)[[pos[it["item_id"]] for it in items]]
        r, m, chosen = wa(items, H, meta["layers"], use_morph=not args.no_morph, readouts=args.readouts.split(","))
        rows.append(("W-A ridge readout on hidden states", top1(items, r)))
        for name, sc in m.items():
            rows.append((f"W-A {name}", top1(items, sc)))
        print("W-A layers chosen (per held-out game):", dict(chosen))
    lp = os.path.join(args.out, "lora.jsonl")
    if os.path.exists(lp) and os.path.getsize(lp):
        rows.append(("W-B LoRA, held-out folds", top1(items, load_scores(lp))))
    for p in sorted(glob.glob(os.path.join(args.out, "wc_*.jsonl"))):
        rows.append((f"W-C {os.path.basename(p)[3:-6]} zero-shot", top1(items, load_scores(p))))
    print(f"\n{'option':44} {'Q-goal (primary)':>18} {'Q-self':>18}")
    for name, r in rows:
        print(f"{name:44} {fmt(r.get('goal')):>18} {fmt(r.get('self')):>18}")


if __name__ == "__main__":
    main()
