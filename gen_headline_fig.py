"""看板図: few-shot 高次元での量子 vs 古典。(a) 探索 (seeds 0-9, d=4-12),
(b) 事前登録確証 (seeds 10-29, d=12-16)。"""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from avqkan import Config, Problem
from classical_baseline import fit_kan, fit_mlp

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})

def es(pat, seeds):
    out = []
    for s in seeds:
        rows = list(csv.DictReader(open(pat.replace("SEED", str(s)))))
        te = np.array([float(r["test_absdist"]) for r in rows])
        co = np.array([float(r["cost"]) for r in rows])
        out.append(te[int(np.argmin(co))])
    return np.array(out)

def classical(d, seeds):
    kan, m16, lin, const, oracle = [], [], [], [], []
    for s in seeds:
        prob = Problem(Config(nq=d, seed=s), f"sepn{d}")
        vals = {}
        p, _, _ = fit_kan(prob, 8); vals["kan"] = np.sum(np.abs(p - prob.ff))
        p, _, _ = fit_mlp(prob, (8, 8), s); vals["m88"] = np.sum(np.abs(p - prob.ff))
        p, _, _ = fit_mlp(prob, (16,), s); vals["m16"] = np.sum(np.abs(p - prob.ff))
        X, Xf = prob.X[:, :d], prob.Xf[:, :d]
        Xa = np.hstack([X, np.ones((10, 1))]); Xfa = np.hstack([Xf, np.ones((50, 1))])
        w, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
        vals["lin"] = np.sum(np.abs(np.clip(Xfa @ w, -1, 1) - prob.ff))
        kan.append(vals["kan"]); m16.append(vals["m16"]); lin.append(vals["lin"])
        const.append(np.sum(np.abs(prob.ff - np.median(prob.f))))
        oracle.append(min(vals.values()))
    return (np.array(kan), np.array(m16), np.array(lin),
            np.array(const), np.array(oracle))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

# (a) exploration, seeds 0-9, d=4..12
DS = [4, 6, 8, 10, 12]
curves = {k: [] for k in ("ng", "cob", "kan", "m16", "lin", "const")}
for d in DS:
    s09 = range(10)
    curves["ng"].append(es(f"results_c/ite_sepn{d}_seedSEED.csv", s09))
    curves["cob"].append(es(f"results_c_m30/cobyla_sepn{d}_seedSEED.csv", s09))
    kan, m16, lin, const, _ = classical(d, s09)
    curves["kan"].append(kan); curves["m16"].append(m16)
    curves["lin"].append(lin); curves["const"].append(const)
style = {"ng": ("#d62728", "-", "s", "quantum, natural gradient"),
         "cob": ("#1f77b4", "-", "o", "quantum, COBYLA"),
         "kan": ("#2ca02c", "--", "^", "classical KAN"),
         "m16": ("#9467bd", "--", "v", "MLP 16"),
         "lin": ("#8c564b", "--", "D", "linear regression"),
         "const": ("k", ":", "", "constant predictor")}
for k, (c, ls, mk, lb) in style.items():
    mu = [v.mean() for v in curves[k]]
    if k == "const":
        ax[0].plot(DS, mu, ls, color=c, label=lb)
    else:
        sd = [v.std(ddof=1) for v in curves[k]]
        ax[0].errorbar(DS, mu, yerr=sd, color=c, ls=ls, marker=mk, ms=4,
                       capsize=2, lw=1.4, label=lb)
ax[0].set_xticks(DS); ax[0].set_xlabel("Input dimension $d$ (= qubits)")
ax[0].set_ylabel("Few-shot test error (10 training points)")
ax[0].set_title("(a) Exploration, seeds 0--9")
ax[0].legend(fontsize=7.5)

# (b) confirmation, seeds 10-29, d=12/14/16
DC = [12, 14, 16]
S = range(10, 30)
w = 0.27
holm = {12: "0.0037", 14: "0.0003", 16: "0.0002"}
for i, d in enumerate(DC):
    ng = es(f"results_confirm/ite_sepn{d}_seedSEED.csv", S)
    cob = es(f"results_confirm_m30/cobyla_sepn{d}_seedSEED.csv", S)
    _, _, _, const, oracle = classical(d, S)
    ax[1].bar(i - w, ng.mean(), w, yerr=ng.std(ddof=1), capsize=3,
              color="#d62728", label="quantum, natural gradient" if i == 0 else None)
    ax[1].bar(i, cob.mean(), w, yerr=cob.std(ddof=1), capsize=3,
              color="#1f77b4", label="quantum, COBYLA" if i == 0 else None)
    ax[1].bar(i + w, oracle.mean(), w, yerr=oracle.std(ddof=1), capsize=3,
              color="#7f7f7f", label="oracle-best classical" if i == 0 else None)
    ax[1].plot([i - 1.6 * w, i + 1.6 * w], [const.mean()] * 2, "k:",
               label="constant predictor" if i == 0 else None)
    ax[1].text(i, 21.3,
               f"$p_{{\\rm Holm}}={holm[d]}$", ha="center", fontsize=8)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"$d={d}$" for d in DC])
ax[1].set_ylim(0, 23.5)
ax[1].set_title("(b) Pre-registered confirmation, fresh seeds 10--29")
ax[1].legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig("figs/fig_headline.png", dpi=200)
print("wrote figs/fig_headline.png")
