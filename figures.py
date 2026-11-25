"""Uma função por figura do notebook (usa as funções do mrisim.py)."""
import os, time, csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mrisim import make_mask, recon_zf, recon_cs, add_noise, metrics

# ---------------------------------------------------------------- utilitários de figuras
def _save(fig, name, save_dir):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, name), dpi=150)


# ---------------------------------------------------------------- Figura exp1
def fig_kspace(img, ksp, radius=20, save_dir=None):
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - ny // 2, xx - nx // 2)
    low, high = ksp * (r <= radius), ksp * (r > radius)
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))
    ax[0, 0].imshow(np.log1p(np.abs(ksp) * 100), cmap="gray"); ax[0, 0].set_title("k-space completo (log)")
    ax[0, 1].imshow(np.log1p(np.abs(low) * 100), cmap="gray"); ax[0, 1].set_title(f"só centro (r≤{radius})")
    ax[0, 2].imshow(np.log1p(np.abs(high) * 100), cmap="gray"); ax[0, 2].set_title(f"só periferia (r>{radius})")
    ax[1, 0].imshow(img, cmap="gray"); ax[1, 0].set_title("imagem original")
    ax[1, 1].imshow(recon_zf(low), cmap="gray"); ax[1, 1].set_title("recon centro: contraste")
    h = recon_zf(high)
    ax[1, 2].imshow(h, cmap="gray", vmax=np.percentile(h, 99.5)); ax[1, 2].set_title("recon periferia: bordas")
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); _save(fig, "exp1_kspace_center_vs_periphery.png", save_dir)
    return fig


# ---------------------------------------------------------------- Figura exp0
def fig_zero_filled(img, ksp, R=4, seed=0, save_dir=None):
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))
    for i, kind in enumerate(("regular", "random")):
        m = make_mask(kind, img.shape, R, seed=seed)
        zf = recon_zf(ksp * m)
        p, s = metrics(img, zf)
        ax[i, 0].imshow(m, cmap="gray", aspect="auto"); ax[i, 0].set_title(f"máscara {kind} (R={R})")
        ax[i, 1].imshow(zf, cmap="gray"); ax[i, 1].set_title(f"zero-filled\nPSNR {p:.1f} dB | SSIM {s:.2f}", fontsize=9)
        ax[i, 2].imshow(np.abs(img - zf), cmap="hot", vmax=0.2); ax[i, 2].set_title("erro |GT − ZF|")
        for a in ax[i]: a.axis("off")
    fig.tight_layout(); _save(fig, f"exp0_zero_filled_R{R}.png", save_dir)
    return fig


# ---------------------------------------------------------------- Experiências (tabela)
def run_experiments(img, ksp, Rs=(2, 4, 8), masks=("regular", "random"),
                    lamda=1e-2, max_iter=100, seed=0, verbose=True):
    rows, store = [], {}
    for R in Rs:
        for kind in masks:
            m = make_mask(kind, img.shape, R, seed=seed)
            k = ksp * m
            t0 = time.time(); zf = recon_zf(k); t_zf = time.time() - t0
            t0 = time.time(); cs = recon_cs(k, lamda, max_iter); t_cs = time.time() - t0
            for meth, x, t in (("zero-filled", zf, t_zf), ("L1-wavelet", cs, t_cs)):
                p, s = metrics(img, x)
                rows.append(dict(R=R, mask=kind, method=meth, PSNR=round(p, 2), SSIM=round(s, 4),
                                 sampled_pct=round(100 * float(m[0].mean()), 1), time_s=round(t, 2)))
                if verbose: print(rows[-1])
            store[(R, kind)] = (m, zf, cs)
    return rows, store


def save_table(rows, path="results/metrics.csv"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- Figura exp3
def fig_mask_type(img, ksp, R=4, lamda=1e-2, max_iter=100, seed=0, save_dir=None):
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    for i, kind in enumerate(("regular", "random")):
        m = make_mask(kind, img.shape, R, seed=seed)
        k = ksp * m
        zf, cs = recon_zf(k), recon_cs(k, lamda, max_iter)
        ax[i, 0].imshow(m, cmap="gray", aspect="auto"); ax[i, 0].set_title(f"máscara {kind} (R={R})")
        ax[i, 1].imshow(zf, cmap="gray"); ax[i, 1].set_title("zero-filled")
        ax[i, 2].imshow(cs, cmap="gray"); ax[i, 2].set_title("L1-wavelet")
        ax[i, 3].imshow(np.abs(img - cs), cmap="hot", vmax=0.2); ax[i, 3].set_title("erro |GT − CS|")
        for a in ax[i]: a.axis("off")
    fig.tight_layout(); _save(fig, f"exp3_mask_type_R{R}.png", save_dir)
    return fig


# ---------------------------------------------------------------- Figura exp2_acceleration
def fig_acceleration(img, store, Rs=(2, 4, 8), mask="random", save_dir=None):
    fig, ax = plt.subplots(len(Rs) + 1, 4, figsize=(13, 3.2 * (len(Rs) + 1)))
    ax[0, 0].imshow(img, cmap="gray"); ax[0, 0].set_title("Ground truth")
    for a in ax[0, 1:]: a.axis("off")
    for i, R in enumerate(Rs, start=1):
        m, zf, cs = store[(R, mask)]
        pz, sz = metrics(img, zf); pc, sc = metrics(img, cs)
        ax[i, 1].imshow(m, cmap="gray", aspect="auto"); ax[i, 1].set_title(f"máscara R={R}")
        ax[i, 2].imshow(zf, cmap="gray"); ax[i, 2].set_title(f"ZF  PSNR {pz:.1f} dB | SSIM {sz:.2f}", fontsize=9)
        ax[i, 3].imshow(cs, cmap="gray"); ax[i, 3].set_title(f"CS  PSNR {pc:.1f} dB | SSIM {sc:.2f}", fontsize=9)
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout()
    _save(fig, "exp2_acceleration.png" if mask == "random" else f"exp2_acceleration_{mask}.png", save_dir)
    return fig


# ---------------------------------------------------------------- Figura exp2_metrics_vs_R
def fig_metrics_vs_R(rows, masks=("random",), save_dir=None):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    estilos = {"zero-filled": ("o", "C0"), "L1-wavelet": ("s", "C1")}
    ls_mask = {"random": "-", "regular": ":"}
    for kind in masks:
        for meth, (mk, cor) in estilos.items():
            sel = [r for r in rows if r["mask"] == kind and r["method"] == meth]
            ls = ("--" if meth == "zero-filled" else "-") if len(masks) == 1 else ls_mask.get(kind, "-")
            lab = meth if len(masks) == 1 else f"{meth} ({kind})"
            for j, metric in enumerate(("PSNR", "SSIM")):
                ax[j].plot([r["R"] for r in sel], [r[metric] for r in sel], marker=mk, color=cor, ls=ls, label=lab)
    ax[0].set_ylabel("PSNR (dB)"); ax[0].legend(); ax[1].set_ylabel("SSIM")
    for a in ax: a.set_xlabel("fator de aceleração R"); a.grid(alpha=.3)
    fig.tight_layout(); _save(fig, "exp2_metrics_vs_R.png", save_dir)
    return fig


# ---------------------------------------------------------------- Figura exp5
def fig_metrics_bars(rows, save_dir=None):
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, metric in zip(ax, ("PSNR", "SSIM")):
        piv = tab.pivot_table(index="R", columns=["mask", "method"], values=metric)
        piv.plot(kind="bar", ax=a, rot=0, width=0.8)
        a.set_ylabel("PSNR (dB)" if metric == "PSNR" else "SSIM"); a.set_xlabel("fator de aceleração R")
        a.grid(alpha=.3, axis="y")
        a.legend([f"{m} / {me}" for m, me in piv.columns], fontsize=8)
    fig.tight_layout(); _save(fig, "exp5_metrics_bars.png", save_dir)
    return fig


# ---------------------------------------------------------------- exp4: lambda
def lambda_sweep(img, ksp, R=4, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed=0, max_iter=100):
    k = ksp * make_mask("random", img.shape, R, seed=seed)
    out = []
    for lam in lams:
        x = recon_cs(k, lam, max_iter); p, s = metrics(img, x)
        print(f"lambda={lam:g}: PSNR={p:.2f} dB | SSIM={s:.4f}")
        out.append(dict(lam=lam, img=x, PSNR=p, SSIM=s))
    return out


def fig_lambda_images(sweep, save_dir=None):
    fig, ax = plt.subplots(1, len(sweep), figsize=(3 * len(sweep), 3.4))
    for a, d in zip(np.atleast_1d(ax), sweep):
        a.imshow(d["img"], cmap="gray"); a.axis("off")
        a.set_title(f"λ={d['lam']:g}\n{d['PSNR']:.1f} dB | {d['SSIM']:.2f}", fontsize=9)
    fig.tight_layout(); _save(fig, "exp4_lambda.png", save_dir)
    return fig


def fig_lambda_curve(sweep, save_dir=None):
    lams = [d["lam"] for d in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for a, metric, lab in zip(ax, ("PSNR", "SSIM"), ("PSNR (dB)", "SSIM")):
        v = [d[metric] for d in sweep]
        a.semilogx(lams, v, "s-", color="C1")
        a.plot(lams[int(np.argmax(v))], max(v), "k*", ms=12, label="melhor")
        a.set_xlabel("λ"); a.set_ylabel(lab); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); _save(fig, "exp4_lambda_curve.png", save_dir)
    return fig


# ---------------------------------------------------------------- exp6: ruído
def noise_sweep(img, ksp, R=4, sigmas=(0, 1e-4, 5e-4, 1e-3, 2e-3), lamda=1e-2, seed=0, max_iter=100):
    m = make_mask("random", img.shape, R, seed=seed)
    out = []
    for sg in sigmas:
        k = (add_noise(ksp, sg, seed=seed) if sg > 0 else ksp) * m   # ruído antes da subamostragem
        zf, cs = recon_zf(k), recon_cs(k, lamda, max_iter)
        pz, sz = metrics(img, zf); pc, sc = metrics(img, cs)
        print(f"sigma={sg:g} zero-filled: PSNR={pz:.2f} dB | SSIM={sz:.4f}")
        print(f"sigma={sg:g} L1-wavelet: PSNR={pc:.2f} dB | SSIM={sc:.4f}")
        out.append(dict(sigma=sg, zf=zf, cs=cs, PSNR_zf=pz, SSIM_zf=sz, PSNR_cs=pc, SSIM_cs=sc))
    return out


def fig_noise(nsweep, save_dir=None):
    n = len(nsweep)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for j, d in enumerate(nsweep):
        ax[0, j].imshow(d["zf"], cmap="gray")
        ax[0, j].set_title(f"σ={d['sigma']:g}\nZF {d['PSNR_zf']:.1f} dB | {d['SSIM_zf']:.2f}", fontsize=9)
        ax[1, j].imshow(d["cs"], cmap="gray")
        ax[1, j].set_title(f"CS {d['PSNR_cs']:.1f} dB | {d['SSIM_cs']:.2f}", fontsize=9)
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); _save(fig, "exp6_noise.png", save_dir)
    return fig
