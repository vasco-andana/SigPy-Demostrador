"""Demonstrador SigPy: reconstrução de RM a partir de k-space subamostrado.
Corre:  python run_demo.py   (cria figuras em figures/ e tabela em results/)"""
import csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.mri as mr
from mrisim import load_slice, add_noise, mask_random, mask_regular, metrics

SEED = 0
NOISE = 0.0      # sigma do ruído no k-space (ex.: 0.0005 para testar robustez)
LAMBDA = 1e-2    # regularização L1-wavelet
img = load_slice()
ksp = sp.fft(img.astype(np.complex64))          # k-space completo, centrado
if NOISE > 0:
    ksp = add_noise(ksp, sigma=NOISE, seed=SEED)
print(f"sigpy {sp.__version__ if hasattr(sp,'__version__') else ''} | imagem {img.shape}")


def recon_zf(k):
    return np.abs(sp.ifft(k))


def recon_cs(k, lamda=LAMBDA, max_iter=100):
    """Compressed sensing L1-wavelet (single-coil: eixo de bobinas = 1)."""
    mps = np.ones((1,) + k.shape, dtype=np.complex64)
    app = mr.app.L1WaveletRecon(k[None], mps, lamda, max_iter=max_iter, show_pbar=False)
    return np.abs(app.run())


# ---------- Exp. 1: papel do k-space (centro vs periferia) ----------
ny, nx = img.shape
yy, xx = np.mgrid[:ny, :nx]
r = np.hypot(yy - ny // 2, xx - nx // 2)
low = ksp * (r <= 20)
high = ksp * (r > 20)
fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))
ax[0, 0].imshow(np.log1p(np.abs(ksp) * 100), cmap="gray"); ax[0, 0].set_title("k-space completo (log)")
ax[0, 1].imshow(np.log1p(np.abs(low) * 100), cmap="gray"); ax[0, 1].set_title("só centro (r≤20)")
ax[0, 2].imshow(np.log1p(np.abs(high) * 100), cmap="gray"); ax[0, 2].set_title("só periferia (r>20)")
ax[1, 0].imshow(img, cmap="gray"); ax[1, 0].set_title("imagem original")
ax[1, 1].imshow(recon_zf(low), cmap="gray"); ax[1, 1].set_title("recon centro: contraste")
h = recon_zf(high); ax[1, 2].imshow(h, cmap="gray", vmax=np.percentile(h, 99.5)); ax[1, 2].set_title("recon periferia: bordas")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig("figures/exp1_kspace_center_vs_periphery.png", dpi=150); plt.close()

# ---------- Exp. 2 e 3: fator de aceleração e tipo de máscara ----------
rows, store = [], {}
for R in (2, 4, 8):
    for mname, mfun in (("regular", mask_regular), ("random", lambda s, R: mask_random(s, R, seed=SEED))):
        m = mfun(img.shape, R)
        k = ksp * m
        t0 = time.time(); zf = recon_zf(k); t_zf = time.time() - t0
        t0 = time.time(); cs = recon_cs(k); t_cs = time.time() - t0
        for meth, x, t in (("zero-filled", zf, t_zf), ("L1-wavelet", cs, t_cs)):
            p, s = metrics(img, x)
            rows.append(dict(R=R, mask=mname, method=meth, PSNR=round(p, 2), SSIM=round(s, 4),
                             sampled_pct=round(100 * m[0].mean(), 1), time_s=round(t, 2)))
            print(rows[-1])
        store[(R, mname)] = (m, zf, cs)

with open("results/metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# Figura: aliasing coerente (regular) vs incoerente (random), R=4
fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
for i, mname in enumerate(("regular", "random")):
    m, zf, cs = store[(4, mname)]
    ax[i, 0].imshow(m, cmap="gray", aspect="auto"); ax[i, 0].set_title(f"máscara {mname} (R=4)")
    ax[i, 1].imshow(zf, cmap="gray"); ax[i, 1].set_title("zero-filled")
    ax[i, 2].imshow(cs, cmap="gray"); ax[i, 2].set_title("L1-wavelet")
    ax[i, 3].imshow(np.abs(img - cs), cmap="hot", vmax=0.2); ax[i, 3].set_title("erro |GT − CS|")
    for a in ax[i]: a.axis("off")
plt.tight_layout(); plt.savefig("figures/exp3_mask_type_R4.png", dpi=150); plt.close()

# Figura: vários R com máscara aleatória
fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))
ax[0, 0].imshow(img, cmap="gray"); ax[0, 0].set_title("Ground truth")
for a in ax[0, 1:]: a.axis("off")
for i, R in enumerate((2, 4, 8)):
    m, zf, cs = store[(R, "random")]
    pz, sz = metrics(img, zf); pc, sc = metrics(img, cs)
    ax[i, 1].imshow(m, cmap="gray", aspect="auto"); ax[i, 1].set_title(f"máscara R={R}")
    ax[i, 2].imshow(zf, cmap="gray"); ax[i, 2].set_title(f"ZF  PSNR {pz:.1f} dB | SSIM {sz:.2f}", fontsize=9)
    ax[i, 3].imshow(cs, cmap="gray"); ax[i, 3].set_title(f"CS  PSNR {pc:.1f} dB | SSIM {sc:.2f}", fontsize=9)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig("figures/exp2_acceleration.png", dpi=150); plt.close()

# Gráfico PSNR/SSIM vs R (máscara aleatória)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for meth, st in (("zero-filled", "o--"), ("L1-wavelet", "s-")):
    sel = [r for r in rows if r["mask"] == "random" and r["method"] == meth]
    ax[0].plot([r["R"] for r in sel], [r["PSNR"] for r in sel], st, label=meth)
    ax[1].plot([r["R"] for r in sel], [r["SSIM"] for r in sel], st, label=meth)
ax[0].set_xlabel("fator de aceleração R"); ax[0].set_ylabel("PSNR (dB)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("fator de aceleração R"); ax[1].set_ylabel("SSIM"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/exp2_metrics_vs_R.png", dpi=150); plt.close()

# ---------- Exp. 4 (opcional): regularização lambda, R=4 aleatória ----------
m = mask_random(img.shape, 4, seed=SEED); k = ksp * m
lams = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
fig, ax = plt.subplots(1, len(lams), figsize=(3 * len(lams), 3.4))
for a, lam in zip(ax, lams):
    x = recon_cs(k, lamda=lam); p, s = metrics(img, x)
    a.imshow(x, cmap="gray"); a.set_title(f"λ={lam:g}\n{p:.1f} dB | {s:.2f}", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig("figures/exp4_lambda.png", dpi=150); plt.close()
print("Concluído. Ver figures/ e results/metrics.csv")
