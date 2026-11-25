"""Funções auxiliares: carregar dados, máscaras de subamostragem e métricas."""
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def load_slice(path="data/mni152_t1.nii.gz", z=None, size=256):
    """Carrega uma fatia axial 2D, normaliza para [0,1] e coloca em size x size."""
    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    z = vol.shape[2] // 2 if z is None else z
    sl = np.rot90(vol[:, :, z])                    # orientação mais intuitiva
    sl = sl / sl.max()
    out = np.zeros((size, size), dtype=np.float32)  # pad/crop central
    h, w = sl.shape
    ys, xs = max((size - h) // 2, 0), max((size - w) // 2, 0)
    yc, xc = max((h - size) // 2, 0), max((w - size) // 2, 0)
    hh, ww = min(h, size), min(w, size)
    out[ys:ys + hh, xs:xs + ww] = sl[yc:yc + hh, xc:xc + ww]
    return out


def add_noise(ksp, sigma, seed=0):
    """Ruído gaussiano complexo no k-space (sigma relativo ao máx. de |k|)."""
    rng = np.random.default_rng(seed)
    s = sigma * np.abs(ksp).max()
    return ksp + s * (rng.standard_normal(ksp.shape) + 1j * rng.standard_normal(ksp.shape))


def mask_random(shape, R, n_center=16, seed=0):
    """Máscara cartesiana 1D (colunas), aleatória com densidade variável.
    Centro totalmente amostrado; resto sorteado com densidade decrescente."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    m = np.zeros(shape, dtype=bool)
    centro = np.arange(nx // 2 - n_center // 2, nx // 2 + n_center // 2)
    m[:, centro] = True
    n_extra = max(nx // R - n_center, 0)
    p = (1 - np.abs(np.linspace(-1, 1, nx))) ** 3 + 1e-6
    p[centro] = 0
    p /= p.sum()
    extra = rng.choice(nx, size=n_extra, replace=False, p=p)
    m[:, extra] = True
    return m


def mask_regular(shape, R, n_center=16):
    """Máscara cartesiana equiespaçada (1 de cada R colunas) + centro completo."""
    ny, nx = shape
    m = np.zeros(shape, dtype=bool)
    m[:, ::R] = True
    m[:, nx // 2 - n_center // 2: nx // 2 + n_center // 2] = True
    return m


def metrics(ref, x):
    """PSNR (dB) e SSIM entre referência e reconstrução (magnitudes)."""
    dr = float(ref.max() - ref.min())
    return _psnr(ref, x, data_range=dr), _ssim(ref, x, data_range=dr)


# ---------------------------------------------------------------- SigPy: k-space e reconstrução
import sigpy as sp
import sigpy.mri as mr


def to_kspace(img):
    """Imagem -> espaço-k (FFT centrada do SigPy)."""
    return sp.fft(img.astype(np.complex64))


def recon_zf(k):
    """Zero-filled: FFT inversa do SigPy."""
    return np.abs(sp.ifft(k))


def recon_cs(k, lamda=1e-2, max_iter=100):
    """Compressed sensing L1-wavelet do SigPy (1 bobina)."""
    mps = np.ones((1,) + k.shape, dtype=np.complex64)
    app = mr.app.L1WaveletRecon(k[None], mps, lamda, max_iter=max_iter, show_pbar=False)
    return np.abs(app.run())


def make_mask(kind, shape, R, seed=0):
    """'regular' ou 'random'."""
    if kind == "regular":
        return mask_regular(shape, R)
    return mask_random(shape, R, seed=seed)
