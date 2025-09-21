import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift

import pandas as pd
import os
import json

np.random.seed(42)

# Grids and FFT helpers

def makeGrid(N: int, pixscale: float = 1.0):
    """Return 2D coordinate grids centered at 0 with pixel scale (pixels -> same units)."""
    ax = (np.arange(N) - N//2) * pixscale
    X, Y = np.meshgrid(ax, ax, indexing='xy')
    return X, Y

def padding(shape, target_shape):
    """Return np.pad 'pad_width' to center-pad an array to target_shape with zeros."""
    ny, nx = shape
    Ty, Tx = target_shape
    py = Ty - ny
    px = Tx - nx
    if py < 0 or px < 0:
        raise ValueError("target_shape must be >= shape")
    return ((py//2, py - py//2), (px//2, px - px//2))

def gaussian_psf_ft(beta: float, kx: np.ndarray, ky: np.ndarray):
    """Fourier transform of isotropic Gaussian PSF (no 2π prefactors here): exp(-0.5 * beta^2 * k^2)."""
    k2 = kx**2 + ky**2
    return np.exp(-0.5 * (beta**2) * k2)

def fft_convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve via FFT with zero-padding to avoid wrap-around artifacts; crop back to img size."""
    Ny, Nx = img.shape
    Ky, Kx = kernel.shape
    Ty, Tx = Ny + Ky - 1, Nx + Kx - 1
    # round up to next power of 2 for speed
    Ty = 1 << (Ty - 1).bit_length()
    Tx = 1 << (Tx - 1).bit_length()
    pad_img = np.pad(img, padding(img.shape, (Ty, Tx)))
    pad_ker = np.pad(kernel, padding(kernel.shape, (Ty, Tx)))
    F_img = fft2(pad_img)
    F_ker = fft2(ifftshift(pad_ker))  # kernel centered at (0,0)
    conv = np.real(ifft2(F_img * F_ker))
    # crop to original size, centered
    sy, sx = (Ty - Ny)//2, (Tx - Nx)//2
    return conv[sy:sy+Ny, sx:sx+Nx]

def fourier_reconvolve_to_gaussian(observed: np.ndarray, orig_psf: np.ndarray, beta_desired: float) -> np.ndarray:
    """
    In Fourier space, transform from an arbitrary PSF to an isotropic Gaussian of width beta_desired.
    """
    Ny, Nx = observed.shape
    Ty = 1 << (Ny - 1).bit_length()
    Tx = 1 << (Nx - 1).bit_length()
    pad_obs = np.pad(observed, padding(observed.shape, (Ty, Tx)))
    pad_psf = np.pad(orig_psf, padding(orig_psf.shape, (Ty, Tx)))

    F_obs = fft2(pad_obs)
    F_psf = fft2(ifftshift(pad_psf))

    ky = 2*np.pi * (np.fft.fftfreq(Ty))
    kx = 2*np.pi * (np.fft.fftfreq(Tx))
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    F_desired = gaussian_psf_ft(beta_desired, KX, KY)

    eps = 1e-6 * np.max(np.abs(F_psf))
    H = F_desired / (F_psf + eps)
    new_img = np.real(ifft2(F_obs * H))

    sy, sx = (Ty - Ny)//2, (Tx - Nx)//2
    return new_img[sy:sy+Ny, sx:sx+Nx]


# Galaxy models
@dataclass
class RegularGalaxyParams:
    N: int = 128
    rd: float | None = None   # disk scale length (pixels)
    rs: float | None = None   # spheroid scale length (pixels)
    fsd: float = 1.0          # relative bulge weight

    def __post_init__(self):
        if self.rd is None:
            self.rd = self.N / 32.0
        if self.rs is None:
            self.rs = self.rd / 2.0

def regular_galaxy(params: RegularGalaxyParams, q: float, phi: float) -> np.ndarray:
    """
    Exponential disc + de Vaucouleurs bulge; project to ellipse with axis ratio q and rotate by phi.
    Normalized to unit flux.
    """
    N = params.N
    X, Y = makeGrid(N, 1.0)
    c, s = np.cos(phi), np.sin(phi)
    Xr = c*X + s*Y
    Yr = -s*X + c*Y
    r = np.sqrt(Xr**2 + (Yr / q)**2 + 1e-12)  # elliptical radius
    f_disk = np.exp(-r / params.rd)
    f_sph = np.exp(-(r / params.rs)**0.25)    # de Vaucouleurs
    img = f_disk + params.fsd * f_sph
    img /= img.sum()
    return img

def random_walk_irregular(N: int = 256, steps: int = 20000, reset_radius_frac: float = 1/6) -> np.ndarray:
    """
    Irregular galaxy: accumulate a 2D random walk visitation image, periodically resetting near center.
    Normalized to unit flux.
    """
    img = np.zeros((N, N), dtype=np.float32)
    center = np.array([N//2, N//2], dtype=int)
    pos = center.copy()
    max_r2 = (reset_radius_frac * N)**2
    for _ in range(steps):
        img[pos[0], pos[1]] += 1.0
        step = np.random.randint(0, 4)
        if step == 0:
            pos[0] = (pos[0] + 1) % N
        elif step == 1:
            pos[0] = (pos[0] - 1) % N
        elif step == 2:
            pos[1] = (pos[1] + 1) % N
        else:
            pos[1] = (pos[1] - 1) % N
        dy = pos[0] - center[0]
        dx = pos[1] - center[1]
        if (dx*dx + dy*dy) > max_r2:
            pos = center.copy()
    img /= img.sum()
    return img

# Lensing mapping (constant shear)

def bilinear_sample(img: np.ndarray, Xs: np.ndarray, Ys: np.ndarray) -> np.ndarray:
    """Sample img at continuous coords with bilinear interpolation; coordinates are pixels centered at 0."""
    N = img.shape[0]
    x = Xs + N/2.0
    y = Ys + N/2.0
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, N-1); y1 = np.clip(y0 + 1, 0, N-1)
    x0 = np.clip(x0, 0, N-1); y0 = np.clip(y0, 0, N-1)
    wx = x - x0; wy = y - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    top = Ia*(1-wx) + Ib*wx
    bot = Ic*(1-wx) + Id*wx
    return top*(1-wy) + bot*wy

def apply_shear(source_img: np.ndarray, gamma1: float, gamma2: float, kappa: float = 0.0) -> np.ndarray:
    """Apply constant (reduced) shear via affine mapping, resampling the source by inverse mapping."""
    N = source_img.shape[0]
    X, Y = makeGrid(N, 1.0)
    A = np.array([[1 + kappa + gamma1, gamma2],
                  [gamma2, 1 + kappa - gamma1]], dtype=float)
    Ainv = np.linalg.inv(A)
    coords = np.stack([X.ravel(), Y.ravel()], axis=0)
    src = (Ainv @ coords).reshape(2, N, N)
    Xs, Ys = src[0], src[1]
    return bilinear_sample(source_img, Xs, Ys)


# PSF models
def psf_W1(N: int, r: float, theta: float) -> np.ndarray:
    X, Y = makeGrid(N, 1.0)
    c, s = np.cos(theta), np.sin(theta)
    xr = c*X + s*Y
    yr = -s*X + c*Y
    val = np.exp(- (np.abs(xr - yr) + np.abs(xr + yr))**2 / (8.0 * r**2))
    val /= val.sum()
    return val

def psf_W2(N: int, r: float, theta: float) -> np.ndarray:
    X, Y = makeGrid(N, 1.0)
    c, s = np.cos(theta), np.sin(theta)
    xr = c*X + s*Y
    yr = -s*X + c*Y
    val = np.exp(- (xr**2 + 0.8*yr**2) / (2.0 * r**2))
    val /= val.sum()
    return val

# Fourier-space shear estimator

def shear_estimator_fourier(img: np.ndarray, beta: float) -> Tuple[float, float]:
    """
    Compute gamma_1, gamma_2 from quadratic combinations of derivatives using Parseval in Fourier space.
    """
    Ny, Nx = img.shape
    f = img - np.mean(img)  # remove DC
    Ty = 1 << (Ny - 1).bit_length()
    Tx = 1 << (Nx - 1).bit_length()
    pad = np.pad(f, padding(f.shape, (Ty, Tx)))
    F = fft2(pad)

    ky = 2*np.pi * (np.fft.fftfreq(Ty))
    kx = 2*np.pi * (np.fft.fftfreq(Tx))
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    P = np.abs(F)**2

    Sxx = np.sum((KX**2) * P)
    Syy = np.sum((KY**2) * P)
    Sxy = np.sum((KX * KY) * P)
    Sgradlap = - np.sum((K2**2) * P)

    D = (Sxx + Syy) + (beta**2 / 2.0) * Sgradlap
    N1 = 0.5 * (Sxx - Syy)
    N2 = Sxy

    gamma1 = - N1 / D if D != 0 else 0.0
    gamma2 = - N2 / D if D != 0 else 0.0
    return float(gamma1), float(gamma2)

# Experiments

def experiment_regular(num_gal: int = 80, N: int = 128) -> "np.ndarray[dict]":
    """
    Regular galaxies through two anisotropic PSFs (W1/W2) and different input shears.
    Returns a list of result dicts.
    """
    params = RegularGalaxyParams(N=N)
    r_psf = 6.0  # pixels
    beta_desired = r_psf * 4.0 / 3.0

    shears_W1 = [(-0.012, 0.035), (-0.032, -0.005), (0.01, 0.02)]
    shears_W2 = [(0.015, -0.024), (0.05, 0.01), (-0.04, -0.04)]

    results = []
    for label, psf_fun, shears in [("W1", psf_W1, shears_W1), ("W2", psf_W2, shears_W2)]:
        for (g1, g2) in shears:
            g1hats, g2hats = [], []
            for _ in range(num_gal):
                q = float(np.clip(np.random.uniform(0.2, 1.0), 0.2, 1.0))
                phi = float(np.random.uniform(0, 2*np.pi))
                gal = regular_galaxy(params, q=q, phi=phi)
                lensed = apply_shear(gal, g1, g2, kappa=0.0)
                theta = float(np.random.uniform(0, 2*np.pi))
                psf = psf_fun(N, r_psf, theta)
                obs = fft_convolve(lensed, psf)
                obs_iso = fourier_reconvolve_to_gaussian(obs, psf, beta_desired)
                g1_hat, g2_hat = shear_estimator_fourier(obs_iso, beta_desired)
                g1hats.append(g1_hat); g2hats.append(g2_hat)
            g1hats = np.array(g1hats); g2hats = np.array(g2hats)
            results.append(dict(
                psf=label, input_gamma1=g1, input_gamma2=g2,
                mean_gamma1_hat=float(g1hats.mean()), std_gamma1_hat=float(g1hats.std(ddof=1)),
                mean_gamma2_hat=float(g2hats.mean()), std_gamma2_hat=float(g2hats.std(ddof=1)),
                n=num_gal, Npix=N
            ))
    return results

def experiment_irregular(num_gal: int = 30, N: int = 160):
    """
    Irregular galaxies smoothed by isotropic Gaussian PSFs of various β.
    Returns (table_rows, per_beta_samples).
    """
    target_g1, target_g2 = 0.03, 0.0
    betas = [8.0, 4.0, 2.0, 1.0]

    table = []
    per_beta = {}

    for beta in betas:
        g1s = []
        for _ in range(num_gal):
            gal = random_walk_irregular(N=N, steps=20000, reset_radius_frac=1/6)
            lensed = apply_shear(gal, target_g1, target_g2, kappa=0.0)
            # Make Gaussian PSF
            rad = int(max(5, math.ceil(4*beta)))
            size = 2*rad + 1
            X, Y = makeGrid(size, 1.0)
            gauss = np.exp(-(X**2 + Y**2) / (2*beta*beta))
            gauss /= gauss.sum()
            obs = fft_convolve(lensed, gauss)
            g1_hat, g2_hat = shear_estimator_fourier(obs, beta)
            g1s.append(g1_hat)
        g1s = np.array(g1s)
        per_beta[beta] = g1s
        table.append(dict(
            beta=beta, n=num_gal, input_gamma1=target_g1,
            mean_gamma1_hat=float(g1s.mean()),
            std_gamma1_hat=float(g1s.std(ddof=1)),
            var_gamma1_hat=float(g1s.var(ddof=1))
        ))
    # sort by beta
    table = sorted(table, key=lambda r: r["beta"])
    return table, per_beta

# Plot utilities

def plot_regular(results, out_png: str):
    df = pd.DataFrame(results)
    fig = plt.figure(figsize=(6,5))
    for psf_label, marker in [("W1", "o"), ("W2", "s")]:
        sub = df[df["psf"] == psf_label]
        plt.errorbar(sub["input_gamma1"], sub["mean_gamma1_hat"],
                     yerr=sub["std_gamma1_hat"]/np.sqrt(sub["n"]),
                     fmt=marker, label=f"{psf_label} gamma1")
        plt.errorbar(sub["input_gamma2"], sub["mean_gamma2_hat"],
                     yerr=sub["std_gamma2_hat"]/np.sqrt(sub["n"]),
                     fmt=marker, linestyle='--', label=f"{psf_label} gamma2")
    all_inputs = np.concatenate([df["input_gamma1"].values, df["input_gamma2"].values])
    xx = np.linspace(all_inputs.min()-0.06, all_inputs.max()+0.06, 100)
    plt.plot(xx, xx, linewidth=1)
    plt.xlabel("Input shear component")
    plt.ylabel("Recovered shear component")
    plt.title("Regular galaxies: shear recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return df

def plot_irregular_histograms(per_beta: Dict[float, np.ndarray], target_g1: float, out_prefix: str):
    paths = []
    for beta in sorted(per_beta.keys(), reverse=True):
        fig = plt.figure(figsize=(6,4))
        vals = per_beta[beta]
        plt.hist(vals, bins=20, density=True)
        plt.axvline(target_g1, linestyle='--')
        plt.xlabel("Recovered gamma1")
        plt.ylabel("Probability density")
        plt.title(f"Irregular galaxies: PDF of gamma1 (β={beta:.0f} px)")
        plt.tight_layout()
        out_path = f"{out_prefix}_beta_{int(beta)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        paths.append(out_path)
    return paths

def plot_irregular_variance(table_rows, out_png: str):
    df = pd.DataFrame(table_rows).sort_values("beta")
    fig = plt.figure(figsize=(6,4))
    x = df["beta"].values
    y = df["var_gamma1_hat"].values
    plt.loglog(x, y, marker='o')
    # power-law fit
    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    alpha = coeffs[0]; A = np.exp(coeffs[1])
    xx = np.linspace(x.min()*0.9, x.max()*1.1, 100)
    plt.loglog(xx, A*xx**alpha, linestyle='--')
    plt.xlabel("beta (pixels)")
    plt.ylabel("Var[gamma1]")
    plt.title(f"Irregular galaxies: variance vs beta (slope ≈ {alpha:.2f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return df

def plot_galaxy(img: np.ndarray, title: str = "Galaxy", out_png: str='out.png'):
    plt.figure(figsize=(4,4))
    plt.imshow(img, origin="lower", cmap="inferno", interpolation="nearest")
    plt.colorbar(label="Flux")
    plt.title(title)
    plt.axis("off")
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



# Main CLI

def main():
    p = argparse.ArgumentParser(description="Zhang (2008) Fourier-space shear estimator demo")
    p.add_argument("--regular", action="store_true", help="Run regular-galaxy PSF experiment (W1/W2).")
    p.add_argument("--irregular", action="store_true", help="Run irregular-galaxy beta-sweep experiment.")
    p.add_argument("--outdir", type=str, default=".", help="Directory to write outputs (plots, CSVs).")
    p.add_argument("--num_gal_regular", type=int, default=80, help="Galaxies per shear per PSF for regular run.")
    p.add_argument("--num_gal_irregular", type=int, default=30, help="Galaxies per beta for irregular run.")
    p.add_argument("--N_regular", type=int, default=128, help="Stamp size for regular galaxies.")
    p.add_argument("--N_irregular", type=int, default=160, help="Stamp size for irregular galaxies.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.regular:
        res = experiment_regular(num_gal=args.num_gal_regular, N=args.N_regular)
        df = plot_regular(res, os.path.join(args.outdir, "regular_shear_recovery.png"))
        csv_path = os.path.join(args.outdir, "regular_shear_results.csv")
        df.to_csv(csv_path, index=False)
        print("Regular experiment saved:")
        print("  - Plot:", os.path.join(args.outdir, "regular_shear_recovery.png"))
        print("  - CSV :", csv_path)

        params = RegularGalaxyParams(N=args.N_regular)
        gal = regular_galaxy(params, q=0.6, phi=np.pi/6)
        lensed = apply_shear(gal, gamma1=0.05, gamma2=0.02)
        psf = psf_W1(args.N_regular, r=6.0, theta=0.0)
        obs = fft_convolve(lensed, psf)
        plot_galaxy(gal, "Regular galaxy (intrinsic)", os.path.join(args.outdir, "sample_regular_intrinsic.png"))
        plot_galaxy(lensed, "Regular galaxy (lensed)", os.path.join(args.outdir, "sample_regular_lensed.png"))
        plot_galaxy(obs, "Regular galaxy (lensed + PSF)", os.path.join(args.outdir, "sample_regular_observed.png"))
        print("Saved sample regular galaxy plots.")

    if args.irregular:
        table, per_beta = experiment_irregular(num_gal=args.num_gal_irregular, N=args.N_irregular)
        df = plot_irregular_variance(table, os.path.join(args.outdir, "irregular_variance_vs_beta.png"))
        hist_paths = plot_irregular_histograms(per_beta, target_g1=0.03,
                                               out_prefix=os.path.join(args.outdir, "irregular_pdf"))
        csv_path = os.path.join(args.outdir, "irregular_variance_results.csv")
        df.to_csv(csv_path, index=False)
        print("Irregular experiment saved:")
        print("  - Plot:", os.path.join(args.outdir, "irregular_variance_vs_beta.png"))
        print("  - PDFs:", ", ".join(hist_paths))
        print("  - CSV :", csv_path)

        gal = random_walk_irregular(N=args.N_irregular)
        lensed = apply_shear(gal, gamma1=0.03, gamma2=0.0)
        plot_galaxy(gal, "Irregular galaxy (intrinsic)", os.path.join(args.outdir, "sample_irregular_intrinsic.png"))
        plot_galaxy(lensed, "Irregular galaxy (lensed)", os.path.join(args.outdir, "sample_irregular_lensed.png"))
        print("Saved sample irregular galaxy plots.")

    if not (args.regular or args.irregular):
        print("Nothing selected. Try: --regular and/or --irregular")

if __name__ == "__main__":
    main()
