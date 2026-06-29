"""
Verificación de reinserción D203 — vista amplia
================================================
Por cada lesión muestra 3 filas × N slices Z:
  Fila 1 — volumen reinsertado sin máscara (contexto amplio ±40 vox XY)
  Fila 2 — mismo recorte con contorno rojo de la máscara superpuesto
  Fila 3 — zoom ×4 centrado en la lesión (sólo la lesión, sin contexto)

Uso:
    python verify_reinsert_v2.py                   # 4 sujetos aleatorios
    python verify_reinsert_v2.py --n 10
    python verify_reinsert_v2.py --subj SCMB_183
    python verify_reinsert_v2.py --ratio           # sólo ratios numéricos
"""

import os, argparse, random
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label as scipy_label

IMGS_DIR  = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw/Dataset203_LesionGAN/imagesTr'
LBLS_DIR  = IMGS_DIR.replace('imagesTr', 'labelsTr')
OUT_DIR   = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN/verify_reinsert_v2'

CROP_WIDE = 40   # semiancho recorte amplio (Fila 1 y 2)
CROP_ZOOM = 8    # semiancho zoom lesión     (Fila 3)
Z_HALF    = 2    # slices por cada lado del centroide Z


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_pair(subj):
    img = nib.load(os.path.join(IMGS_DIR, f"{subj}_0000.nii.gz")).get_fdata().astype(np.float32)
    lbl = nib.load(os.path.join(LBLS_DIR, f"{subj}.nii.gz")).get_fdata().astype(np.uint8)
    return img, lbl


def get_lesion_centroids(lbl, max_lesions=4):
    labeled, n = scipy_label(lbl > 0)
    centroids  = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        centroids.append((int(coords[:,0].mean()), int(coords[:,1].mean()), int(coords[:,2].mean())))
    sizes = [((lbl[labeled == i] > 0).sum(), c) for i, c in enumerate(centroids, 1)]
    sizes.sort(reverse=True)
    return [c for _, c in sizes[:max_lesions]]


def compute_ratio(img, lbl):
    lesion_vox   = img[lbl > 0]
    surround     = binary_dilation(lbl > 0, iterations=3) & (lbl == 0) & (img > 0)
    surround_vox = img[surround]
    if len(lesion_vox) == 0 or len(surround_vox) == 0:
        return None
    return float(lesion_vox.mean() / surround_vox.mean())


# ---------------------------------------------------------------------------
# Helpers de slice
# ---------------------------------------------------------------------------

def get_slice(vol, cx, cy, cz, half_xy, z):
    """Devuelve recorte 2D (Y×X transpuesto para imshow) del volumen en z dado."""
    X, Y, Z = vol.shape
    x0 = max(0, cx - half_xy); x1 = min(X, cx + half_xy)
    y0 = max(0, cy - half_xy); y1 = min(Y, cy + half_xy)
    z  = max(0, min(Z-1, z))
    return vol[x0:x1, y0:y1, z].T, x0, y0   # T → (Y, X) para imshow origin='lower'


def vrange(img, cx, cy, cz, half_xy, z_slices):
    """Rango de intensidad robusto del recorte amplio."""
    X, Y, Z = img.shape
    x0 = max(0, cx - half_xy); x1 = min(X, cx + half_xy)
    y0 = max(0, cy - half_xy); y1 = min(Y, cy + half_xy)
    region = img[x0:x1, y0:y1, :][:,:,[max(0,min(Z-1,z)) for z in z_slices]]
    vals   = region[region > 0]
    if len(vals) == 0:
        return 0, 1
    return float(np.percentile(vals, 1)), float(np.percentile(vals, 99))


# ---------------------------------------------------------------------------
# Figura por sujeto
# ---------------------------------------------------------------------------

def make_figure(subj, max_lesions=4):
    img, lbl = load_pair(subj)
    centroids = get_lesion_centroids(lbl, max_lesions=max_lesions)
    if not centroids:
        print(f"  {subj}: sin lesiones → skip")
        return None

    ratio     = compute_ratio(img, lbl)
    n_les     = len(centroids)
    n_cols    = 2 * Z_HALF + 1          # número de slices Z (5 por defecto)
    n_rows    = n_les * 3               # 3 filas por lesión

    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 2.8 + 0.6))
    gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.06, wspace=0.04,
                           top=0.94, bottom=0.04, left=0.08, right=0.98)

    ratio_str  = f"{ratio:.3f}" if ratio is not None else "N/A"
    hipoflag   = "✓ hipointenso" if ratio is not None and ratio < 1.0 else "✗ NO hipointenso"
    title_col  = 'green' if ratio is not None and ratio < 1.0 else 'red'
    fig.suptitle(f"{subj}  |  {n_les} lesión(es)  |  ratio inside/outside = {ratio_str}  {hipoflag}",
                 fontsize=10, color=title_col, y=0.98)

    for li, (cx, cy, cz) in enumerate(centroids):
        z_slices = list(range(cz - Z_HALF, cz + Z_HALF + 1))
        vmin, vmax = vrange(img, cx, cy, cz, CROP_WIDE, z_slices)

        row_ctx  = li * 3       # Fila 1: contexto amplio sin máscara
        row_msk  = li * 3 + 1  # Fila 2: contexto amplio + contorno máscara
        row_zoom = li * 3 + 2  # Fila 3: zoom ×4 lesión sola

        for col, z in enumerate(z_slices):
            # ---- Fila 1: contexto amplio, sin máscara ----
            ax1 = fig.add_subplot(gs[row_ctx, col])
            sl, _, _ = get_slice(img, cx, cy, cz, CROP_WIDE, z)
            ax1.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal', interpolation='nearest')
            ax1.set_title(f"Z={z}", fontsize=7, pad=2)
            ax1.axis('off')
            if col == 0:
                ax1.set_ylabel(f"L{li+1} contexto", fontsize=7)

            # ---- Fila 2: contexto amplio + contorno máscara ----
            ax2 = fig.add_subplot(gs[row_msk, col])
            ax2.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal', interpolation='nearest')
            # contorno rojo
            sl_msk, x0, y0 = get_slice(lbl, cx, cy, cz, CROP_WIDE, z)
            if sl_msk.max() > 0:
                ax2.contour(sl_msk, levels=[0.5], colors='red', linewidths=1.2)
            # overlay semitransparente rojo sobre lesión
            if sl_msk.max() > 0:
                overlay = np.zeros((*sl_msk.shape, 4))
                overlay[sl_msk > 0] = [1, 0.1, 0.1, 0.35]
                ax2.imshow(overlay, origin='lower', aspect='equal')
            n_vox = int(sl_msk.sum())
            ax2.set_title(f"mask={n_vox}vox", fontsize=7, pad=2)
            ax2.axis('off')
            if col == 0:
                ax2.set_ylabel(f"L{li+1} + máscara", fontsize=7)

            # ---- Fila 3: zoom lesión sola ----
            ax3 = fig.add_subplot(gs[row_zoom, col])
            sl_z,  _, _ = get_slice(img, cx, cy, cz, CROP_ZOOM, z)
            slm_z, _, _ = get_slice(lbl, cx, cy, cz, CROP_ZOOM, z)

            # rango de intensidad local muy ajustado al zoom
            vals_z = sl_z[sl_z > 0]
            vmin_z = float(np.percentile(vals_z, 1))  if len(vals_z) else vmin
            vmax_z = float(np.percentile(vals_z, 99)) if len(vals_z) else vmax

            ax3.imshow(sl_z, cmap='gray', vmin=vmin_z, vmax=vmax_z,
                       origin='lower', aspect='equal', interpolation='nearest')
            if slm_z.max() > 0:
                overlay_z = np.zeros((*slm_z.shape, 4))
                overlay_z[slm_z > 0] = [1, 0.1, 0.1, 0.45]
                ax3.imshow(overlay_z, origin='lower', aspect='equal')
                ax3.contour(slm_z, levels=[0.5], colors='red', linewidths=0.8)
            ax3.set_title(f"zoom ×{CROP_WIDE//CROP_ZOOM}", fontsize=7, pad=2)
            ax3.axis('off')
            if col == 0:
                ax3.set_ylabel(f"L{li+1} zoom", fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# Modo --ratio
# ---------------------------------------------------------------------------

def print_ratios(subjs):
    ratios = []
    for subj in subjs:
        try:
            img, lbl = load_pair(subj)
        except FileNotFoundError:
            continue
        r = compute_ratio(img, lbl)
        if r is None:
            print(f"  {subj}: sin lesiones")
            continue
        flag = "✓" if r < 1.0 else "✗"
        print(f"  {flag} {subj}: ratio={r:.3f}")
        ratios.append(r)
    if ratios:
        print(f"\n  Ratio medio : {np.mean(ratios):.3f}  (n={len(ratios)})")
        print(f"  Ratio min   : {np.min(ratios):.3f}")
        print(f"  Ratio max   : {np.max(ratios):.3f}")
        print(f"  Hipointensos: {sum(r < 1.0 for r in ratios)}/{len(ratios)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',     type=int, default=4)
    parser.add_argument('--subj',  type=str, default=None)
    parser.add_argument('--ratio', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    all_subjs = sorted([f.replace('.nii.gz', '')
                        for f in os.listdir(LBLS_DIR) if f.endswith('.nii.gz')])
    if not all_subjs:
        print(f"No se encontraron labels en {LBLS_DIR}")
        return

    if args.ratio:
        print_ratios(all_subjs)
        return

    subjs = [args.subj] if args.subj else random.sample(all_subjs, min(args.n, len(all_subjs)))
    print(f"Sujetos: {subjs}")

    for subj in subjs:
        print(f"  Procesando {subj}...")
        try:
            fig = make_figure(subj)
        except FileNotFoundError as e:
            print(f"    ERROR: {e}"); continue
        if fig is None:
            continue
        out_path = os.path.join(OUT_DIR, f"{subj}_verify.png")
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"    → {out_path}")

    print(f"\nFiguras en: {OUT_DIR}")


if __name__ == '__main__':
    main()
