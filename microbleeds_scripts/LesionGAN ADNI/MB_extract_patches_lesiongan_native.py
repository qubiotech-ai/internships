"""
Extracción de Parches para LesionGAN — Espacio Nativo Anisotrópico
===================================================================
CAMBIO FUNDAMENTAL respecto a versiones anteriores:
    NO se resamplea a isotrópico.

    Por qué:
    Las imágenes ADNI tienen resolución 1×1×4mm.
    Al resamplear a 1mm isotrópico, el eje Z se interpolaba ×4,
    haciendo que el generador aprendiera CMBs que ocupaban 6-9 slices
    isotrópicos = 1.5-2.25 slices nativos. Las CMBs reales ocupan
    1-2 slices nativos como máximo.

    Solución: extraer parches directamente en espacio nativo.
    El parche es ANISOTRÓPICO: 16×16×4 voxeles nativos
    → 16×16mm en XY (resolución ~1mm)
    → 16mm en Z  (resolución ~4mm, solo 4 slices)
    Una CMB de 3-5mm en Z ocupa 1-2 slices nativos — correcto.

    Los centroides se calculan en espacio nativo directamente con
    regionprops — no hay transformación de coordenadas necesaria
    porque no hay resampling.

Parches generados:
    PATCH_XY = 16  voxeles en X e Y (~16mm a ~1mm/vox)
    PATCH_Z  = 4   voxeles en Z     (~16mm a ~4mm/vox)
    Shape final: (4, 16, 16) — eje 0 = Z, eje 1 = Y, eje 2 = X

Erosión de máscara D201:
    Radio 1 voxel en XY, 0 en Z (la resolución Z es 4mm, no erosionar)
    Equivale a ~1mm de erosión en XY

Filtro de hipointensidad:
    inside_mean < percentil_25(entorno) — filtro estricto

Paralelización:
    N_WORKERS = 2 hilos CPU

Uso:
    python extract_patches_lesiongan_native.py
"""

import os
import json
import shutil
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from skimage.morphology import disk
from scipy.ndimage import binary_erosion, binary_dilation
from multiprocessing import Pool
from functools import partial

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE      = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw'
D200_IMGS = os.path.join(BASE, 'Dataset200_NoCMB/imagesTr')
D201_IMGS = os.path.join(BASE, 'Dataset201_SyntheticCMB/imagesTr')
D201_LBLS = os.path.join(BASE, 'Dataset201_SyntheticCMB/labelsTr')
D202_IMGS = os.path.join(BASE, 'Dataset202_RealCMB/imagesTr')
D202_LBLS = os.path.join(BASE, 'Dataset202_RealCMB/labelsTr')

OUT_BASE  = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN_Native'
OUT_TRAIN = os.path.join(OUT_BASE, 'train')
OUT_INFER = os.path.join(OUT_BASE, 'infer')

# Tamaño del parche en espacio nativo anisotrópico
PATCH_XY = 16   # voxeles en X e Y (~1mm/vox → ~16mm)
PATCH_Z  = 4    # voxeles en Z     (~4mm/vox → ~16mm)

MIN_CMB_COVERAGE  = 0.7
RANDOM_SEED       = 42
N_WORKERS         = 2   # hilos paralelos — compartimos servidor


# =============================================================================
# UTILIDADES
# =============================================================================

def normalize_patch(patch):
    m = np.mean(patch)
    s = np.std(patch) + 1e-8
    return (patch - m) / s, float(m), float(s)


def extract_patch_native(data, centroid_vox, pad_value=0.0):
    """
    Nibabel: data.shape=(X,Y,Z), centroid=(X,Y,Z), eje 2 = Z a 4mm.
    Output: (PATCH_Z, PATCH_XY, PATCH_XY) transpuesto a (Z,Y,X) para PyTorch.
    """
    cx = int(round(centroid_vox[0]))  # eje X (~1mm)
    cy = int(round(centroid_vox[1]))  # eje Y (~1mm)
    cz = int(round(centroid_vox[2]))  # eje Z (~4mm)

    hxy = PATCH_XY // 2
    hz  = PATCH_Z  // 2

    vx_s = max(0, cx-hxy); vx_e = min(data.shape[0], cx+hxy)
    vy_s = max(0, cy-hxy); vy_e = min(data.shape[1], cy+hxy)
    vz_s = max(0, cz-hz);  vz_e = min(data.shape[2], cz+hz)

    px_s = vx_s-(cx-hxy); px_e = px_s+(vx_e-vx_s)
    py_s = vy_s-(cy-hxy); py_e = py_s+(vy_e-vy_s)
    pz_s = vz_s-(cz-hz);  pz_e = pz_s+(vz_e-vz_s)

    buf = np.full((PATCH_XY, PATCH_XY, PATCH_Z), pad_value, dtype=np.float32)
    buf[px_s:px_e, py_s:py_e, pz_s:pz_e] = data[vx_s:vx_e, vy_s:vy_e, vz_s:vz_e]
    patch = buf.transpose(2, 1, 0)  # (Z, Y, X)

    bbox = {
        'z': [int(cz-hz),  int(cz+hz)],
        'y': [int(cy-hxy), int(cy+hxy)],
        'x': [int(cx-hxy), int(cx+hxy)],
    }
    return patch, bbox


def erode_mask_native(mask):
    """
    Erosión en XY con disco de radio 1 voxel.
    NO se erosiona en Z porque la resolución es 4mm — un voxel Z es muy grueso.
    Se aplica slice a slice en Z.
    """
    struct_xy = disk(1)  # estructura 2D 3×3
    eroded    = mask.copy()
    for z in range(mask.shape[0]):
        if mask[z].sum() > 0:
            # Extender a 3D para scipy (shape 1×N×M)
            sl    = mask[z:z+1].astype(bool)
            st3d  = struct_xy[np.newaxis, :, :]  # (1, 3, 3)
            e_sl  = binary_erosion(sl, structure=st3d)
            eroded[z] = e_sl[0].astype(np.uint8)
    # Si la erosión vacía la máscara, devolver original
    return eroded if eroded.sum() > 0 else mask


def cmb_coverage(lbl_patch, lbl_full):
    total = float(lbl_full.sum())
    return 0.0 if total == 0 else float(lbl_patch.sum()) / total


def is_hypointense(patch, mask_patch):
    """
    Filtro sobre el parche local (no el volumen completo).
    En espacio nativo la máscara ocupa 1-2 voxeles en Z — muy pocos
    para calcular estadísticas robustas sobre el volumen.
    Criterio: intensidad media dentro de la máscara < media fuera.
    """
    if mask_patch.sum() == 0:
        return False
    inside  = patch[mask_patch.astype(bool)].mean()
    outside = patch[~mask_patch.astype(bool)].mean()
    return inside < outside


# =============================================================================
# PROCESADO DE UN SUJETO — PARCHES SANOS + MÁSCARAS (D200 + D201)
# =============================================================================

def process_healthy_subject(args):
    """
    Función top-level para multiprocessing.
    Procesa un sujeto de D201 y extrae parches sanos + máscaras en espacio nativo.
    """
    lbl_file, out_dir, for_inference = args
    subject_id   = lbl_file.replace('.nii.gz', '')
    healthy_path = os.path.join(D200_IMGS,
                                lbl_file.replace('.nii.gz', '_0000.nii.gz'))
    lbl_path     = os.path.join(D201_LBLS, lbl_file)

    if not os.path.exists(healthy_path):
        return [], 0, 1

    patches  = []
    skipped  = 0

    try:
        # Cargar en espacio NATIVO — sin resampling
        healthy_nii = nib.load(healthy_path)
        lbl_nii     = nib.load(lbl_path)

        healthy_data = healthy_nii.get_fdata().astype(np.float32)
        lbl_data     = lbl_nii.get_fdata().astype(np.uint8)

        # Centroide directamente en espacio nativo
        labeled  = label(lbl_data > 0)
        regions  = regionprops(labeled)

        for i, region in enumerate(regions):
            centroid = region.centroid  # (z, y, x) en voxeles nativos

            # Extraer parche sano anisotrópico
            h_patch, bbox = extract_patch_native(
                healthy_data, centroid,
                pad_value=float(np.mean(healthy_data))
            )

            # Extraer parche de máscara
            cmb_mask_full = (labeled == region.label).astype(np.uint8)
            m_patch, _    = extract_patch_native(
                cmb_mask_full, centroid, pad_value=0.0
            )

            if cmb_coverage(m_patch, cmb_mask_full) < MIN_CMB_COVERAGE:
                skipped += 1
                continue

            # Descartar si hay más de una CMB en el parche (evita forma alubia)
            from skimage.measure import label as skl
            if skl(m_patch > 0).max() > 1:
                skipped += 1
                continue

            # Erosión en XY de la máscara del parche
            m_eroded = erode_mask_native(m_patch)

            h_norm, h_mean, h_std = normalize_patch(h_patch)

            name = f"{subject_id}_cmb{i:03d}"
            np.save(os.path.join(out_dir, f"{name}_healthy.npy"),
                    h_norm.astype(np.float32))
            np.save(os.path.join(out_dir, f"{name}_mask.npy"),
                    m_eroded.astype(np.float32))

            meta = {
                'subject_id':    subject_id,
                'cmb_index':     i,
                'split':         'infer' if for_inference else 'train',
                'centroid_vox':  [float(x) for x in centroid],
                'bbox':          bbox,
                'healthy_mean':  h_mean,
                'healthy_std':   h_std,
                'affine_orig':   healthy_nii.affine.tolist(),
                'shape_orig':    list(healthy_data.shape),
                'img_path_orig': healthy_path,
                'lbl_path_orig': lbl_path,
                'vox_size_mm':   list(np.sqrt((healthy_nii.affine[:3,:3]**2)
                                              .sum(axis=0)).tolist()),
            }
            with open(os.path.join(out_dir, f"{name}_meta.json"), 'w') as mf:
                json.dump(meta, mf)

            patches.append(name)

    except Exception as e:
        print(f"  [ERROR] {subject_id}: {e}")
        return [], 0, 1

    return patches, skipped, 0


# =============================================================================
# PROCESADO DE UN SUJETO — PARCHES rCMB REALES (D202)
# =============================================================================

def process_real_subject(args):
    """
    Función top-level para multiprocessing.
    Procesa un sujeto de D202 y extrae parches de rCMBs en espacio nativo.
    """
    lbl_file, out_dir = args
    subject_id = lbl_file.replace('.nii.gz', '')
    img_path   = os.path.join(D202_IMGS,
                              lbl_file.replace('.nii.gz', '_0000.nii.gz'))
    lbl_path   = os.path.join(D202_LBLS, lbl_file)

    if not os.path.exists(img_path):
        return [], 0, 0, 1

    patches         = []
    skipped         = 0
    wrong_intensity = 0

    try:
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)

        img_data = img_nii.get_fdata().astype(np.float32)
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)

        labeled = label(lbl_data > 0)
        regions = regionprops(labeled)

        for i, region in enumerate(regions):
            centroid = region.centroid

            t_patch, _ = extract_patch_native(
                img_data, centroid,
                pad_value=float(np.mean(img_data))
            )

            cmb_mask_full = (labeled == region.label).astype(np.uint8)
            m_patch, _    = extract_patch_native(
                cmb_mask_full, centroid, pad_value=0.0
            )

            if cmb_coverage(m_patch, cmb_mask_full) < MIN_CMB_COVERAGE:
                skipped += 1
                continue

            # Descartar si hay más de una CMB en el parche (evita forma alubia)
            from skimage.measure import label as skl
            if skl(m_patch > 0).max() > 1:
                skipped += 1
                continue

            # Filtro sobre parche local — más robusto en espacio nativo
            if m_patch.sum() == 0:
                skipped += 1
                continue
            if not is_hypointense(t_patch, m_patch):
                wrong_intensity += 1
                continue

            t_norm, t_mean, t_std = normalize_patch(t_patch)

            name = f"{subject_id}_cmb{i:03d}"
            np.save(os.path.join(out_dir, f"{name}_target.npy"),
                    t_norm.astype(np.float32))
            np.save(os.path.join(out_dir, f"{name}_target_mask.npy"),
                    m_patch.astype(np.float32))

            meta = {
                'subject_id':  subject_id,
                'cmb_index':   i,
                'target_mean': float(t_mean),
                'target_std':  float(t_std),
                'centroid_vox': [float(x) for x in centroid],
            }
            with open(os.path.join(out_dir, f"{name}_target_meta.json"), 'w') as mf:
                json.dump(meta, mf)

            patches.append(name)

    except Exception as e:
        print(f"  [ERROR] {subject_id}: {e}")
        return [], 0, 0, 1

    return patches, skipped, wrong_intensity, 0


# =============================================================================
# EXTRACCIÓN PARALELA
# =============================================================================

def extract_healthy_parallel(out_dir, for_inference=False):
    os.makedirs(out_dir, exist_ok=True)
    label_files = sorted([f for f in os.listdir(D201_LBLS)
                          if f.endswith('.nii.gz')])
    args = [(f, out_dir, for_inference) for f in label_files]

    all_patches  = []
    total_skip   = 0
    total_miss   = 0

    print(f"  Procesando {len(label_files)} sujetos con {N_WORKERS} workers...")
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_healthy_subject, args)

    for patches, skip, miss in results:
        all_patches.extend(patches)
        total_skip += skip
        total_miss += miss

    print(f"  Parches sanos+máscara: {len(all_patches)} "
          f"(omitidos: {total_skip}, missing: {total_miss})")
    return all_patches


def extract_real_parallel(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    label_files = sorted([f for f in os.listdir(D202_LBLS)
                          if f.endswith('.nii.gz')])
    args = [(f, out_dir) for f in label_files]

    all_patches     = []
    total_skip      = 0
    total_wrong_int = 0
    total_miss      = 0

    print(f"  Procesando {len(label_files)} sujetos con {N_WORKERS} workers...")
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_real_subject, args)

    for patches, skip, wrong_int, miss in results:
        all_patches.extend(patches)
        total_skip      += skip
        total_wrong_int += wrong_int
        total_miss      += miss

    pct_bad = 100 * total_wrong_int / max(1, len(all_patches) + total_wrong_int)
    print(f"  Parches rCMB reales: {len(all_patches)}")
    print(f"  Descartados cobertura   : {total_skip}")
    print(f"  Descartados no hipoints : {total_wrong_int} ({pct_bad:.1f}%)")
    print(f"  Missing                 : {total_miss}")
    return all_patches


# =============================================================================
# BALANCE
# =============================================================================

def balance_and_pair(healthy, real, out_dir, rng):
    n = min(len(healthy), len(real))
    print(f"\n  Balance:")
    print(f"    Parches sanos  : {len(healthy)}")
    print(f"    Parches reales : {len(real)}")
    if len(healthy) > n:
        healthy = rng.choice(healthy, size=n, replace=False).tolist()
        print(f"    Recortados sanos: {len(healthy) - n} eliminados")
    elif len(real) > n:
        real = rng.choice(real, size=n, replace=False).tolist()
        print(f"    Recortados reales: {len(real) - n} eliminados")
    print(f"    Total pares: {n}")
    rng.shuffle(healthy)
    rng.shuffle(real)
    index = {'healthy': healthy, 'real': real, 'n_pairs': n}
    path  = os.path.join(out_dir, 'train_index.json')
    with open(path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"    Índice: {path}")
    return n


# =============================================================================
# MAIN
# =============================================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Limpiar outputs anteriores
    for d in [OUT_TRAIN, OUT_INFER]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Limpiado: {d}")
    os.makedirs(OUT_TRAIN, exist_ok=True)
    os.makedirs(OUT_INFER,  exist_ok=True)

    print("=" * 60)
    print("EXTRACCIÓN PARCHES LESIONGAN — ESPACIO NATIVO ANISOTRÓPICO")
    print(f"  Parche: {PATCH_Z}×{PATCH_XY}×{PATCH_XY} voxeles nativos")
    print(f"  (~{PATCH_Z*4}mm en Z, ~{PATCH_XY}mm en XY)")
    print(f"  Sin resampling isotrópico — CMBs en espacio real")
    print(f"  Workers: {N_WORKERS}")
    print("=" * 60)

    print("\n[1/4] Extrayendo parches sanos + máscaras (D200 + D201)...")
    healthy = extract_healthy_parallel(OUT_TRAIN, for_inference=False)

    print("\n[2/4] Extrayendo targets rCMB reales (D202)...")
    real = extract_real_parallel(OUT_TRAIN)

    print("\n[3/4] Balanceando...")
    n_pairs = balance_and_pair(healthy, real, OUT_TRAIN, rng)

    print("\n[4/4] Extrayendo parches de inferencia...")
    infer = extract_healthy_parallel(OUT_INFER, for_inference=True)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Pares entrenamiento : {n_pairs}")
    print(f"  Parches inferencia  : {len(infer)}")
    print(f"  Output train        : {OUT_TRAIN}")
    print(f"  Output infer        : {OUT_INFER}")
    print()
    print("  Siguiente paso: actualizar train_lesiongan.py con:")
    print(f"    UNET_TRAIN_DIR = '{OUT_TRAIN}'")
    print(f"    UNET_INFER_DIR = '{OUT_INFER}'")
    print(f"    PATCH_XY = {PATCH_XY}, PATCH_Z = {PATCH_Z}")
    print("=" * 60)


if __name__ == '__main__':
    main()
