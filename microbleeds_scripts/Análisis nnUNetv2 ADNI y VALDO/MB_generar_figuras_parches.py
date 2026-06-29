# -*- coding: utf-8 -*-
"""
MB_generar_figuras_parches.py
========================
Generacion unificada de las figuras del TFM con estetica homogenea
(tipografia, color, tamano de panel y espaciado identicos entre figuras).

CONVENCION DE TAMANOS  (para que la letra se vea igual en todas, sin huecos)
---------------------------------------------------------------------------
La clave NO es que el panel mida lo mismo, sino que el factor de reduccion en
LaTeX sea identico en todas las figuras (asi la LETRA sale igual, que es lo que
se percibe). Para lograrlo, las figuras de 3 y 4 columnas se generan con anchos
proporcionales a \\columnwidth y \\textwidth:
    - figuras de 4 columnas (PANEL_4) -> ocupan \\textwidth   -> van en figure*
    - figuras de 3 columnas (PANEL_3) -> ocupan \\columnwidth -> van en figure
PANEL_3 = PANEL_4 * 4 * (columnwidth/textwidth) / 3.  Con IEEEtran de doble
columna (textwidth ~= 2.045 * columnwidth) sale PANEL_3 ~= 1.63 in.
En ambos casos basta con \\includegraphics[width=\\linewidth]: dentro de figure
\\linewidth = ancho de columna, y dentro de figure* = ancho de texto, asi que
el mismo comando da el mismo factor de reduccion (~0.72) y la misma letra.
"""

import os
import time
import random
import datetime
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# ==========================================================
# ESTILO GLOBAL  (unica fuente de verdad para la estetica)
# ==========================================================
plt.style.use('seaborn-v0_8-whitegrid'
              if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    # IEEEtran compone el cuerpo en Times -> figuras en serif Times para casar.
    # En Windows 'Times New Roman' existe; el resto son fallbacks por si acaso.
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',     # math con aspecto Times, por coherencia
    'font.size':        9.5,
    'axes.titlesize':   9.5,
    'axes.labelsize':   9.5,
    'savefig.dpi':      300,
})

# Tamanos calibrados MIDIENDO el ancho real del PNG (bbox_inches='tight' lo recorta:
# 4.16 in las de 3 col y 8.23 in las de 4 col) para que la letra salga a ~8 pt impresos
# (= tamano de caption IEEE) al incluirlas con \linewidth en \columnwidth (~3.5 in).
FS_COL_TITLE = 9.5     # titulos de columna (negrita)  -> ~8 pt impresos
FS_ROW_LABEL = 9.5     # etiquetas de fila  (negrita, rotadas)
FS_SUPTITLE  = 11.5    # titulo del modelo             -> ~10 pt impresos (tamano cuerpo)

# Las de 4 columnas se reducen ~2x mas en LaTeX; su fuente nativa se multiplica por esto
# para que la LETRA salga del mismo tamano impreso que en las de 3 columnas:
FS_SCALE_4COL = 1.98   # = 8.23 / 4.16  (ancho real PNG 4col / 3col, ya recortados)

OVERLAY_CMAP  = 'autumn'   # mismo mapa de color para todas las mascaras
OVERLAY_ALPHA = 0.5

# Etiquetas de fila comunes a Figs 1, 3 y pseudomascaras (cambialas aqui una vez)
ROWLBL_IMG  = "Imagen"   # fila de imagen sin anotar
ROWLBL_MASK = "Máscara"    # fila con la mascara superpuesta

# Dos tamanos de panel para que la LETRA salga igual llenando el ancho (ver cabecera).
PANEL_4   = 2.5        # lado de panel para figuras de 4 columnas (-> \textwidth, figure*)
PANEL_3   = 1.63       # lado de panel para figuras de 3 columnas (-> \columnwidth, figure)
WSPACE    = 0.06       # separacion horizontal entre paneles
HSPACE    = 0.10       # separacion vertical entre paneles
TITLE_PAD = 8

# ==========================================================
# RUTAS
# ==========================================================
SAVE_DIR = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/memoria_img/figuras_tfm_letragrande"
os.makedirs(SAVE_DIR, exist_ok=True)

PATHS = {
    "ADNI_REAL":      "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset202_RealCMB",
    "ADNI_SYNTH":     "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset201_SyntheticCMB",
    "ADNI_SYNTH_GAN": "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset203_LesionGAN",
    "VALDO":          "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset800_VALDO",
    "TEST_MIXED":     "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset205_MixCMB",
    "PREDICTS_800":   "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO/predicts_from_800_PROB",
    "PREDICTS_201":   "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO/predicts_from_201_newspacing",
    "PREDICTS_202":   "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO/predicts_from_202_PROB",
    "PREDICTS_203":   "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO/predicts_from_203_PROB",
    "PREDICTS_208":   "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO/predicts_from_208_PROB_corrected",
}

# ==========================================================
# FUNCIONES DE SOPORTE
# ==========================================================
def get_lesion_center(mask_data):
    regions = regionprops(label(mask_data))
    if not regions:
        return None
    largest = max(regions, key=lambda r: r.area)
    return np.array(largest.centroid).astype(int)


def get_lesion_centers(mask_data):
    return [np.array(r.centroid).astype(int) for r in regionprops(label(mask_data))]


def match_blobs(gt_mask, pred_mask):
    """Replica EXACTAMENTE el matching de micro_metrics(): un blob de GT es TP si
    toca (>=1 voxel) un blob de prediccion aun no usado (asignacion 1-a-1, blobs de
    pred ordenados por id ascendente). Devuelve los centroides de cada categoria.
      TP -> centroide del blob GT emparejado
      FN -> centroide del blob GT sin emparejar
      FP -> centroide del blob de prediccion no asignado
    """
    labeled_gt,   num_gt   = label(gt_mask  > 0, return_num=True)
    labeled_pred, num_pred = label(pred_mask > 0, return_num=True)
    gt_reg   = {r.label: r for r in regionprops(labeled_gt)}
    pred_reg = {r.label: r for r in regionprops(labeled_pred)}

    used_pred = set()
    tp_ids, fn_ids = [], []
    for gt_id in range(1, num_gt + 1):
        overlap = np.unique(labeled_pred[labeled_gt == gt_id])
        overlap = overlap[overlap > 0]
        assigned = next((pid for pid in overlap if pid not in used_pred), None)
        if assigned is not None:
            tp_ids.append(gt_id); used_pred.add(assigned)
        else:
            fn_ids.append(gt_id)
    fp_ids = [pid for pid in range(1, num_pred + 1) if pid not in used_pred]

    def center(region):
        return np.round(region.centroid).astype(int)   # round, no truncamiento

    return {
        'TP': [center(gt_reg[i])   for i in tp_ids],
        'FN': [center(gt_reg[i])   for i in fn_ids],
        'FP': [center(pred_reg[i]) for i in fp_ids],
    }


def identify_cohort(case_id):
    if case_id.startswith("sub-1"): return "SABRE"
    if case_id.startswith("sub-2"): return "RSS"
    if case_id.startswith("sub-3"): return "ALFA"
    return "ADNI"


def extract_patch(image, center, window=25, is_mask=False):
    """Parche axial 2D centrado en 'center' (normalizacion percentil [2,98] si imagen)."""
    x, y, z = int(center[0]), int(center[1]), int(center[2])
    z = int(np.clip(z, 0, image.shape[2] - 1))
    x_s, x_e = max(0, x - window), min(image.shape[0], x + window)
    y_s, y_e = max(0, y - window), min(image.shape[1], y + window)
    patch = image[x_s:x_e, y_s:y_e, z]
    if is_mask:
        return np.rot90(patch)
    p_low, p_high = np.percentile(patch, (2, 98))
    if p_high - p_low == 0:
        return np.rot90(np.zeros_like(patch))
    patch = np.clip(patch, p_low, p_high)
    patch = (patch - p_low) / (p_high - p_low)
    return np.rot90(patch)


def find_random_valid_cases(ds_path, n_required=4, seed=None, folders=("labelsTs", "labelsTr")):
    """Devuelve hasta n_required tuplas (case_id, img_path, mask_path, center)
    con al menos una lesion. Por defecto busca en Ts y Tr; pasa folders=("labelsTr",)
    para restringir a entrenamiento."""
    if seed is not None:
        random.seed(seed)
    valid = []
    for folder in folders:
        label_dir = os.path.join(ds_path, folder)
        if not os.path.exists(label_dir):
            continue
        img_folder = "imagesTs" if folder == "labelsTs" else "imagesTr"
        fnames = [f for f in os.listdir(label_dir) if f.endswith(".nii.gz")]
        random.shuffle(fnames)
        for fname in fnames:
            mask_path = os.path.join(label_dir, fname)
            center = get_lesion_center(nib.load(mask_path).get_fdata())
            if center is None:
                continue
            case_id = fname.replace(".nii.gz", "")
            img_path = os.path.join(ds_path, img_folder, f"{case_id}_0000.nii.gz")
            if os.path.exists(img_path):
                valid.append((case_id, img_path, mask_path, center))
                if len(valid) == n_required:
                    return valid
    return valid


def extract_three_slices(img_path, mask_path, center, win=14):
    """3 cortes axiales (cz-1, cz, cz+1) centrados en la lesion, tamano fijo
    (win*2 x win*2) con zero-padding para que el espaciado sea uniforme."""
    img = nib.load(img_path).get_fdata()
    msk = nib.load(mask_path).get_fdata()
    cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
    size = win * 2

    def fixed(data, z):
        out = np.zeros((size, size), dtype=float)
        z = int(np.clip(z, 0, data.shape[2] - 1))
        xs, xe = cx - win, cx + win
        ys, ye = cy - win, cy + win
        ix0, ix1 = max(0, xs), min(data.shape[0], xe)
        iy0, iy1 = max(0, ys), min(data.shape[1], ye)
        out[ix0 - xs:ix0 - xs + (ix1 - ix0),
            iy0 - ys:iy0 - ys + (iy1 - iy0)] = data[ix0:ix1, iy0:iy1, z]
        return out

    slices = []
    for dz in (-1, 0, 1):
        z = int(np.clip(cz + dz, 0, img.shape[2] - 1))
        pi = fixed(img, z)
        nz = pi[pi > 0]
        p2, p98 = (np.percentile(nz, (2, 98)) if nz.size else (0, 1))
        if p98 > p2:
            pi = np.clip((pi - p2) / (p98 - p2), 0, 1)
        slices.append((np.rot90(pi), np.rot90(fixed(msk, z))))
    return slices


# ---- helpers de estilo de ejes (uniforman el aspecto de cada panel) ----
def _blank(ax):
    ax.axis('off')


def _row_label(ax, text, fontsize=FS_ROW_LABEL, labelpad=8):
    """Etiqueta de fila rotada a la izquierda; oculta ticks y spines pero
    mantiene visible el ylabel (sin apagar el eje)."""
    ax.set_ylabel(text, fontsize=fontsize, fontweight='bold',
                  rotation=90, labelpad=labelpad, va='center')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _overlay(ax, img2d, mask2d):
    ax.imshow(img2d, cmap='gray')
    ax.imshow(np.ma.masked_where(mask2d == 0, mask2d),
              cmap=OVERLAY_CMAP, alpha=OVERLAY_ALPHA)


def _save(fig, name):
    out = os.path.join(SAVE_DIR, name)
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] {out}")


# ==========================================================
# FIGURA 1 — VARIABILIDAD DE DOMINIO (2x4)
# Fila 1: imagen; Fila 2: overlay de mascara. Columnas: dominios.
# ==========================================================
def figura_dominios(run_id, seed=None):
    if seed is not None:
        random.seed(seed)

    adni = find_random_valid_cases(PATHS["ADNI_REAL"], n_required=1, folders=("labelsTr",))
    if not adni:
        print("[AVISO] Sin caso ADNI valido."); return
    cid, ip, mp, ctr = adni[0]
    targets = [("ADNI", ip, mp, ctr)]

    cohorts = {"SABRE": None, "RSS": None, "ALFA": None}
    for folder in ("labelsTr",):
        l_dir = os.path.join(PATHS["VALDO"], folder)
        if not os.path.exists(l_dir):
            continue
        img_folder = "imagesTr"
        fnames = [f for f in os.listdir(l_dir) if f.endswith(".nii.gz")]
        random.shuffle(fnames)
        for fname in fnames:
            if all(cohorts.values()):
                break
            center = get_lesion_center(nib.load(os.path.join(l_dir, fname)).get_fdata())
            if center is None:
                continue
            c_id = fname.replace(".nii.gz", "")
            coh = identify_cohort(c_id)
            if coh in cohorts and cohorts[coh] is None:
                i_path = os.path.join(PATHS["VALDO"], img_folder, f"{c_id}_0000.nii.gz")
                if os.path.exists(i_path):
                    cohorts[coh] = (f"VALDO ({coh})", i_path,
                                    os.path.join(l_dir, fname), center)
    for c in ("SABRE", "RSS", "ALFA"):
        if cohorts[c]:
            targets.append(cohorts[c])

    if len(targets) < 4:
        print("[AVISO] No se completaron los 4 dominios."); return

    ncol = 4
    fig, axes = plt.subplots(2, ncol,
                             figsize=(ncol * PANEL_4, 2 * PANEL_4 + 0.4),
                             gridspec_kw={'wspace': WSPACE, 'hspace': HSPACE})
    fs_t = FS_COL_TITLE * FS_SCALE_4COL      # fuentes mayores: esta figura es de 4 col
    fs_r = FS_ROW_LABEL * FS_SCALE_4COL
    lp_r = 8 * FS_SCALE_4COL                 # y separacion etiqueta-panel tambien escalada
    for col, (title, img_p, msk_p, center) in enumerate(targets):
        img = extract_patch(nib.load(img_p).get_fdata(), center, is_mask=False)
        msk = extract_patch(nib.load(msk_p).get_fdata(), center, is_mask=True)
        axes[0, col].imshow(img, cmap='gray')
        # "VALDO (X)" -> dos lineas para que quepa en panel estrecho (1 columna)
        axes[0, col].set_title(title.replace("VALDO (", "VALDO\n("),
                               fontsize=fs_t, fontweight='bold', pad=TITLE_PAD)
        _overlay(axes[1, col], img, msk)
        if col == 0:                         # etiqueta de fila solo en la 1a columna
            _row_label(axes[0, col], ROWLBL_IMG, fontsize=fs_r, labelpad=lp_r)
            _row_label(axes[1, col], ROWLBL_MASK, fontsize=fs_r, labelpad=lp_r)
        else:
            _blank(axes[0, col]); _blank(axes[1, col])

    _save(fig, f"Fig1_Dominios_2x4_{run_id}.png")


# ==========================================================
# FIGURA 3 — COMPARATIVA DE SINTESIS (4x3)
# Columnas: rCMB / sCMB Gauss / sCMB GAN.
# Filas 1 y 3: imagen; filas 2 y 4: overlay. Etiquetas de fila T2*/Mascara en col 0.
# ==========================================================
def figura_sintesis(run_id, seed=None):
    if seed is not None:
        random.seed(seed)

    cols = [
        ("rCMB",       PATHS["ADNI_REAL"]),
        ("sCMB Gauss", PATHS["ADNI_SYNTH"]),
        ("sCMB GAN",   PATHS["ADNI_SYNTH_GAN"]),
    ]
    cases = []
    for title, path in cols:
        c = find_random_valid_cases(path, n_required=2, folders=("labelsTr",))
        if len(c) < 2:
            print(f"[AVISO] '{title}': menos de 2 casos validos en {path}"); return
        cases.append(c)

    ncol = 3
    fig, axes = plt.subplots(4, ncol,
                             figsize=(ncol * PANEL_3, 4 * PANEL_3 + 0.4),
                             gridspec_kw={'wspace': WSPACE, 'hspace': HSPACE})
    for col, ((title, _), case_pair) in enumerate(zip(cols, cases)):
        for k in range(2):                    # k=0 -> filas 0,1 ; k=1 -> filas 2,3
            case_id, img_p, msk_p, center = case_pair[k]
            img = extract_patch(nib.load(img_p).get_fdata(), center, is_mask=False)
            msk = extract_patch(nib.load(msk_p).get_fdata(), center, is_mask=True)
            r_img, r_ovl = k * 2, k * 2 + 1
            axes[r_img, col].imshow(img, cmap='gray')
            _overlay(axes[r_ovl, col], img, msk)
            if col == 0:
                _row_label(axes[r_img, col], ROWLBL_IMG)
                _row_label(axes[r_ovl, col], ROWLBL_MASK)
            else:
                _blank(axes[r_img, col]); _blank(axes[r_ovl, col])
        axes[0, col].set_title(title, fontsize=FS_COL_TITLE, fontweight='bold', pad=TITLE_PAD)

    _save(fig, f"Fig3_Sintesis_4x3_{run_id}.png")


# ==========================================================
# FIGURAS DE ANALISIS DE ERROR (4x3)  -- una por modelo
# Filas: cohortes (etiqueta rotada). Columnas: TP / FP / FN.
# ==========================================================
def figura_error_analysis(predicts_path, ground_truth_path, model_name, run_id, seed=None):
    if seed is not None:
        random.seed(seed)

    ROW_LABELS = {0: "ADNI", 1: "VALDO (SABRE)", 2: "VALDO (RSS)", 3: "VALDO (ALFA)"}
    COHORT_ROW = {"ADNI": 0, "SABRE": 1, "RSS": 2, "ALFA": 3}
    COL_TITLES = ["TP", "FP", "FN"]   # se desarrollan en el pie de figura
    cat_by_col = {0: 'TP', 1: 'FP', 2: 'FN'}

    slots = {'TP': {r: None for r in range(4)},
             'FP': {r: None for r in range(4)},
             'FN': {r: None for r in range(4)}}

    files = []
    for folder in ("labelsTs", "labelsTr"):
        d = os.path.join(ground_truth_path, folder)
        if os.path.exists(d):
            files += [(folder, x) for x in os.listdir(d) if x.endswith(".nii.gz")]
    random.shuffle(files)

    for folder, fname in files:
        if all(all(s[r] is not None for r in range(4)) for s in slots.values()):
            break
        case_id = fname.replace(".nii.gz", "")
        img_folder = "imagesTs" if folder == "labelsTs" else "imagesTr"
        img_p = os.path.join(ground_truth_path, img_folder, f"{case_id}_0000.nii.gz")
        pred_p = os.path.join(predicts_path, fname)
        if not (os.path.exists(img_p) and os.path.exists(pred_p)):
            continue
        row = COHORT_ROW.get(identify_cohort(case_id))
        if row is None:
            continue
        mask = nib.load(os.path.join(ground_truth_path, folder, fname)).get_fdata()
        pred = nib.load(pred_p).get_fdata()
        cats = match_blobs(mask, pred)          # TP/FP/FN por solapamiento (= micro_metrics)
        img = None                              # se carga solo si hay algo que mostrar
        for cat in ('TP', 'FP', 'FN'):
            if slots[cat][row] is None and cats[cat]:
                if img is None:
                    img = nib.load(img_p).get_fdata()
                slots[cat][row] = (img, random.choice(cats[cat]))

    ncol = 3
    fig, axes = plt.subplots(4, ncol,
                             figsize=(ncol * PANEL_3, 4 * PANEL_3 + 0.3),
                             gridspec_kw={'wspace': WSPACE, 'hspace': HSPACE})
    fig.subplots_adjust(top=0.90)          # sube los paneles hacia el titulo
    if model_name:
        # y mas bajo = titulo mas pegado a las imagenes (ajustable)
        fig.suptitle(f"Modelo {model_name}", fontsize=FS_SUPTITLE, fontweight='bold', y=0.965)

    for col in range(ncol):
        cat = cat_by_col[col]
        for row in range(4):
            ax = axes[row, col]
            data = slots[cat][row]
            if data is not None:
                im, cent = data
                ax.imshow(extract_patch(im, cent, is_mask=False), cmap='gray')
            else:
                ax.text(0.5, 0.5, "No disp.", ha='center', va='center',
                        color='gray', fontsize=FS_ROW_LABEL)
            if col == 0:
                _row_label(ax, ROW_LABELS[row])
            else:
                _blank(ax)
        axes[0, col].set_title(COL_TITLES[col], fontsize=FS_COL_TITLE,
                               fontweight='bold', pad=TITLE_PAD)

    _save(fig, f"FigError_{model_name}_4x3_{run_id}.png")


# ==========================================================
# FIGURA APENDICE — PSEUDOMASCARAS ADNI (D202) (6x3)
# Columnas: Corte anterior / central / posterior.
# Filas 1,3,5: imagen real; filas 2,4,6: overlay. Etiquetas de fila T2*/Mascara en col 0.
# ==========================================================
def figura_pseudomascaras(run_id, seed=42):
    random.seed(seed)
    cases = find_random_valid_cases(PATHS["ADNI_REAL"], n_required=3, folders=("labelsTr",))
    if len(cases) < 3:
        print("[AVISO] Menos de 3 casos ADNI validos."); return

    COL_TITLES = ["Corte anterior", "Corte central", "Corte posterior"]
    ncol = 3
    fig, axes = plt.subplots(6, ncol,
                             figsize=(ncol * PANEL_3, 6 * PANEL_3 + 0.4),
                             gridspec_kw={'wspace': WSPACE, 'hspace': HSPACE})
    for ci, (case_id, img_p, msk_p, center) in enumerate(cases):
        slices = extract_three_slices(img_p, msk_p, center, win=14)
        r_img, r_ovl = ci * 2, ci * 2 + 1
        for col, (pi, pm) in enumerate(slices):
            axes[r_img, col].imshow(pi, cmap='gray')
            _overlay(axes[r_ovl, col], pi, pm)
            if col == 0:
                _row_label(axes[r_img, col], ROWLBL_IMG)
                _row_label(axes[r_ovl, col], ROWLBL_MASK)
            else:
                _blank(axes[r_img, col]); _blank(axes[r_ovl, col])
            if ci == 0:
                axes[0, col].set_title(COL_TITLES[col], fontsize=FS_COL_TITLE,
                                       fontweight='bold', pad=TITLE_PAD)

    _save(fig, f"FigApx_Pseudomascaras_6x3_{run_id}.png")


# ==========================================================
# FIGURA SEGMENTACIONES — HEAD TO HEAD (5x4, solo TP)
# Filas: 2 ADNI + 3 VALDO (etiqueta rotada).
# Columnas: Ground Truth (cursiva) / D201 / D203 / D208.
# Fondo = parche GT; overlay = mascara de cada modelo en el mismo centro.
# ==========================================================
def figura_segmentaciones(run_id, seed=None):
    if seed is not None:
        random.seed(seed)

    COL_TITLES = ["GT", "D201", "D203", "D208"]   # GT = Ground Truth (desarrollar en pie)
    MODEL_KEYS = ["PREDICTS_201", "PREDICTS_203", "PREDICTS_208"]
    target_rows = [("ADNI", 0), ("ADNI", 1), ("SABRE", 2), ("RSS", 3), ("ALFA", 4)]
    # Etiquetas de fila cortas: en 4 col / 1 columna el panel (~0.9 in) no admite "VALDO (X)".
    ROW_LABELS = {0: "ADNI", 1: "ADNI", 2: "SABRE", 3: "RSS", 4: "ALFA"}

    gt_dir = os.path.join(PATHS["TEST_MIXED"], "labelsTs")
    if not os.path.exists(gt_dir):
        print(f"[ERROR] No existe {gt_dir}"); return
    files = [f for f in os.listdir(gt_dir) if f.endswith(".nii.gz")]
    random.shuffle(files)

    needed_by_cohort = {}
    for coh, r in target_rows:
        needed_by_cohort.setdefault(coh, []).append(r)

    filled = {}
    for fname in files:
        if len(filled) == len(target_rows):
            break
        case_id = fname.replace(".nii.gz", "")
        coh = identify_cohort(case_id)
        rows_free = [r for r in needed_by_cohort.get(coh, []) if r not in filled]
        if not rows_free:
            continue
        img_p = os.path.join(PATHS["TEST_MIXED"], "imagesTs", f"{case_id}_0000.nii.gz")
        preds_p = {k: os.path.join(PATHS[k], fname) for k in MODEL_KEYS}
        if not (os.path.exists(img_p) and all(os.path.exists(p) for p in preds_p.values())):
            continue
        gt = nib.load(os.path.join(gt_dir, fname)).get_fdata()
        centers = get_lesion_centers(gt)
        if not centers:
            continue
        filled[rows_free[0]] = {
            "case_id": case_id,
            "img": nib.load(img_p).get_fdata(),
            "gt": gt,
            "center": random.choice(centers),
            "preds": {k: nib.load(p).get_fdata() for k, p in preds_p.items()},
        }

    if len(filled) < len(target_rows):
        print(f"[AVISO] Solo se llenaron {len(filled)}/{len(target_rows)} filas "
              f"(revisa que existan predicciones para esos casos).")

    nrow, ncol = len(target_rows), 4
    fs_t = FS_COL_TITLE * FS_SCALE_4COL      # fuentes mayores: figura de 4 col
    fs_r = FS_ROW_LABEL * FS_SCALE_4COL
    lp_r = 8 * FS_SCALE_4COL                 # separacion etiqueta-panel tambien escalada
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * PANEL_4, nrow * PANEL_4 + 0.5),
                             gridspec_kw={'wspace': WSPACE, 'hspace': HSPACE})
    for r in range(nrow):
        data = filled.get(r)
        for col in range(ncol):
            ax = axes[r, col]
            if data is None:
                ax.text(0.5, 0.5, "No disp.", ha='center', va='center',
                        color='gray', fontsize=fs_r)
            else:
                img2d = extract_patch(data["img"], data["center"], is_mask=False)
                ax.imshow(img2d, cmap='gray')
                if col == 0:
                    m = extract_patch(data["gt"], data["center"], is_mask=True)
                else:
                    m = extract_patch(data["preds"][MODEL_KEYS[col - 1]],
                                      data["center"], is_mask=True)
                ax.imshow(np.ma.masked_where(m == 0, m), cmap=OVERLAY_CMAP, alpha=OVERLAY_ALPHA)
            if col == 0:
                _row_label(ax, ROW_LABELS[r], fontsize=fs_r, labelpad=lp_r)
            else:
                _blank(ax)

    for col, t in enumerate(COL_TITLES):
        style = 'italic' if col == 0 else 'normal'
        axes[0, col].set_title(t, fontsize=fs_t, fontweight='bold',
                               fontstyle=style, pad=TITLE_PAD)

    _save(fig, f"FigSeg_HeadToHead_5x4_{run_id}.png")


# ==========================================================
# BUCLE DE EJECUCION
# ==========================================================
if __name__ == "__main__":
    NUM_EJEMPLOS = 5   # nº de muestreos aleatorios distintos por figura

    for i in range(1, NUM_EJEMPLOS + 1):
        run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v{i}"
        print(f"\n--- Lote {i}/{NUM_EJEMPLOS} ({run_id}) ---")

        #figura_dominios(run_id)
        figura_sintesis(run_id)
        # figura_pseudomascaras(run_id, seed=42 + i)
        # figura_segmentaciones(run_id)

        # Matrices de error: descomenta el/los modelo(s) que vayas a incluir.
        # IMPORTANTE: el ground_truth SIEMPRE es el test mixto (Dataset205), porque la
        # funcion elige los casos recorriendo el GT y solo ahi conviven ADNI (I...) y VALDO.
        # Si pasas un GT que no tiene ADNI (p.ej. VALDO), la fila ADNI sale vacia aunque
        # existan las predicciones de los I... en la carpeta de predicciones.
        # figura_error_analysis(PATHS["PREDICTS_208"], PATHS["TEST_MIXED"], "D208", run_id)
        # figura_error_analysis(PATHS["PREDICTS_201"], PATHS["TEST_MIXED"], "D201", run_id)
        # figura_error_analysis(PATHS["PREDICTS_203"], PATHS["TEST_MIXED"], "D203", run_id)
        # figura_error_analysis(PATHS["PREDICTS_800"], PATHS["TEST_MIXED"], "D800", run_id)
        # figura_error_analysis(PATHS["PREDICTS_202"], PATHS["TEST_MIXED"], "D202", run_id)

        time.sleep(1)

    print("\nProceso finalizado.")