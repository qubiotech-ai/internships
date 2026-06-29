# -*- coding: utf-8 -*-
"""
MB_generar_graficos_tfm.py
=======================
Generación UNIFICADA de las figuras del TFM (detección de CMBs).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

try:
    import nibabel as nib
    from skimage.measure import label, regionprops
    _NIFTI_OK = True
except Exception:
    _NIFTI_OK = False

import warnings
warnings.filterwarnings("ignore")

import locale
import matplotlib as mpl

# Establishes the numeric locale to Spanish
try:
    locale.setlocale(locale.LC_NUMERIC, 'es_ES.UTF-8')
except:
    # Fallback for systems with alternative locale naming
    locale.setlocale(locale.LC_NUMERIC, 'es_ES')

# Forces matplotlib to use the system locale formatting for axes
mpl.rcParams['axes.formatter.use_locale'] = True


# =====================================================================
# 1) CONFIG  -- única zona a editar
# =====================================================================

RESULTS_BASE = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO/RESULTS"
OUT_DIR      = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/memoria_img/finales_completo2"

MODELS_DICT = {
    "800_results": "D800", "201_results": "D201", "211_results": "D211",
    "203_results": "D203", "202_results": "D202", "205_results": "D205",
    "207_results": "D207", "208_results": "D208",
}

CSV_COMPREHENSIVE = os.path.join(RESULTS_BASE, "comprehensive_lesion_database.csv")

# IMPORTANTE: Forzar a True una vez más para generar la nueva columna Volume_GT_mm3
EXTRAER_DESDE_NIFTI = False
UMBRAL_PRED         = 0.5
INTENSIDAD_COL      = "Mean_Intensity_z"

GT_DIR_GLOBAL = "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset205_MixCMB/labelsTs"
IMAGE_DIR_GLOBAL = "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset205_MixCMB/imagesTs"
PREDICTS_BASE = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO"

NIFTI_EXTRACTION_CONFIG = {
    "D201": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_201_newspacing"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D202": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_202_PROB"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D203": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_203_PROB"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D205": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_205_newspacing"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D207": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_207_PROB_corrected"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D208": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_208_PROB_corrected"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D211": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_211_PROB"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"},
    "D800": {"gt_dir": GT_DIR_GLOBAL, "pred_dir": os.path.join(PREDICTS_BASE, "predicts_from_800_PROB"), "image_dir": IMAGE_DIR_GLOBAL, "gt_suffix": ".nii.gz", "pred_suffix": ".nii.gz", "img_suffix": "_0000.nii.gz"}
}

MODEL_ORDER = ["D800", "D201", "D211", "D203", "D202", "D205", "D207", "D208"]

MODEL_STYLE = {
    "D800": {"color": "#7F7F7F", "marker": "o"},
    "D201": {"color": "#0072B2", "marker": "s"},
    "D211": {"color": "#56B4E9", "marker": "^"},
    "D203": {"color": "#009E73", "marker": "v"},
    "D202": {"color": "#E69F00", "marker": "D"},
    "D205": {"color": "#CC79A7", "marker": "p"},
    "D207": {"color": "#D55E00", "marker": "h"},
    "D208": {"color": "#7A0019", "marker": "P"},
}
MODEL_PALETTE = {m: MODEL_STYLE[m]["color"] for m in MODEL_ORDER}

CAT_ORDER   = ["TP", "FN", "FP"]
CAT_PALETTE = {"TP": "#2E9E5B", "FN": "#D55E00", "FP": "#E6B800"}

DATASET_ORDER = ["ADNI", "VALDO"]
VALDO_COHORTS = ["SABRE", "RSS", "ALFA"]

W_COL  = 3.5
W_TEXT = 7.16

# =====================================================================
# 2) ESTILO GLOBAL
# =====================================================================
def aplicar_estilo_ieee():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size':        9,
        'axes.titlesize':   9,
        'axes.labelsize':   9,
        'legend.fontsize':  8,
        'xtick.labelsize':  8,
        'ytick.labelsize':  8,
        'axes.linewidth':   0.7,
        'grid.linewidth':   0.5,
        'lines.linewidth':  1.0,
        'savefig.dpi':      300,
    })

def _limpiar_ejes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

def _guardar(fig, nombre):
    os.makedirs(OUT_DIR, exist_ok=True)
    ruta = os.path.join(OUT_DIR, nombre)
    fig.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] {ruta}")

def _norm_modelo(serie):
    s = serie.astype(str).str.replace('_results', '', regex=False).str.strip()
    return s.apply(lambda x: f"D{x}" if x.isdigit() else (x if x.startswith('D') else f"D{x}"))

# =====================================================================
# 3) CARGA DE DATOS DE DETECCIÓN (SCAN-LEVEL)
# =====================================================================
def cargar_resultados_deteccion(base=RESULTS_BASE, models_dict=MODELS_DICT):
    alias = {
        'cohort': {'cohort', 'cohorts', 'cohorte'},
        'dataset': {'dataset', 'dataset_name'},
        'f1_score': {'f1', 'f1_score', 'f1-score', 'f1score'},
        'precision': {'precision'}, 'recall': {'recall', 'sensibilidad'},
        'tp_count': {'tp', 'tp_count', 'verdaderos_positivos'},
        'fp_count': {'fp', 'fp_count', 'falsos_positivos'},
        'fn_count': {'fn', 'fn_count', 'falsos_negativos'},
    }

    def _serie(df, col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = df[col]
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

    frames = []
    for folder, modelo in models_dict.items():
        ruta = os.path.join(base, folder, "detailed_results.csv")
        if not os.path.exists(ruta):
            continue
        df = pd.read_csv(ruta)
        df.columns = [str(c).strip().lower() for c in df.columns]
        rename = {c: canon for canon, names in alias.items() for c in df.columns if c in names}
        df = df.rename(columns=rename).loc[:, lambda d: ~d.columns.duplicated()]
        tp, fn = _serie(df, 'tp_count'), _serie(df, 'fn_count')
        frames.append(pd.DataFrame({
            'modelo': modelo,
            'cohort': _serie(df, 'cohort').astype(str).str.strip().str.upper(),
            'dataset': _serie(df, 'dataset').astype(str).str.strip().str.upper(),
            'f1_score': _serie(df, 'f1_score'), 'precision': _serie(df, 'precision'),
            'recall': _serie(df, 'recall'), 'tp_count': tp,
            'fp_count': _serie(df, 'fp_count'), 'fn_count': fn,
            'carga_lesional': tp + fn,
        }))
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    df_all['dataset'] = df_all['dataset'].replace({'VALDO_GLOBAL': 'VALDO', 'VALDO_TOTAL': 'VALDO', 'NAN': np.nan})
    df_all['cohort'] = df_all['cohort'].replace({'VALDO GLOBAL': 'VALDO_TOTAL', 'VALDO': 'VALDO_TOTAL'})
    return df_all

# =====================================================================
# 4) FIGURAS DE DETECCIÓN (SCAN-LEVEL)
# =====================================================================
def fig_f1_fp_por_dataset(df_all, nombre="fig1_f1_fp_por_dataset.png"):
    df = df_all[df_all['dataset'].isin(DATASET_ORDER)].copy()
    if df.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(W_TEXT, 5.0), sharex='col')
    for fila, (col, ylab, ylim) in enumerate(
            [('f1_score', 'F1-Score', (-0.03, 1.03)), ('fp_count', 'FP por scan', None)]):
        for columna, ds in enumerate(DATASET_ORDER):
            ax = axes[fila, columna]
            sns.boxplot(data=df[df['dataset'] == ds], x='modelo', y=col, order=MODEL_ORDER,
                        hue='modelo', palette=MODEL_PALETTE, legend=False, ax=ax,
                        fliersize=2, linewidth=0.8, width=0.7)
            if ylim: ax.set_ylim(*ylim)
            ax.set_xlabel(''); ax.set_ylabel(ylab if columna == 0 else '')
            if fila == 0: ax.set_title(ds, fontweight='bold')
            if fila == 1:
                ax.set_xticks(range(len(MODEL_ORDER)))
                ax.set_xticklabels(MODEL_ORDER, rotation=0, ha='center')
            _limpiar_ejes(ax)
    fig.tight_layout(); _guardar(fig, nombre)


def fig_f1_fp_por_dataset_v2(df_all, nombre="fig1b_f1_fp_por_dataset.png",
                             fp_cap=None):   # fp_cap = {'ADNI': 20, 'VALDO': None}
    df = df_all[df_all['dataset'].isin(DATASET_ORDER)].copy()
    if df.empty: return
    fp_cap = fp_cap or {}
    fig, axes = plt.subplots(2, 2, figsize=(W_TEXT, 3.8), sharex='col')
    for fila, (col, ylab, ylim) in enumerate(
            [('f1_score', 'F1-Score', (-0.03, 1.03)), ('fp_count', 'FP por scan', None)]):
        for columna, ds in enumerate(DATASET_ORDER):
            ax = axes[fila, columna]
            sns.boxplot(data=df[df['dataset'] == ds], x='modelo', y=col, order=MODEL_ORDER,
                        hue='modelo', palette=MODEL_PALETTE, legend=False, ax=ax,
                        fliersize=2, linewidth=0.8, width=0.7)
            if ylim: ax.set_ylim(*ylim)
            # --- recorte SOLO en la fila de FP y SOLO en los modelos marcados ---
            if col == 'fp_count' and fp_cap.get(ds) is not None:
                ax.set_ylim(-0.5, fp_cap[ds])
            ax.set_xlabel(''); ax.set_ylabel(ylab if columna == 0 else '')
            if fila == 0: ax.set_title(ds, fontweight='bold')
            if fila == 1:
                ax.set_xticks(range(len(MODEL_ORDER)))
                ax.set_xticklabels(MODEL_ORDER, rotation=0, ha='center')
            _limpiar_ejes(ax)
    fig.tight_layout(); _guardar(fig, nombre)


def fig_f1_vs_carga(df_all, modelos=('D205', 'D207', 'D208'), nombre="fig2_f1_vs_carga.png"):
    df = df_all[df_all['modelo'].isin(modelos)].dropna(subset=['carga_lesional', 'f1_score'])
    df = df[df['carga_lesional'] > 0]
    if df.empty: return
    cohortes = [c for c in (VALDO_COHORTS + ['ADNI']) if c in df['cohort'].unique()]
    cmap = dict(zip(cohortes, sns.color_palette('Set1', n_colors=max(len(cohortes), 1))))
    fig, axes = plt.subplots(1, len(modelos), figsize=(W_TEXT, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for i, m in enumerate(modelos):
        ax = axes[i]; sub = df[df['modelo'] == m]
        sns.scatterplot(data=sub, x='carga_lesional', y='f1_score', hue='cohort',
                        hue_order=cohortes, palette=cmap, s=28, alpha=0.8, edgecolor='w',
                        linewidth=0.3, ax=ax, legend=(i == len(modelos) - 1))
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['1', '10', '100', '1000'])

        if len(sub) > 2:
            sns.regplot(data=sub, x='carga_lesional', y='f1_score', scatter=False, logx=True,
                        ci=None, ax=ax, line_kws={'linestyle': '--', 'color': 'gray', 'linewidth': 1})
        ax.set_title(m, fontweight='bold'); ax.set_xlabel('Carga lesional (log)')
        ax.set_ylabel('F1-Score' if i == 0 else ''); ax.set_ylim(-0.03, 1.03)
        _limpiar_ejes(ax)
        if i == len(modelos) - 1 and ax.get_legend():
            ax.legend(title='Cohorte', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); _guardar(fig, nombre)

#def _curvas_iso_f1(ax, valores=(0.2, 0.4, 0.6, 0.8)):
    #r = np.linspace(0.001, 1, 400)
    #for f in valores:
        #p = f * r / (2 * r - f)
        #p[(2 * r - f) <= 0] = np.nan; p[p > 1.02] = np.nan
        #ax.plot(r, p, color='0.75', linestyle='--', linewidth=0.7, zorder=0)
        #idx = np.nanargmin(np.abs(p - 0.98)) if np.nanmax(p) >= 0.98 else np.nanargmax(p)
        #if np.isfinite(p[idx]):
            #ax.annotate(f'F1={f:.1f}', xy=(r[idx], p[idx]), fontsize=8, color='0.5', ha='left', va='bottom')

def _curvas_iso_f1(ax, valores=(0.2, 0.4, 0.6, 0.8), f_size=8): # (added by me): parámetro f_size
    r = np.linspace(0.001, 1, 400)
    for i, f in enumerate(valores):
        p = f * r / (2 * r - f)
        p[(2 * r - f) <= 0] = np.nan; p[p > 1.02] = np.nan
        ax.plot(r, p, color='0.75', linestyle='--', linewidth=0.7, zorder=0)
        
        # Se varía ligeramente el punto de anclaje (target_p) según la iteración para separar los textos
        target_p = 0.98 - (i * 0.03)
        idx = np.nanargmin(np.abs(p - target_p)) if np.nanmax(p) >= target_p else np.nanargmax(p)
        if np.isfinite(p[idx]):
            ax.annotate(f'F1={f:.1f}', xy=(r[idx], p[idx]), fontsize=f_size, color='0.5', ha='left', va='bottom')


def fig_pr_tradeoff(df_all, split_datasets=True, nombre="fig3_pr_tradeoff.png"):
    def _micro_exacto(g):
        tp, fp, fn = g['tp_count'].sum(), g['fp_count'].sum(), g['fn_count'].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return pd.Series({'precision': prec, 'recall': rec})

    datasets = [d for d in DATASET_ORDER if d in df_all['dataset'].unique()] if split_datasets else [None]
    fig, axes = plt.subplots(1, len(datasets), figsize=(W_TEXT, 3.3), sharey=True, squeeze=False)
    axes = axes[0]

    for j, ds in enumerate(datasets):
        ax = axes[j]
        df = df_all if ds is None else df_all[df_all['dataset'] == ds]

        met = df.groupby('modelo').apply(_micro_exacto, include_groups=False).reset_index()

        _curvas_iso_f1(ax)
        for _, row in met.iterrows():
            m = row['modelo']
            if m not in MODEL_STYLE: continue
            ax.scatter(row['recall'], row['precision'], color=MODEL_STYLE[m]['color'],
                       marker=MODEL_STYLE[m]['marker'], s=90, edgecolor='black', linewidth=0.5, zorder=5)

        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Sensibilidad')
        ax.set_ylabel('Precisión' if j == 0 else '')
        ax.set_title("Global" if ds is None else ds, fontweight='bold')
        ax.set_aspect('equal', adjustable='box'); _limpiar_ejes(ax)

    handles = [mlines.Line2D([], [], color=MODEL_STYLE[m]['color'], marker=MODEL_STYLE[m]['marker'],
                             linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=m)
               for m in MODEL_ORDER]
    axes[-1].legend(handles=handles, title='Modelo', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', handletextpad=0.3)
    fig.tight_layout(); _guardar(fig, nombre)

def fig_pr_tradeoff_1col_horiz(df_all, split_datasets=True, nombre="fig3_pr_tradeoff_1col_horiz.png"):
    def _micro_exacto(g):
        tp, fp, fn = g['tp_count'].sum(), g['fp_count'].sum(), g['fn_count'].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return pd.Series({'precision': prec, 'recall': rec})

    datasets = [d for d in DATASET_ORDER if d in df_all['dataset'].unique()] if split_datasets else [None]

    fig, axes = plt.subplots(1, len(datasets), figsize=(W_COL, 1.7), sharey=True, squeeze=False, gridspec_kw={'wspace': 0.05})
    axes = axes[0]

    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8}):
        for j, ds in enumerate(datasets):
            ax = axes[j]
            df = df_all if ds is None else df_all[df_all['dataset'] == ds]

            met = df.groupby('modelo').apply(_micro_exacto, include_groups=False).reset_index()

            _curvas_iso_f1(ax, f_size=6) # (added by me): Se reduce el tamaño de fuente a 6
            for _, row in met.iterrows():
                m = row['modelo']
                if m not in MODEL_STYLE: continue
                ax.scatter(row['recall'], row['precision'], color=MODEL_STYLE[m]['color'],
                           marker=MODEL_STYLE[m]['marker'], s=45, edgecolor='black', linewidth=0.5, zorder=5)

            ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)

            ax.set_xticks([0.25, 0.5, 0.75])
            ax.set_xticklabels(['0.25', '0.5', '0.75'])
            if j == 0:
                ax.set_yticks([0.25, 0.5, 0.75])
                ax.set_yticklabels(['0.25', '0.5', '0.75'])

            ax.set_xlabel('Sensibilidad')
            ax.set_ylabel('Precisión' if j == 0 else '')
            
            # Se añade pad=12 para separar el título de los paneles
            ax.set_title("Global" if ds is None else ds, fontweight='bold', pad=12)
            ax.set_aspect('equal', adjustable='box'); _limpiar_ejes(ax)

        handles = [mlines.Line2D([], [], color=MODEL_STYLE[m]['color'], marker=MODEL_STYLE[m]['marker'],
                                 linestyle='None', markersize=6, markeredgecolor='black', markeredgewidth=0.5, label=m)
                   for m in MODEL_ORDER]

        axes[-1].legend(handles=handles, title='Modelo', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', handletextpad=0.3)

        fig.tight_layout(pad=0.2, w_pad=0.1)
        _guardar(fig, nombre)


# =====================================================================
# 5) NIFTI LESION-LEVEL MATCHING & EXTRACTION
# =====================================================================
def _match_lesiones_nifti(gt_mask, pred_mask, z_img, vox_mm3):
    lg, ng = label(gt_mask > 0, return_num=True)
    lp, npred = label(pred_mask > 0, return_num=True)
    gt_reg = {r.label: r for r in regionprops(lg)}
    pr_reg = {r.label: r for r in regionprops(lp)}
    usado, filas = set(), []

    def _get_stats(region, img_z):
        vol = region.area * vox_mm3
        inten = float(img_z[tuple(region.coords.T)].mean()) if img_z is not None else np.nan
        return inten, vol

    for gid in range(1, ng + 1):
        ov = np.unique(lp[lg == gid])
        ov = ov[ov > 0]
        asig = next((p for p in ov if p not in usado), None)
        
        # Guardamos siempre el volumen físico real (GT) para la Fig 4
        _, vol_gt = _get_stats(gt_reg[gid], z_img)
        
        if asig is not None:
            usado.add(asig)
            # Para TP usamos intensidad y volumen PREDICHO en las cajas (Fig 5/6)
            inten_pred, vol_pred = _get_stats(pr_reg[asig], z_img)
            filas.append(("TP", inten_pred, vol_pred, vol_gt))
        else:
            # Para FN no hay predicción, las cajas usan el GT
            inten_gt, _ = _get_stats(gt_reg[gid], z_img)
            filas.append(("FN", inten_gt, vol_gt, vol_gt))

    for pid in range(1, npred + 1):
        if pid not in usado:
            # Para FP no hay GT, el volumen GT se marca como NaN
            inten_fp, vol_fp = _get_stats(pr_reg[pid], z_img)
            filas.append(("FP", inten_fp, vol_fp, np.nan))

    return filas

def extraer_datos_lesiones_desde_nifti(config=NIFTI_EXTRACTION_CONFIG, umbral=UMBRAL_PRED, out_csv=CSV_COMPREHENSIVE):
    if not _NIFTI_OK: return pd.DataFrame()
    if not config: return pd.DataFrame()

    filas = []
    for modelo, cfg in config.items():
        gt_dir, pred_dir, img_dir = cfg.get("gt_dir"), cfg.get("pred_dir"), cfg.get("image_dir")
        gsuf, psuf, isuf = cfg.get("gt_suffix", ".nii.gz"), cfg.get("pred_suffix", ".nii.gz"), cfg.get("img_suffix", "_0000.nii.gz")

        if not (os.path.isdir(gt_dir) and os.path.isdir(pred_dir)):
            continue

        n0 = len(filas)
        for fname in sorted(os.listdir(gt_dir)):
            if not fname.endswith(gsuf): continue
            cid = fname[:-len(gsuf)]

            if cid.startswith("I"):
                dataset = "ADNI"
            elif cid.startswith("sub-"):
                dataset = "VALDO"
            else:
                dataset = "UNKNOWN"

            gt_p = os.path.join(gt_dir, fname)
            pr_p = os.path.join(pred_dir, f"{cid}{psuf}")

            if not os.path.exists(pr_p): continue

            try:
                nii_gt = nib.load(gt_p)
                gt = nii_gt.get_fdata() > 0
                pred = nib.load(pr_p).get_fdata() >= umbral
                vox = float(np.prod(nii_gt.header.get_zooms()[:3]))

                z = None
                if img_dir and os.path.isdir(img_dir):
                    im_p = os.path.join(img_dir, f"{cid}{isuf}")
                    if os.path.exists(im_p):
                        img = nib.load(im_p).get_fdata()
                        brain = img[img > 0]
                        if brain.size > 0 and brain.std() > 0:
                            z = (img - brain.mean()) / brain.std()

                # (added by me): Ahora la función desempaqueta 4 variables
                for cat, inten, vol_box, vol_gt in _match_lesiones_nifti(gt, pred, z, vox):
                    filas.append({
                        "Model": modelo, "Dataset": dataset, "Subject": cid,
                        "Category": cat, "Volume_mm3": vol_box, 
                        "Volume_GT_mm3": vol_gt, "Mean_Intensity_z": inten
                    })
            except Exception as e:
                continue
        print(f"[OK] {modelo}: {len(filas) - n0} lesiones procesadas.")

    df_out = pd.DataFrame(filas)
    if not df_out.empty:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df_out.to_csv(out_csv, index=False)
    return df_out

# =====================================================================
# 6) FIGURAS A NIVEL DE LESIÓN (FORMATO LARGO)
# =====================================================================
def fig_deteccion_vs_volumen(csv_path=CSV_COMPREHENSIVE, nombre="fig4_deteccion_vs_volumen.png", n_bins=15):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    df['Model'] = _norm_modelo(df['Model'])

    df_gt = df[df["Category"].isin(["TP", "FN"])].copy()
    if df_gt.empty: return

    modelos = [m for m in MODEL_ORDER if m in df_gt['Model'].unique()]
    
    # Los bins y el Recall se calculan EXCLUSIVAMENTE sobre Volume_GT_mm3
    bins = np.logspace(np.log10(df_gt["Volume_GT_mm3"].min()), np.log10(df_gt["Volume_GT_mm3"].max()), n_bins + 1)
    centros = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(2, 2, figsize=(W_TEXT, 3.6), sharex='col', gridspec_kw={'height_ratios': [1.4, 1]})

    for col, dominio in enumerate(DATASET_ORDER):
        d_dom = df_gt[df_gt["Dataset"] == dominio]
        top, bot = axes[0, col], axes[1, col]

        for m in modelos:
            d_mod = d_dom[d_dom["Model"] == m]
            tasas = []
            for i in range(len(bins) - 1):
                mask = (d_mod["Volume_GT_mm3"] >= bins[i]) & (d_mod["Volume_GT_mm3"] < bins[i + 1])
                tasas.append((d_mod[mask]["Category"] == "TP").mean() if mask.any() else np.nan)
            top.plot(centros, tasas, marker=MODEL_STYLE[m]['marker'], markersize=4, linewidth=1,
                     color=MODEL_STYLE[m]['color'], alpha=0.9, label=m)

        top.set_xscale('log'); top.set_ylim(-0.05, 1.05)
        top.set_title(f'Límite de detección por volumen - {dominio}', fontweight='bold')
        top.set_ylabel('Sensibilidad' if col == 0 else '')
        _limpiar_ejes(top)
        if col == 1:
            top.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

        modelo_ref = modelos[0]
        d_gt_unico = d_dom[d_dom["Model"] == modelo_ref]
        counts = [((d_gt_unico["Volume_GT_mm3"] >= bins[i]) & (d_gt_unico["Volume_GT_mm3"] < bins[i + 1])).sum() for i in range(len(bins) - 1)]

        bot.bar(centros, counts, width=np.diff(bins), color='0.6', alpha=0.6, edgecolor='gray', linewidth=0.4)
        bot.set_xscale('log')
        bot.set_xticks([1, 10, 100, 1000])
        bot.set_xticklabels(['1', '10', '100', '1000'])

        bot.set_ylabel('Nº lesiones (GT)' if col == 0 else '')
        bot.set_xlabel('Volumen de lesión (mm$^3$, log)')
        _limpiar_ejes(bot)

    fig.tight_layout(); _guardar(fig, nombre)

def _boxplots_2x4_por_categoria(df, valor_col, ylabel, titulo_base, fname_base, log=True):
    df = df.copy()
    df['Model'] = _norm_modelo(df['Model'])
    for dominio in DATASET_ORDER:
        d = df[df["Dataset"] == dominio]
        if d.empty: continue
        modelos = [m for m in MODEL_ORDER if m in d['Model'].unique()]
        fig, axes = plt.subplots(2, 4, figsize=(W_TEXT, 4.2), sharey=True)
        axes = axes.flatten()
        for idx, m in enumerate(modelos):
            ax = axes[idx]
            # Usa la columna Volume_mm3 general (que contiene la predicción para TP/FP y el GT para FN)
            sns.boxplot(data=d[d["Model"] == m], x="Category", y=valor_col, order=CAT_ORDER,
                        hue="Category", palette=CAT_PALETTE, legend=False, ax=ax,
                        fliersize=2, linewidth=0.7, width=0.6)
            if log: 
                ax.set_yscale('log')
                ax.set_yticks([1, 10, 100, 1000]) # añadido para forzar ejes a no tener notación científica
                ax.set_yticklabels(['1', '10', '100', '1000']) # añadido

            ax.set_title(m, fontweight='bold'); ax.set_xlabel('')
            ax.set_ylabel(ylabel if idx % 4 == 0 else '')
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            _limpiar_ejes(ax)
        for k in range(len(modelos), len(axes)):
            axes[k].axis('off')
        fig.suptitle(f'{titulo_base} - {dominio}', fontweight='bold', y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _guardar(fig, f'{fname_base}_{dominio}.png')

def _boxplots_combinados(df, valor_col, ylabel, titulo_base, fname_base, log=True):
    df = df.copy()
    df['Model'] = _norm_modelo(df['Model'])
    
    datasets_apilar = ["ADNI", "VALDO"]
    
    # Se añade layout='constrained' y se aumenta ligeramente el alto a 8.8
    # para que Matplotlib calcule automáticamente el espacio y los títulos no colisionen.
    fig = plt.figure(figsize=(W_TEXT, 8.8), layout='constrained')
    subfigs = fig.subfigures(2, 1, hspace=0.05)
    
    fig.suptitle(titulo_base, fontweight='bold', fontsize=12)
    
    for i, dominio in enumerate(datasets_apilar):
        subfig = subfigs[i]
        subfig.suptitle(dominio, fontweight='bold', fontsize=11)
        d = df[df["Dataset"] == dominio]
        
        if d.empty:
            continue
            
        modelos = [m for m in MODEL_ORDER if m in d['Model'].unique()]
        axes = subfig.subplots(2, 4, sharey=True).flatten()
        
        for idx, m in enumerate(modelos):
            ax = axes[idx]
            sns.boxplot(data=d[d["Model"] == m], x="Category", y=valor_col, order=CAT_ORDER,
                        hue="Category", palette=CAT_PALETTE, legend=False, ax=ax,
                        fliersize=2, linewidth=0.7, width=0.6)
            if log:
                ax.set_yscale('log')
                ax.set_yticks([1, 10, 100, 1000])
                ax.set_yticklabels(['1', '10', '100', '1000'])

            ax.set_title(m, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(ylabel if idx % 4 == 0 else '')
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            _limpiar_ejes(ax)
            
        for k in range(len(modelos), len(axes)):
            axes[k].axis('off')

    _guardar(fig, f'{fname_base}_combinado.png')

def fig_volumen_tp_fn_fp(csv_path=CSV_COMPREHENSIVE, fname_base="fig5_volumen"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    _boxplots_2x4_por_categoria(df, "Volume_mm3", "Volumen (mm$^3$, log)", "Distribución volumétrica", fname_base, log=True)
    _boxplots_combinados(df, "Volume_mm3", "Volumen (mm$^3$, log)", "Distribución volumétrica", fname_base, log=True)


def fig_intensidad_tp_fn_fp(csv_path=CSV_COMPREHENSIVE, fname_base="fig6_intensidad"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    col = INTENSIDAD_COL if INTENSIDAD_COL in df.columns else None
    if col is None or df[col].dropna().empty: return
    _boxplots_2x4_por_categoria(df, col, "Intensidad de lesión (z)", "Intensidad por categoría", fname_base, log=False)
    _boxplots_combinados(df, col, "Intensidad de lesión (z)", "Distribución de Intensidad", fname_base, log=False)


def fig_metrica_vs_carga(df_all, metrica='f1_score', ylabel='F1-Score',
                         modelos=('D205', 'D207', 'D208'),
                         nombre="fig2_metrica_vs_carga.png"):
    df = df_all[df_all['modelo'].isin(modelos)].dropna(subset=['carga_lesional', metrica])
    df = df[df['carga_lesional'] > 0]
    if df.empty: return
    cohortes = [c for c in (VALDO_COHORTS + ['ADNI']) if c in df['cohort'].unique()]
    cmap = dict(zip(cohortes, sns.color_palette('Set1', n_colors=max(len(cohortes), 1))))
    fig, axes = plt.subplots(1, len(modelos), figsize=(W_TEXT, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for i, m in enumerate(modelos):
        ax = axes[i]; sub = df[df['modelo'] == m]
        sns.scatterplot(data=sub, x='carga_lesional', y=metrica, hue='cohort',
                        hue_order=cohortes, palette=cmap, s=28, alpha=0.8, edgecolor='w',
                        linewidth=0.3, ax=ax, legend=(i == len(modelos) - 1))
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1000'])
        if len(sub) > 2:
            sns.regplot(data=sub, x='carga_lesional', y=metrica, scatter=False, logx=True,
                        ci=None, ax=ax, line_kws={'linestyle': '--', 'color': 'gray', 'linewidth': 1})
        ax.set_title(m, fontweight='bold'); ax.set_xlabel('Carga lesional (log)')
        ax.set_ylabel(ylabel if i == 0 else ''); ax.set_ylim(-0.03, 1.03)
        _limpiar_ejes(ax)
        if i == len(modelos) - 1 and ax.get_legend():
            ax.legend(title='Cohorte', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); _guardar(fig, nombre)



def fig_metrica_vs_carga_binned(df_all, metrica='f1_score', ylabel='F1-Score',
                                modelos=('D205', 'D207', 'D208'), umbral=10,
                                nombre="fig2c_metrica_carga_binned.png"):
    df = df_all[df_all['modelo'].isin(modelos)].dropna(subset=['carga_lesional', metrica]).copy()
    df = df[df['carga_lesional'] > 0]
    if df.empty: return
    lo, hi = rf'$<{umbral}$', r'$\geq{umbral}$'
    df['carga_cat'] = np.where(df['carga_lesional'] < umbral, lo, hi)
    print(df.groupby(['modelo', 'carga_cat']).size())   # revisa N por bin
    pal_cat = {lo: '#56B4E9', hi: '#D55E00'}
    fig, ax = plt.subplots(figsize=(W_COL, 2.9))
    sns.boxplot(data=df, x='modelo', y=metrica, order=list(modelos),
                hue='carga_cat', hue_order=[lo, hi], palette=pal_cat, ax=ax,
                fliersize=2, linewidth=0.8, width=0.7)
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel('Modelo'); ax.set_ylabel(ylabel)
    ax.legend(title='Carga lesional', frameon=False)
    _limpiar_ejes(ax)
    fig.tight_layout(); _guardar(fig, nombre)


def fig_metrica_vs_carga_2pendientes(df_all, metrica='f1_score', ylabel='F1-Score',
                                     modelos=('D205', 'D207', 'D208'), umbral=10,
                                     nombre="fig2e_metrica_carga_2slopes.png"):
    df = df_all[df_all['modelo'].isin(modelos)].dropna(subset=['carga_lesional', metrica])
    df = df[df['carga_lesional'] > 0]
    if df.empty: return
    cohortes = [c for c in (VALDO_COHORTS + ['ADNI']) if c in df['cohort'].unique()]
    cmap = dict(zip(cohortes, sns.color_palette('Set1', n_colors=max(len(cohortes), 1))))
    fig, axes = plt.subplots(1, len(modelos), figsize=(W_TEXT, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for i, m in enumerate(modelos):
        ax = axes[i]; sub = df[df['modelo'] == m]
        sns.scatterplot(data=sub, x='carga_lesional', y=metrica, hue='cohort',
                        hue_order=cohortes, palette=cmap, s=28, alpha=0.8, edgecolor='w',
                        linewidth=0.3, ax=ax, legend=(i == len(modelos) - 1))
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1000'])
        izq = sub[sub['carga_lesional'] <  umbral]
        der = sub[sub['carga_lesional'] >= umbral]
        for seg, color in [(izq, '#0072B2'), (der, '#D55E00')]:
            if len(seg) > 2:
                sns.regplot(data=seg, x='carga_lesional', y=metrica, scatter=False,
                            logx=True, ci=None, ax=ax,
                            line_kws={'linestyle': '--', 'color': color, 'linewidth': 1.3})
        ax.axvline(umbral, color='0.7', linestyle=':', linewidth=0.8, zorder=0)
        ax.set_title(m, fontweight='bold'); ax.set_xlabel('Carga lesional (log)')
        ax.set_ylabel(ylabel if i == 0 else ''); ax.set_ylim(-0.03, 1.03)
        _limpiar_ejes(ax)
        if i == len(modelos) - 1 and ax.get_legend():
            ax.legend(title='Cohorte', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); _guardar(fig, nombre)



from scipy.stats import spearmanr

def fig_panel_carga_3x3(df_all, modelos=('D205', 'D207', 'D208'),
                        metricas=(('precision', 'Precisión'),
                                  ('recall',    'Sensibilidad'),
                                  ('f1_score',  'F1-Score')),
                        nombre="fig2g_panel_carga_3x3.png"):
    base = df_all[df_all['modelo'].isin(modelos)].copy()
    base = base[base['carga_lesional'] > 0]
    if base.empty: return
    cohortes = [c for c in (['ADNI'] + VALDO_COHORTS) if c in base['cohort'].unique()]
    cmap = dict(zip(cohortes, sns.color_palette('Set1', n_colors=max(len(cohortes), 1))))

    nfil, ncol = len(metricas), len(modelos)
    
    # Reducido de 2.3 a 1.6 para achatar las filas
    fig, axes = plt.subplots(nfil, ncol, figsize=(W_TEXT, 1.6 * nfil),
                             sharex='col', sharey=True)
    axes = np.atleast_2d(axes)

    for r, (mcol, ylab) in enumerate(metricas):
        for c, m in enumerate(modelos):
            ax = axes[r, c]
            sub = base[base['modelo'] == m].dropna(subset=['carga_lesional', mcol])
            sns.scatterplot(data=sub, x='carga_lesional', y=mcol, hue='cohort',
                            hue_order=cohortes, palette=cmap, s=24, alpha=0.8,
                            edgecolor='w', linewidth=0.3, ax=ax,
                            legend=(r == 0 and c == ncol - 1))
            ax.set_xscale('log')
            ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1000'])
            if len(sub) > 2:
                sns.regplot(data=sub, x='carga_lesional', y=mcol, scatter=False,
                            logx=True, ci=None, ax=ax,
                            line_kws={'linestyle': '--', 'color': 'gray', 'linewidth': 1})
            if r == 0:
                ax.set_title(m, fontweight='bold')
            ax.set_xlabel('Carga lesional (log)' if r == nfil - 1 else '')
            ax.set_ylabel(ylab if c == 0 else '')
            ax.set_ylim(-0.03, 1.03)
            _limpiar_ejes(ax)
            if r == 0 and c == ncol - 1 and ax.get_legend():
                ax.legend(title='Cohorte', frameon=False,
                          bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); _guardar(fig, nombre)


def fig_panel_carga_3x3_2slopes(df_all, umbral=10, modelos=('D205', 'D207', 'D208'),
                                metricas=(('precision', 'Precisión'),
                                          ('recall',    'Sensibilidad'),
                                          ('f1_score',  'F1-Score')),
                                nombre=None):
    if nombre is None:
        nombre = f"fig2h_panel_carga_2slopes_u{umbral}.png"
    base = df_all[df_all['modelo'].isin(modelos)].copy()
    base = base[base['carga_lesional'] > 0]
    if base.empty: return
    cohortes = [c for c in (VALDO_COHORTS + ['ADNI']) if c in base['cohort'].unique()]
    cmap = dict(zip(cohortes, sns.color_palette('Set1', n_colors=max(len(cohortes), 1))))

    nfil, ncol = len(metricas), len(modelos)
    fig, axes = plt.subplots(nfil, ncol, figsize=(W_TEXT, 2.3 * nfil),
                             sharex='col', sharey=True)
    axes = np.atleast_2d(axes)

    for r, (mcol, ylab) in enumerate(metricas):
        for c, m in enumerate(modelos):
            ax = axes[r, c]
            sub = base[base['modelo'] == m].dropna(subset=['carga_lesional', mcol])
            sns.scatterplot(data=sub, x='carga_lesional', y=mcol, hue='cohort',
                            hue_order=cohortes, palette=cmap, s=24, alpha=0.8,
                            edgecolor='w', linewidth=0.3, ax=ax,
                            legend=(r == 0 and c == ncol - 1))
            ax.set_xscale('log')
            ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1000'])
            izq = sub[sub['carga_lesional'] <  umbral]
            der = sub[sub['carga_lesional'] >= umbral]
            for seg, color in [(izq, '#0072B2'), (der, '#D55E00')]:
                if len(seg) > 2:
                    sns.regplot(data=seg, x='carga_lesional', y=mcol, scatter=False,
                                logx=True, ci=None, ax=ax,
                                line_kws={'linestyle': '--', 'color': color, 'linewidth': 1.3})
            ax.axvline(umbral, color='0.7', linestyle=':', linewidth=0.8, zorder=0)
            if r == 0:
                ax.set_title(m, fontweight='bold')
            ax.set_xlabel('Carga lesional (log)' if r == nfil - 1 else '')
            ax.set_ylabel(ylab if c == 0 else '')
            ax.set_ylim(-0.03, 1.03)
            _limpiar_ejes(ax)
            if r == 0 and c == ncol - 1 and ax.get_legend():
                ax.legend(title='Cohorte', frameon=False,
                          bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); _guardar(fig, nombre)


def spearman_carga(df_all, umbral=10, modelos=('D205', 'D207', 'D208'),
                   metricas=('precision', 'recall', 'f1_score')):
    base = df_all[df_all['modelo'].isin(modelos)].copy()
    base = base[base['carga_lesional'] > 0]

    def _sp(d, mc):
        if len(d) < 3:
            return (np.nan, np.nan, len(d))
        rho, p = spearmanr(d['carga_lesional'], d[mc])
        return (rho, p, len(d))

    filas = []
    for m in modelos:
        for mc in metricas:
            sub = base[base['modelo'] == m].dropna(subset=['carga_lesional', mc])
            g  = _sp(sub, mc)
            lo = _sp(sub[sub['carga_lesional'] <  umbral], mc)
            hi = _sp(sub[sub['carga_lesional'] >= umbral], mc)
            filas.append({'modelo': m, 'metrica': mc,
                          'rho_glob': g[0],  'p_glob': g[1],  'n_glob': g[2],
                          f'rho_<{umbral}': lo[0], f'p_<{umbral}': lo[1], f'n_<{umbral}': lo[2],
                          f'rho_>={umbral}': hi[0], f'p_>={umbral}': hi[1], f'n_>={umbral}': hi[2]})
    tab = pd.DataFrame(filas)
    print(tab.round(3).to_string(index=False))
    return tab

def inspeccionar_sujetos_sin_lesiones(df_all, modelos=('D205', 'D207', 'D208')):
    """
    Muestra los sujetos con carga lesional = 0 (sin lesiones GT) para los modelos indicados.
    Útil para decidir si deben entrar en el Spearman de precisión.
    """
    base = df_all[df_all['modelo'].isin(modelos)].copy()
    
    # Carga = 0: tp+fn = 0, es decir, no hay lesiones GT
    sanos = base[base['carga_lesional'] == 0]
    
    if sanos.empty:
        print("No hay sujetos con carga lesional = 0 en estos modelos.")
        return
    
    cols = ['modelo', 'cohort', 'dataset', 'tp_count', 'fp_count', 
            'fn_count', 'carga_lesional', 'precision', 'recall', 'f1_score']
    cols_presentes = [c for c in cols if c in sanos.columns]
    
    print(f"Sujetos con carga lesional = 0: {len(sanos)} filas\n")
    print(sanos[cols_presentes].to_string(index=False))
    
    print("\n--- Resumen por modelo ---")
    for m in modelos:
        sub = sanos[sanos['modelo'] == m]
        if sub.empty:
            continue
        print(f"\n{m}: {len(sub)} sujeto(s) sin lesiones")
        print(f"  FP > 0 (precisión definida y = 0): {(sub['fp_count'] > 0).sum()}")
        print(f"  FP = 0 (precisión indefinida):     {(sub['fp_count'] == 0).sum()}")
        print(f"  Precisión NaN:                     {sub['precision'].isna().sum()}")

# =====================================================================
# 7) MAIN
# =====================================================================
def main():
    aplicar_estilo_ieee()

    df_all = cargar_resultados_deteccion(RESULTS_BASE, MODELS_DICT)
    inspeccionar_sujetos_sin_lesiones(df_all)

    if not df_all.empty:
        fig_f1_fp_por_dataset(df_all)
        fig_f1_fp_por_dataset_v2(df_all, fp_cap={'ADNI': 20, 'VALDO': None})  # ajustar
        fig_f1_vs_carga(df_all)
        fig_pr_tradeoff(df_all, split_datasets=True)
        fig_pr_tradeoff_1col_horiz(df_all, split_datasets=True)

        fig_metrica_vs_carga(df_all, 'recall', 'Sensibilidad', nombre="fig2b_recall_vs_carga.png")
        fig_metrica_vs_carga(df_all, 'precision', 'Precisión', nombre="fig2x_precision_vs_carga.png")
        #fig_metrica_vs_carga_binned(df_all, 'f1_score', 'F1-Score', nombre="fig2c_f1_carga_binned.png")    
        #fig_metrica_vs_carga_binned(df_all, 'recall', 'Sensibilidad', nombre="fig2d_recall_carga_binned.png") 
        fig_metrica_vs_carga_2pendientes(df_all, 'f1_score', 'F1-Score', umbral=10, nombre="fig2e_f1_2slopes.png")
        fig_metrica_vs_carga_2pendientes(df_all, 'recall', 'Sensibilidad', umbral=10, nombre="fig2f_recall_2slopes.png")
        fig_metrica_vs_carga_2pendientes(df_all, 'precision', 'Precisión', umbral=10, nombre="fig2e_precision_2slopes.png")

        fig_panel_carga_3x3(df_all)                                   # (1) single-slope + ρ
        fig_panel_carga_3x3_2slopes(df_all, umbral=10)                # (2) dos pendientes, corte 10
        fig_panel_carga_3x3_2slopes(df_all, umbral=5)                 # (3) dos pendientes, corte 5
        spearman_carga(df_all, umbral=10)                             # tabla para decidir
        spearman_carga(df_all, umbral=5)                              # idem con corte 5

    print(df_all[(df_all.dataset=='ADNI') & (df_all.modelo=='D202')]['fp_count'].sort_values().tail(10))

    if EXTRAER_DESDE_NIFTI:
        extraer_datos_lesiones_desde_nifti()

    fig_deteccion_vs_volumen(CSV_COMPREHENSIVE)
    fig_volumen_tp_fn_fp(CSV_COMPREHENSIVE)
    fig_intensidad_tp_fn_fp(CSV_COMPREHENSIVE)

    print("\nListo. Figuras en:", OUT_DIR)

if __name__ == "__main__":
    main()
