"""
Crea Dataset207_MixCMB (versión corregida) con:
  - 57 VALDO reales (TODAS las de imagesTr del D800)
  - 100 sCMB ADNI sintéticas con mayor carga lesional (excluyendo sCMB=0)
  - 100 rCMB ADNI reales con mayor carga lesional (excluyendo outliers extremos)

Criterio de selección:
  - Sintéticos: excluir sCMB=0, ordenar por carga lesional descendente, top 100
  - Reales: excluir outliers (n_lesiones >= 300), ordenar descendente, top 100
    Justificación: los (2) sujetos con >300 lesiones son casos extremos que podrían
    sesgar el modelo hacia patología severa poco representativa del test set.

Test set: copiado tal cual desde D205 (mismo que en todos los experimentos).
"""


import os
import shutil
import json
import pandas as pd

# =========================
# PATHS
# =========================
base_path = "/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw"

ds_syn       = os.path.join(base_path, "Dataset201_SyntheticCMB")
ds_adni_real = os.path.join(base_path, "Dataset202_RealCMB")
ds_valdo     = os.path.join(base_path, "Dataset800_VALDO")
ds_205       = os.path.join(base_path, "Dataset205_MixCMB")
ds_mix       = os.path.join(base_path, "Dataset207_MixCMB")

CSV_PATH = "/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/ADNI_Master_Unified_Dataset.csv"

subfolders = ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]

N_SYN = 100
N_ADNI = 100

# =========================
# SETUP
# =========================
def setup():
    for f in subfolders:
        os.makedirs(os.path.join(ds_mix, f), exist_ok=True)

# =========================
# COPY TRAIN PAIR
# =========================
def copy_pair(src_base, dst_base, case_image_name):
    case_id = case_image_name.replace("_0000.nii.gz", "")

    img_src = os.path.join(src_base, "imagesTr", case_image_name)
    lbl_src = os.path.join(src_base, "labelsTr", case_id + ".nii.gz")

    img_dst = os.path.join(dst_base, "imagesTr", case_image_name)
    lbl_dst = os.path.join(dst_base, "labelsTr", case_id + ".nii.gz")

    if not os.path.exists(img_src):
        print(f"⚠ missing image: {img_src}")
        return False

    if not os.path.exists(lbl_src):
        print(f"⚠ missing label: {lbl_src}")
        return False

    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)
    return True

# =========================
# VALDO
# =========================
def get_valdo_cases():
    folder = os.path.join(ds_valdo, "imagesTr")
    return sorted([f for f in os.listdir(folder) if f.endswith("_0000.nii.gz")])

# =========================
# SYNTHETIC SELECTION
# =========================
# Carga lesional de los sintéticos según tu log (sCMB=0 excluidos del pool)
SCMB_COUNTS = {
    "SCMB_001": 7,  "SCMB_002": 18, "SCMB_003": 6,  "SCMB_004": 7,
    "SCMB_005": 8,  "SCMB_006": 6,  "SCMB_007": 0,  "SCMB_008": 18,
    "SCMB_009": 12, "SCMB_010": 0,  "SCMB_011": 0,  "SCMB_012": 5,
    "SCMB_013": 3,  "SCMB_014": 29, "SCMB_015": 26, "SCMB_016": 0,
    "SCMB_017": 0,  "SCMB_018": 16, "SCMB_019": 0,  "SCMB_020": 0,
    "SCMB_021": 24, "SCMB_022": 29, "SCMB_023": 2,  "SCMB_024": 6,
    "SCMB_025": 0,  "SCMB_026": 0,  "SCMB_027": 3,  "SCMB_028": 9,
    "SCMB_029": 3,  "SCMB_030": 3,  "SCMB_031": 0,  "SCMB_032": 4,
    "SCMB_033": 5,  "SCMB_034": 0,  "SCMB_035": 30, "SCMB_036": 9,
    "SCMB_037": 4,  "SCMB_038": 25, "SCMB_039": 0,  "SCMB_040": 2,
    "SCMB_041": 8,  "SCMB_042": 5,  "SCMB_043": 9,  "SCMB_044": 6,
    "SCMB_045": 10, "SCMB_046": 0,  "SCMB_047": 6,  "SCMB_048": 1,
    "SCMB_049": 28, "SCMB_050": 17, "SCMB_051": 14, "SCMB_052": 9,
    "SCMB_053": 23, "SCMB_054": 3,  "SCMB_055": 15, "SCMB_056": 0,
    "SCMB_057": 8,  "SCMB_058": 27, "SCMB_059": 23, "SCMB_060": 27,
    "SCMB_061": 0,  "SCMB_062": 3,  "SCMB_063": 5,  "SCMB_064": 22,
    "SCMB_065": 6,  "SCMB_066": 1,  "SCMB_067": 8,  "SCMB_068": 0,
    "SCMB_069": 0,  "SCMB_070": 0,  "SCMB_071": 9,  "SCMB_072": 15,
    "SCMB_073": 21, "SCMB_074": 5,  "SCMB_075": 15, "SCMB_076": 9,
    "SCMB_077": 1,  "SCMB_078": 6,  "SCMB_079": 8,  "SCMB_080": 4,
    "SCMB_081": 23, "SCMB_082": 10, "SCMB_083": 3,  "SCMB_084": 22,
    "SCMB_085": 7,  "SCMB_086": 1,  "SCMB_087": 8,  "SCMB_088": 26,
    "SCMB_089": 20, "SCMB_090": 12, "SCMB_091": 5,  "SCMB_092": 24,
    "SCMB_093": 26, "SCMB_094": 2,  "SCMB_095": 8,  "SCMB_096": 4,
    "SCMB_097": 0,  "SCMB_098": 0,  "SCMB_099": 3,  "SCMB_100": 13,
    "SCMB_101": 3,  "SCMB_102": 3,  "SCMB_103": 14, "SCMB_104": 15,
    "SCMB_105": 0,  "SCMB_106": 10, "SCMB_107": 22, "SCMB_108": 17,
    "SCMB_109": 0,  "SCMB_110": 1,  "SCMB_111": 26, "SCMB_112": 0,
    "SCMB_113": 0,  "SCMB_114": 4,  "SCMB_115": 0,  "SCMB_116": 7,
    "SCMB_117": 22, "SCMB_118": 3,  "SCMB_119": 0,  "SCMB_120": 7,
    "SCMB_121": 0,  "SCMB_122": 25, "SCMB_123": 1,  "SCMB_124": 8,
    "SCMB_125": 4,  "SCMB_126": 10, "SCMB_127": 3,  "SCMB_128": 6,
    "SCMB_129": 1,  "SCMB_130": 11, "SCMB_131": 26, "SCMB_132": 20,
    "SCMB_133": 9,  "SCMB_134": 2,  "SCMB_135": 11, "SCMB_136": 7,
    "SCMB_137": 2,  "SCMB_138": 6,  "SCMB_139": 4,  "SCMB_140": 1,
    "SCMB_141": 8,  "SCMB_142": 0,  "SCMB_143": 0,  "SCMB_144": 1,
    "SCMB_145": 3,  "SCMB_146": 7,  "SCMB_147": 0,  "SCMB_148": 7,
    "SCMB_149": 8,  "SCMB_150": 11, "SCMB_151": 8,  "SCMB_152": 17,
    "SCMB_153": 8,  "SCMB_154": 2,  "SCMB_155": 9,  "SCMB_156": 5,
    "SCMB_157": 10, "SCMB_158": 12, "SCMB_159": 5,  "SCMB_160": 0,
    "SCMB_161": 15, "SCMB_162": 9,  "SCMB_163": 1,  "SCMB_164": 14,
    "SCMB_165": 1,  "SCMB_166": 3,  "SCMB_167": 25, "SCMB_168": 9,
    "SCMB_169": 9,  "SCMB_170": 26, "SCMB_171": 4,  "SCMB_172": 24,
    "SCMB_173": 2,  "SCMB_174": 0,  "SCMB_175": 23, "SCMB_176": 1,
    "SCMB_177": 0,  "SCMB_178": 0,  "SCMB_179": 1,  "SCMB_180": 17,
    "SCMB_181": 0,  "SCMB_182": 0,  "SCMB_183": 10, "SCMB_184": 2,
    "SCMB_185": 0,  "SCMB_186": 0,  "SCMB_187": 1,  "SCMB_188": 22,
    "SCMB_189": 30, "SCMB_190": 25, "SCMB_191": 1,  "SCMB_192": 11,
    "SCMB_193": 10, "SCMB_194": 8,  "SCMB_195": 0,  "SCMB_196": 5,
    "SCMB_197": 0,  "SCMB_198": 0,  "SCMB_199": 3,  "SCMB_200": 4,
}



def select_synthetic_cases(n=100):
    df = pd.DataFrame([
        {"case_id": k, "n_scmb": v, "filename": f"{k}_0000.nii.gz"}
        for k, v in SCMB_COUNTS.items()
        if v > 0
    ])

    df = df.sort_values("n_scmb", ascending=False)

    selected = df.head(n)

    print("\n[SYN]")
    print(f"Pool: {len(df)}")
    print(f"Selected: {len(selected)}")
    print(f"Range: {selected['n_scmb'].min()} - {selected['n_scmb'].max()}")

    return selected["filename"].tolist()

# =========================
# ADNI (CORRECTO)
# =========================
def normalize_id(x):
    return str(x).strip().replace(".0", "")

def select_real_adni_cases(csv_path, images_dir, n=100, max_outlier=300):

    df = pd.read_csv(csv_path)

    # limpiar IDs
    df["LONI_IMG_ID"] = df["LONI_IMG_ID"].apply(normalize_id)

    # contar filas por imagen (robusto a CSV mixto lesion/image-level)
    lesion_count = (
        df.groupby("LONI_IMG_ID")
        .size()
        .reset_index(name="n_lesions")
    )

    lesion_count["filename"] = lesion_count["LONI_IMG_ID"] + "_0000.nii.gz"

    # filtrar existentes
    existing = set(os.listdir(images_dir))
    lesion_count = lesion_count[lesion_count["filename"].isin(existing)]


    print("\n========================")
    print("[ADNI FULL DISTRIBUTION]")
    print("========================")

    # estadísticos globales
    print("\nSTATS:")
    print(lesion_count["n_lesions"].describe())

    # frecuencia exacta de microhemorragias
    freq = lesion_count["n_lesions"].value_counts().sort_index()

    print("\nFREQUENCY TABLE (n_lesions -> count of subjects):")
    print(freq)

    print("\nTOP 20 MOST LESIONAL SUBJECTS:")
    print(
        lesion_count.sort_values("n_lesions", ascending=False)
        .head(20)[["LONI_IMG_ID", "n_lesions"]]
    )

    # SOLO OUTLIERS EXTREMOS
    lesion_count = lesion_count[lesion_count['n_lesions'] <= max_outlier]

    # ordenar por carga lesional
    lesion_count = lesion_count.sort_values('n_lesions', ascending=False)

    selected = lesion_count.head(n)

    print("\n[REAL ADNI FINAL SELECTION]")
    print(f"Total usable: {len(lesion_count)}")
    print(f"Selected: {len(selected)}")
    print(f"Range: {selected['n_lesions'].min()} - {selected['n_lesions'].max()}")

    return selected["filename"].tolist()

# =========================
# TEST SET FIXED
# =========================
def copy_test(ds_205, ds_mix):

    img_dir = os.path.join(ds_205, "imagesTs")
    lbl_dir = os.path.join(ds_205, "labelsTs")

    images = [f for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz")]

    missing_labels = 0

    for img in images:

        img_src = os.path.join(img_dir, img)
        img_dst = os.path.join(ds_mix, "imagesTs", img)

        label = img.replace("_0000.nii.gz", ".nii.gz")
        lbl_src = os.path.join(lbl_dir, label)
        lbl_dst = os.path.join(ds_mix, "labelsTs", label)

        shutil.copy2(img_src, img_dst)

        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)
        else:
            missing_labels += 1

    print(f"\n[TEST]")
    print(f"imagesTs: {len(images)}")
    print(f"missing labels: {missing_labels}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    setup()

    copied = 0

    # -------- VALDO --------
    valdo = get_valdo_cases()
    print(f"\n[VALDO] {len(valdo)} cases")

    for c in valdo:
        if copy_pair(ds_valdo, ds_mix, c):
            copied += 1

    # -------- SYN --------
    syn = select_synthetic_cases(N_SYN)
    print(f"\n[SYN] copying {len(syn)}")

    for c in syn:
        if copy_pair(ds_syn, ds_mix, c):
            copied += 1

    # -------- ADNI --------
    adni = select_real_adni_cases(
        CSV_PATH,
        os.path.join(ds_adni_real, "imagesTr"),
        N_ADNI
    )

    print(f"\n[ADNI] copying {len(adni)}")

    for c in adni:
        if copy_pair(ds_adni_real, ds_mix, c):
            copied += 1

    # -------- TEST --------
    copy_test(ds_205, ds_mix)

    # -------- JSON --------
    train_files = os.listdir(os.path.join(ds_mix, "imagesTr"))

    with open(os.path.join(ds_mix, "dataset.json"), "w") as f:
        json.dump({
            "channel_names": {"0": "T2*"},
            "labels": {"background": 0, "CMB": 1},
            "numTraining": len(train_files),
            "file_ending": ".nii.gz"
        }, f, indent=4)

    # -------- FINAL --------
    print("\n====================")
    print("DATASET 207 DONE")
    print("====================")
    print(f"VALDO: {len(valdo)}")
    print(f"SYN: {len(syn)}")
    print(f"ADNI: {len(adni)}")
    print(f"TOTAL: {len(train_files)}")
