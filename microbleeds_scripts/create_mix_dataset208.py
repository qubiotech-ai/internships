import os
import shutil
import json
import random

random.seed(42)

# =========================
# PATHS
# =========================
base_path = "/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw"

ds_adni_real = os.path.join(base_path, "Dataset202_RealCMB")
ds_valdo = os.path.join(base_path, "Dataset800_VALDO")

ds_mix = os.path.join(base_path, "Dataset208_RealMixCMB")

subfolders = ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]

# =========================
# CONFIG
# =========================
N_ADNI = 200
N_VALDO = 57

# =========================
def setup():
    for f in subfolders:
        os.makedirs(os.path.join(ds_mix, f), exist_ok=True)

def copy_pair(src, dst, case_image_name):
    """
    case_image_name = ADNI_001_0000.nii.gz
    """

    case_id = case_image_name.replace("_0000.nii.gz", "")

    img_src = os.path.join(src, "imagesTr", case_image_name)
    lbl_src = os.path.join(src, "labelsTr", case_id + ".nii.gz")

    img_dst = os.path.join(dst, "imagesTr", case_image_name)
    lbl_dst = os.path.join(dst, "labelsTr", case_id + ".nii.gz")

    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)

def sample_cases(path, n):
    cases = [f for f in os.listdir(os.path.join(path, "imagesTr")) if f.endswith("_0000.nii.gz")]
    return random.sample(cases, n)

# =========================
if __name__ == "__main__":
    setup()

    # -------- ADNI REAL --------
    adni_cases = sample_cases(ds_adni_real, N_ADNI)
    for c in adni_cases:
        copy_pair(ds_adni_real, ds_mix, c)

    # -------- VALDO REAL --------
    valdo_cases = sample_cases(ds_valdo, N_VALDO)
    for c in valdo_cases:
        copy_pair(ds_valdo, ds_mix, c)

    # =========================
    # TEST (D205 FIXO)
    # =========================
    print("⚠Copiar TEST desde D205 sin cambios")

    # =========================
    # JSON
    # =========================
    train_images = os.listdir(os.path.join(ds_mix, "imagesTr"))
    num_train = len([f for f in train_images if f.endswith(".nii.gz")])

    metadata = {
        "channel_names": {"0": "T2*"},
        "labels": {"background": 0, "CMB": 1},
        "numTraining": num_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithPrecomputedSpacing"
    }

    with open(os.path.join(ds_mix, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("D208 creado correctamente")
