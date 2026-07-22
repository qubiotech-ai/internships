from pathlib import Path

import nibabel as nib
import numpy as np

# Constantes

BASE_DIR    = Path(__file__).parent.parent  
PATCHES_DIR = BASE_DIR / "patches"
OUTPUT_DIR  = BASE_DIR / "patches_augmented"

CVS_LABELS  = ["CVS_pos"]
MODALITIES  = ["flair", "swi", "mask"]

# datos de voxel ordenados como (X, Y, Z)
ROTATION_AXES = (0, 1)
VARIANTS = {
    "original": 0,
    "rot90":    1,
    "rot180":   2,
    "rot270":   3,
}

# funciones


def load_patch(patch_path: Path):
   
    img = nib.load(str(patch_path))
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    return data, affine


def rotate_patch(data: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return data
    return np.rot90(data, k=k, axes=ROTATION_AXES)


def save_patch(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine=affine)
    nib.save(img, str(output_path))


def process_lesion(lesion_dir: Path, output_lesion_dir: Path) -> int:
    modality_data = {}
    for modality in MODALITIES:
        patch_path = lesion_dir / f"{modality}.nii.gz"
        if not patch_path.exists():
            raise FileNotFoundError(f"falta {patch_path.name}")
        modality_data[modality] = load_patch(patch_path)

    generated = 0
    for variant_name, k in VARIANTS.items():
        variant_dir = output_lesion_dir / variant_name
        for modality in MODALITIES:
            data, affine = modality_data[modality]
            rotated = rotate_patch(data, k)
            save_patch(rotated, affine, variant_dir / f"{modality}.nii.gz")
        generated += 1

    return generated

# Dataset

def _print_progress(current: int, total: int, label: str) -> None:
    bar_len = 30
    filled = int(bar_len * current / total) if total else bar_len
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {current}/{total}  {label:<40}", end="", flush=True)


def process_dataset(cvs_label: str) -> dict:
    input_dir = PATCHES_DIR / cvs_label
    output_dir = OUTPUT_DIR / cvs_label

    if not input_dir.exists():
        print(f"\n[AVISO] No se encontró la carpeta: {input_dir}")
        return {"total": 0, "ok": 0, "errors": []}

    lesion_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    total = len(lesion_dirs)
    print(f"\n=== Procesando {cvs_label}: {total} lesion(es) encontrada(s) ===")

    ok = 0
    errors = []
    for i, lesion_dir in enumerate(lesion_dirs, start=1):
        lesion_name = lesion_dir.name
        _print_progress(i, total, lesion_name)
        try:
            process_lesion(lesion_dir, output_dir / lesion_name)
            ok += 1
        except Exception as e:
            errors.append((lesion_name, str(e)))
    print()  

    if errors:
        print(f"  [AVISO] {len(errors)} lesion(es) con error en {cvs_label}:")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    else:
        print(f"  Todas las lesiones de {cvs_label} se procesaron correctamente.")

    return {"total": total, "ok": ok, "errors": errors}


def main() -> None:
    print("=" * 60)
    print("Data Augmentation — clasificación CVS")
    print("Estrategia: CVSnet — 3 rotaciones de 90° alrededor del eje axial (Z)")
    print("Sin flips, sin ruido, sin escalados, sin deformaciones elásticas")
    print("Solo se aumenta CVS_pos; CVS_neg se deja sin augmentation")
    print("=" * 60)

    summary = {cvs_label: process_dataset(cvs_label) for cvs_label in CVS_LABELS}

    total_lesions = sum(s["total"] for s in summary.values())
    total_ok = sum(s["ok"] for s in summary.values())
    total_augmented = total_ok * len(VARIANTS)
    total_errors = sum(len(s["errors"]) for s in summary.values())

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Número de lesiones originales : {total_lesions}")
    print(f"Número de lesiones aumentadas : {total_augmented}  "
          f"({total_ok} lesiones x {len(VARIANTS)} variantes)")
    print(f"Errores encontrados            : {total_errors}")

    if total_errors:
        print("\nDetalle de errores:")
        for cvs_label, s in summary.items():
            for name, msg in s["errors"]:
                print(f"  [{cvs_label}] {name}: {msg}")

    print(f"\nParches aumentados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
