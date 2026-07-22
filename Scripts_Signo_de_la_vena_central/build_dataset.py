from __future__ import annotations
import json
import re
from pathlib import Path
# Constantes
BASE_DIR = Path(__file__).parent.parent   # directorio segmentaciones
PATCHES_DIR = BASE_DIR / "patches"
DATASET_JSON_PATH = BASE_DIR / "dataset.json"
# Carpeta
CVS_LABELS = {
    "CVS_pos": 1,
    "CVS_neg": 0,
}
# data_augmentation.py
VARIANT_NAMES = ["original", "rot90", "rot180", "rot270"]
# Modalidades que se usan 
MODALITIES = ["flair", "swi"]

# Tamaño fijo de todos los patches en vóxeles 
PATCH_SIZE = (28, 28, 28)

#   ADNI: dígitos_S_dígitos           -> "016_S_6773"
#   AIBL: P + dígitos                 -> "P1098"
PATIENT_ID_PATTERNS = [
    re.compile(r"^\d+_S_\d+"),
    re.compile(r"^P\d+"),
]

# Funciones

def find_lesion_dirs(cvs_dir: Path) -> list[Path]:
    """
    Devuelve la lista de carpetas de lesión dentro de una carpeta de
    clase (CVS_pos o CVS_neg).
    """
    if not cvs_dir.exists():
        print(f"  [AVISO] No existe la carpeta: {cvs_dir}")
        return []
    return sorted(p for p in cvs_dir.iterdir() if p.is_dir())


def find_sample_dirs(lesion_dir: Path) -> list[tuple[str, Path]]:
    """
    Devuelve la lista de muestras dentro de una carpeta de lesión como
    tuplas (nombre_variante, carpeta_muestra).
    """
    variant_dirs = [(name, lesion_dir / name) for name in VARIANT_NAMES
                    if (lesion_dir / name).is_dir()]
    if variant_dirs:
        return variant_dirs
    return [("original", lesion_dir)]


def extract_patient_id(lesion_folder_name: str) -> str | None:
    """
    Extrae el identificador de paciente a partir del nombre de la
    carpeta de la lesión.
    """
    for pattern in PATIENT_ID_PATTERNS:
        match = pattern.match(lesion_folder_name)
        if match:
            return match.group(0)
    return None


def build_sample_entry(lesion_dir: Path, variant_name: str, sample_dir: Path,
                        label: int) -> dict | None:
    """
    Construye el diccionario MONAI para una única muestra (una carpeta
    que debe contener flair.nii.gz y swi.nii.gz).
    """
    paths = {mod: sample_dir / f"{mod}.nii.gz" for mod in MODALITIES}

    missing = [mod for mod, path in paths.items() if not path.exists()]
    if missing:
        print(f"  [AVISO] Faltan archivos en {sample_dir}: {missing}")
        return None

    patient_id = extract_patient_id(lesion_dir.name)
    if patient_id is None:
        print(f"  [AVISO] No se pudo extraer patient_id de: {lesion_dir.name}")
        return None

    lesion_id = f"{lesion_dir.name}_{variant_name}"

    return {
        "lesion_id": lesion_id,
        "patient_id": patient_id,
        "augmentation": variant_name,
        "flair": paths["flair"].relative_to(BASE_DIR).as_posix(),
        "swi": paths["swi"].relative_to(BASE_DIR).as_posix(),
        "label": label,
        "patch_size": list(PATCH_SIZE),
    }


def process_class(cvs_name: str) -> list[dict]:
    """
    Recorre todas las lesiones de una clase (CVS_pos o CVS_neg) y
    devuelve la lista de entradas MONAI válidas para esa clase.
    """
    cvs_dir = PATCHES_DIR / cvs_name
    label = CVS_LABELS[cvs_name]

    lesion_dirs = find_lesion_dirs(cvs_dir)
    print(f"\n=== {cvs_name}: {len(lesion_dirs)} lesion(es) encontrada(s) ===")

    entries = []
    skipped = 0
    for lesion_dir in lesion_dirs:
        for variant_name, sample_dir in find_sample_dirs(lesion_dir):
            entry = build_sample_entry(lesion_dir, variant_name, sample_dir, label)
            if entry is not None:
                entries.append(entry)
            else:
                skipped += 1

    print(f"  Muestras válidas: {len(entries)}")
    if skipped:
        print(f"  Muestras descartadas por archivos faltantes: {skipped}")

    return entries


def build_dataset() -> list[dict]:
    """
    Construye el dataset completo (CVS_pos + CVS_neg) en formato MONAI.
    """
    dataset = []
    for cvs_name in CVS_LABELS:
        dataset.extend(process_class(cvs_name))
    return dataset


def print_summary(dataset: list[dict]) -> None:
    """
    Muestra por pantalla el resumen del dataset.
    """
    n_pos = sum(1 for d in dataset if d["label"] == 1)
    n_neg = sum(1 for d in dataset if d["label"] == 0)

    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET")
    print("=" * 60)
    print(f"Muestras CVS+ (label=1): {n_pos}")
    print(f"Muestras CVS- (label=0): {n_neg}")
    print(f"Total de muestras      : {len(dataset)}")

    print("\nPrimeros 3 elementos del dataset:")
    for i, entry in enumerate(dataset[:3]):
        print(f"\n  [{i}] {entry}")


def save_dataset_json(dataset: list[dict], output_path: Path = DATASET_JSON_PATH) -> None:
    """
    Guarda la lista de diccionarios MONAI en un archivo JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset guardado en: {output_path}")


def main() -> list[dict]:
    print("=" * 60)
    print("build_dataset.py — construcción de la lista MONAI")
    print("=" * 60)

    dataset = build_dataset()
    print_summary(dataset)
    save_dataset_json(dataset)

    return dataset


if __name__ == "__main__":
    main()
