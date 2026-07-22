import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Constantes

BASE_DIR      = Path(__file__).parent.parent    # directorio segmentaciones
CSV_PATH_POS  = BASE_DIR / "lesiones_positivas.csv"
CSV_PATH_NEG  = BASE_DIR / "lesiones_negativas.csv"
PATCHES_DIR   = BASE_DIR / "patches"
PATCH_SIZE    = (28, 28, 28)
HALF_PATCH    = tuple(s // 2 for s in PATCH_SIZE)   # (14, 14, 14)


def get_output_dir(paciente: str, visita: str, lesion_id: str, cvs_label: str = "CVS_pos") -> Path:
    try:
        folder_name = f"{paciente}_{visita}_lesion{int(lesion_id):03d}"
    except ValueError:
        folder_name = f"{paciente}_{visita}_debug"
    return PATCHES_DIR / cvs_label / folder_name


# Funciones

def load_images(dataset: str, paciente: str, visita: str):
   
    patient_dir   = BASE_DIR / dataset / paciente
    resultado_dir = patient_dir / visita

    if not patient_dir.exists():
        raise FileNotFoundError(f"Carpeta de paciente no encontrada: {patient_dir}")
    if not resultado_dir.exists():
        raise FileNotFoundError(f"Carpeta de resultado no encontrada: {resultado_dir}")

    #  FLAIR 
    flair_files = sorted(patient_dir.glob("FLAIR*.nii.gz"))
    if not flair_files:
        raise FileNotFoundError(f"No se encontró FLAIR*.nii.gz en: {patient_dir}")
    if len(flair_files) > 1:
        print(f"  [AVISO] Varios FLAIR encontrados; se usará: {flair_files[0].name}")

    #  Máscara 
    mask_path = resultado_dir / "lesiones_etiquetadas.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"lesiones_etiquetadas.nii.gz no encontrado en: {resultado_dir}"
        )

    #  SWI: buscar registrado primero, luego original 
    # Cubre nombres como: SWI_registered.nii.gz, SWI_*registrada*.nii.gz, etc.
    swi_reg_resultado = sorted(resultado_dir.glob("SWI*registr*.nii.gz"))
    swi_reg_paciente  = sorted(patient_dir.glob("SWI*registr*.nii.gz"))
    swi_original      = sorted(patient_dir.glob("SWI*.nii.gz"))

    if swi_reg_resultado:
        swi_path  = swi_reg_resultado[0]
        swi_label = f"{swi_path.name}  [REGISTRADO — resultado/]"
    elif swi_reg_paciente:
        swi_path  = swi_reg_paciente[0]
        swi_label = f"{swi_path.name}  [REGISTRADO — paciente/]"
    elif swi_original:
        swi_path  = swi_original[0]
        swi_label = f"{swi_path.name}  [ORIGINAL — sin registrar]"
        print(
            "  [AVISO] No se encontró ningún SWI registrado.\n"
            "          Se usará el SWI original; la comprobación de dimensiones\n"
            "          fallará si no está en el mismo espacio que el FLAIR."
        )
    else:
        raise FileNotFoundError(
            f"No se encontró ningún SWI*.nii.gz en:\n"
            f"  {resultado_dir}\n  {patient_dir}\n"
            "Registra el SWI al espacio FLAIR en 3D Slicer y guárdalo con un\n"
            "nombre que contenga 'registr' (p.ej. SWI_registrada.nii.gz)."
        )

    print(f"  FLAIR : {flair_files[0].name}")
    print(f"  SWI   : {swi_label}")
    print(f"  Mask  : {mask_path.name}")

    flair_img = nib.load(str(flair_files[0]))
    swi_img   = nib.load(str(swi_path))
    mask_img  = nib.load(str(mask_path))

    return flair_img, swi_img, mask_img


def check_images(flair_img, swi_img, mask_img) -> None:
   
    shapes = {
        "FLAIR": flair_img.shape,
        "SWI":   swi_img.shape,
        "Mask":  mask_img.shape,
    }
    if len(set(shapes.values())) > 1:
        detail = "\n  ".join(f"{k}: {v}" for k, v in shapes.items())
        raise ValueError(
            f"Las dimensiones de las imágenes no coinciden:\n  {detail}\n"
            "Los tres volúmenes deben estar en el mismo espacio.\n"
            "Registra el SWI al espacio FLAIR en 3D Slicer antes de continuar."
        )
    print(f"  Dimensiones: {flair_img.shape}  —  OK")


def extract_patch(data: np.ndarray, cx: float, cy: float, cz: float):
    
    ix, iy, iz = int(round(cx)), int(round(cy)), int(round(cz))
    x0 = ix - HALF_PATCH[0];  x1 = x0 + PATCH_SIZE[0]
    y0 = iy - HALF_PATCH[1];  y1 = y0 + PATCH_SIZE[1]
    z0 = iz - HALF_PATCH[2];  z1 = z0 + PATCH_SIZE[2]

    nx, ny, nz = data.shape

    if x0 < 0 or y0 < 0 or z0 < 0:
        raise ValueError(
            f"El parche sale fuera de la imagen por el lado inferior.\n"
            f"  Centroide (redondeado): ({ix}, {iy}, {iz})\n"
            f"  Esquina inferior del parche: ({x0}, {y0}, {z0})\n"
            f"  El centroide está demasiado cerca del borde de la imagen."
        )
    if x1 > nx or y1 > ny or z1 > nz:
        raise ValueError(
            f"El parche sale fuera de la imagen por el lado superior.\n"
            f"  Centroide (redondeado): ({ix}, {iy}, {iz})\n"
            f"  Esquina superior del parche: ({x1}, {y1}, {z1})\n"
            f"  Dimensiones de la imagen: ({nx}, {ny}, {nz})\n"
            f"  El centroide está demasiado cerca del borde de la imagen."
        )

    patch = data[x0:x1, y0:y1, z0:z1]

    if patch.shape != PATCH_SIZE:
        raise RuntimeError(
            f"El parche extraído tiene forma {patch.shape}, "
            f"se esperaba {PATCH_SIZE}. Esto no debería ocurrir."
        )

    return patch, (x0, y0, z0)


def save_patch(
    patch: np.ndarray,
    reference_img,
    origin_ijk: tuple,
    output_path: Path,
) -> None:
    
    affine = reference_img.affine.copy()
    origin = np.array(origin_ijk, dtype=float)
    # Desplaza la traslación del affine a la esquina del parche
    affine[:3, 3] = affine[:3, :3] @ origin + affine[:3, 3]

    new_img = nib.Nifti1Image(patch.astype(np.float32), affine=affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_img, str(output_path))
    print(f"  Guardado: {output_path}")



# Main

def process_row(dataset: str, paciente: str, visita: str,
                lesion_id: str, cx: float, cy: float, cz: float,
                fuente: str, cvs_label: str = "CVS_pos") -> bool:
    
    print(f"\n=== Extrayendo parche  [{fuente}] ===")
    print(f"  Dataset  : {dataset}")
    print(f"  Paciente : {paciente}")
    print(f"  Visita   : {visita}")
    print(f"  Lesión   : {lesion_id}")
    print(f"  Centroide: ({cx}, {cy}, {cz})")

    try:
        print("\n[1] Cargando imágenes ...")
        flair_img, swi_img, mask_img = load_images(dataset, paciente, visita)

        print("\n[2] Comprobando dimensiones ...")
        check_images(flair_img, swi_img, mask_img)

        print("\n[3] Extrayendo parches ...")
        flair_data = np.asarray(flair_img.dataobj)
        swi_data   = np.asarray(swi_img.dataobj)
        mask_data  = np.asarray(mask_img.dataobj)

        patch_flair, origin = extract_patch(flair_data, cx, cy, cz)
        patch_swi,   _      = extract_patch(swi_data,   cx, cy, cz)
        patch_mask,  _      = extract_patch(mask_data,  cx, cy, cz)

        print(f"  Forma del parche: {patch_flair.shape}  —  OK")

        print("\n[4] Guardando parches ...")
        out_dir = get_output_dir(paciente, visita, lesion_id, cvs_label)
        save_patch(patch_flair, flair_img, origin, out_dir / "flair.nii.gz")
        save_patch(patch_swi,   swi_img,   origin, out_dir / "swi.nii.gz")
        save_patch(patch_mask,  mask_img,  origin, out_dir / "mask.nii.gz")

        return True

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--row",      type=int,   default=None, help="Índice 0-based de fila en el CSV")
    parser.add_argument("--all",      action="store_true",      help="Procesar todas las filas del CSV")
    parser.add_argument("--neg",      action="store_true",      help="Usar lesiones_negativas.csv (CVS=0) en vez de lesiones_positivas.csv")
    parser.add_argument("--dataset",  type=str,   default=None, help="Ej: ADNI o AIBL")
    parser.add_argument("--paciente", type=str,   default=None, help="Ej: 016_S_6773  (filtro con --all, o modo directo)")
    parser.add_argument("--visita",   type=str,   default=None, help="Ej: resultado_2021-09-24")
    parser.add_argument("--cx",       type=float, default=None, help="Centroide X en vóxeles")
    parser.add_argument("--cy",       type=float, default=None, help="Centroide Y en vóxeles")
    parser.add_argument("--cz",       type=float, default=None, help="Centroide Z en vóxeles")
    args = parser.parse_args()

    cvs_label = "CVS_neg" if args.neg else "CVS_pos"
    csv_path  = CSV_PATH_NEG if args.neg else CSV_PATH_POS

    #  Modo directo 
    direct_args = [args.dataset, args.visita, args.cx, args.cy, args.cz]
    if any(a is not None for a in direct_args):
        missing = [name for name, val in zip(
            ["--dataset", "--paciente", "--visita", "--cx", "--cy", "--cz"],
            [args.dataset, args.paciente, args.visita, args.cx, args.cy, args.cz]
        ) if val is None]
        if missing:
            sys.exit(
                f"[ERROR] Modo directo: faltan los argumentos: {', '.join(missing)}\n"
                "Debes proporcionar todos: --dataset, --paciente, --visita, --cx, --cy, --cz"
            )
        process_row(args.dataset, args.paciente, args.visita,
                    "manual", args.cx, args.cy, args.cz, "argumentos directos",
                    cvs_label)
        return

    # Leer CSV (común a --row y --all) 
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV no encontrado: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    if not all_rows:
        sys.exit(f"[ERROR] {csv_path.name} está vacío.")

    #  Modo --all 
    if args.all:
        rows_to_process = all_rows
        if args.paciente:
            rows_to_process = [r for r in rows_to_process
                               if r["paciente"].strip() == args.paciente]
            if not rows_to_process:
                sys.exit(f"[ERROR] No se encontró el paciente '{args.paciente}' en el CSV.")

        print(f"\nProcesando {len(rows_to_process)} lesión(es) ...")
        ok, fail = 0, 0
        for i, row in enumerate(rows_to_process):
            exito = process_row(
                dataset   = row["dataset"].strip(),
                paciente  = row["paciente"].strip(),
                visita    = row["visita"].strip(),
                lesion_id = row["lesion_id"].strip(),
                cx        = float(row["centroid_x"]),
                cy        = float(row["centroid_y"]),
                cz        = float(row["centroid_z"]),
                fuente    = f"fila {i} — {row['paciente'].strip()} lesión {row['lesion_id'].strip()}",
                cvs_label = cvs_label,
            )
            if exito:
                ok += 1
            else:
                fail += 1

        print(f"\n{'='*50}")
        print(f"Resultado: {ok} OK  |  {fail} con error")
        print(f"Parches guardados en: {PATCHES_DIR / cvs_label}")
        return

    #  Modo --row (por defecto fila 0) 
    row_idx = args.row if args.row is not None else 0
    if row_idx >= len(all_rows):
        sys.exit(
            f"[ERROR] Fila {row_idx} solicitada pero el CSV solo tiene "
            f"{len(all_rows)} filas (índices válidos: 0–{len(all_rows) - 1})."
        )
    row = all_rows[row_idx]
    process_row(
        dataset   = row["dataset"].strip(),
        paciente  = row["paciente"].strip(),
        visita    = row["visita"].strip(),
        lesion_id = row["lesion_id"].strip(),
        cx        = float(row["centroid_x"]),
        cy        = float(row["centroid_y"]),
        cz        = float(row["centroid_z"]),
        fuente    = f"CSV fila {row_idx}",
        cvs_label = cvs_label,
    )


if __name__ == "__main__":
    main()
