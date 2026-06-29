"""
analyze_tp_intersection.py
==========================
Analiza la intersección de Verdaderos Positivos (TP) entre el modelo D201 (Gaussiano)
y el modelo D203 (LesionGAN) a nivel de lesión individual (blob).
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.measure import label

# ========================= CONFIGURACIÓN DE RUTAS =========================
BASE_DIR = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/PREDICTS_ON_VALDO"
GT_DIR   = "/media/PORT-DISK/Practicas/nnUNet_raw_ADNI/Dataset205_MixCMB/labelsTs"

PRED_201_DIR = os.path.join(BASE_DIR, "predicts_from_201_newspacing")
PRED_203_DIR = os.path.join(BASE_DIR, "predicts_from_203_PROB")
OUT_CSV      = os.path.join(BASE_DIR, "RESULTS", "tp_intersection_201_vs_203.csv")

def identify_cohort(fname):
    fname_upper = fname.upper()
    if fname_upper.startswith("SUB-1"): return "SABRE"
    elif fname_upper.startswith("SUB-2"): return "RSS"
    elif fname_upper.startswith("SUB-3"): return "ALFA"
    elif fname_upper.startswith("I") or "ADNI" in fname_upper: return "ADNI"
    return "UNKNOWN"

def main():
    print("Iniciando análisis de intersección de TP (D201 vs D203)...")
    
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".nii.gz")])
    
    stats = {
        "Total_GT_Lesions": 0,
        "Detected_Both": 0,
        "Detected_Only_D201": 0,
        "Detected_Only_D203": 0,
        "Missed_By_Both": 0
    }
    
    records = []

    for fname in gt_files:
        gt_path = os.path.join(GT_DIR, fname)
        p201_path = os.path.join(PRED_201_DIR, fname)
        p203_path = os.path.join(PRED_203_DIR, fname)
        
        if not os.path.exists(p201_path) or not os.path.exists(p203_path):
            continue
            
        try:
            gt_data = nib.load(gt_path).get_fdata()
            p201_data = nib.load(p201_path).get_fdata() > 0
            p203_data = nib.load(p203_path).get_fdata() > 0
        except Exception as e:
            print(f"  [ERROR] Fallo al leer {fname}: {e}")
            continue

        labeled_gt, num_gt = label(gt_data > 0, return_num=True)
        cohort = identify_cohort(fname)
        
        stats["Total_GT_Lesions"] += num_gt

        for gt_id in range(1, num_gt + 1):
            gt_mask = (labeled_gt == gt_id)
            
            # Verificar solapamiento para cada modelo
            hit_201 = np.any(p201_data[gt_mask])
            hit_203 = np.any(p203_data[gt_mask])
            
            status = "Missed"
            if hit_201 and hit_203:
                stats["Detected_Both"] += 1
                status = "Both"
            elif hit_201 and not hit_203:
                stats["Detected_Only_D201"] += 1
                status = "Only_201"
            elif not hit_201 and hit_203:
                stats["Detected_Only_D203"] += 1
                status = "Only_203"
            else:
                stats["Missed_By_Both"] += 1
                
            records.append({
                "Subject": fname,
                "Cohort": cohort,
                "Lesion_ID": gt_id,
                "Detection_Status": status
            })

    # Guardar detalle
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    
    # Calcular totales derivados
    total_tp_201 = stats["Detected_Both"] + stats["Detected_Only_D201"]
    total_tp_203 = stats["Detected_Both"] + stats["Detected_Only_D203"]
    total_detected_any = stats["Detected_Both"] + stats["Detected_Only_D201"] + stats["Detected_Only_D203"]
    
    print("\n" + "="*50)
    print(" RESUMEN DE INTERSECCIÓN (D201 vs D203)")
    print("="*50)
    print(f"Total Lesiones Reales (GT):  {stats['Total_GT_Lesions']}")
    print(f"Detectadas por algún modelo: {total_detected_any}")
    print("-" * 50)
    print(f"TP Totales D201 (Gauss):     {total_tp_201}")
    print(f"TP Totales D203 (GAN):       {total_tp_203}")
    print("-" * 50)
    print(f"Detectadas por AMBOS:        {stats['Detected_Both']} "
          f"({(stats['Detected_Both']/max(1, total_detected_any))*100:.1f}% del total detectado)")
    print(f"Exclusivas D201 (Gauss):     {stats['Detected_Only_D201']}")
    print(f"Exclusivas D203 (GAN):       {stats['Detected_Only_D203']}")
    print(f"Omitidas por ambos (FN):     {stats['Missed_By_Both']}")
    print("="*50)
    print(f"Detalle exportado a: {OUT_CSV}")

if __name__ == "__main__":
    main()
