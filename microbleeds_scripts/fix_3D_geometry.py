import os
import numpy as np
import nibabel as nib

# --- CONFIGURACIÓN ---
folder_3d_npz = r'Ensemble_Input/3D_Fold0'
# Usamos el 2D SOLO para copiar su cabecera (que sabemos que es la buena)
folder_ref_2d = r'Ensemble_Input/2D_AllFolds'
output_folder = r'Resultados_3D_Corregidos'

# UMBRAL:
# 0.5 -> Veremos si es "Ciego"
# 0.3 -> Veremos si es "Ruidoso"
THRESHOLD = 0.5

os.makedirs(output_folder, exist_ok=True)

print(f"Extrayendo 3D usando la lógica EXACTA del Ensemble (Umbral {THRESHOLD})...")

files = [f for f in os.listdir(folder_3d_npz) if f.endswith('.npz')]

for f in files:
    case_id = f.replace('.npz', '')
    
    try:
        # Rutas
        npz_path = os.path.join(folder_3d_npz, f)
        ref_path = os.path.join(folder_ref_2d, case_id + '.nii.gz')
        
        if not os.path.exists(ref_path):
            print(f"Saltando {case_id}: Falta referencia 2D")
            continue

        # 1. Cargar probabilidad 3D (Clase 1)
        prob_3d = np.load(npz_path)['probabilities'][1]
        
        # 2. Aplicar Umbral
        seg_raw = (prob_3d > THRESHOLD).astype(np.uint8)
        
        # 3. --- LA LÓGICA COPIADA DEL ENSEMBLE ---
        # Si la dimensión Z está al principio (Z, X, Y), la pasamos al final (X, Y, Z)
        # Esta es la línea que hacía que tu ensemble funcionara:
        if seg_raw.shape[0] < seg_raw.shape[1]: 
            seg_final = np.transpose(seg_raw, (2, 1, 0))
        else:
            seg_final = seg_raw
            
        # 4. Guardar con la cabecera del 2D
        img_ref = nib.load(ref_path)
        
        # Verificación final de dimensiones (solo informativo)
        if seg_final.shape != img_ref.shape:
             print(f"⚠️ {case_id}: ¡Dimensiones distintas! {seg_final.shape} vs {img_ref.shape}")
             # Si esto pasa, nnU-Net hizo resampling distinto en 2D y 3D, lo cual sería raro.
        
        img_out = nib.Nifti1Image(seg_final, img_ref.affine, img_ref.header)
        nib.save(img_out, os.path.join(output_folder, f"{case_id}.nii.gz"))
        print(f"Procesado: {case_id}")

    except Exception as e:
        print(f"Error en {case_id}: {e}")

print("¡Hecho!")
