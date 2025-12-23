import os
import shutil
import random
import nibabel as nib
import glob

# --- CONFIGURACIÓN ---
base_dir = "/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/nnUNet_raw/Dataset888_VALDO"
imagesTr = os.path.join(base_dir, "imagesTr")
labelsTr = os.path.join(base_dir, "labelsTr")

# Salida
output_dir = "Test_Subset_Ensemble"
imagesTs_out = os.path.join(output_dir, "imagesTs")
labelsTs_out = os.path.join(output_dir, "labelsTs")

# Ratio y Semilla
ratio = 0.20
seed = 42

# ---------------------

def get_z_spacing(nii_path):
    try:
        img = nib.load(nii_path)
        return img.header.get_zooms()[2]
    except:
        return -1

def classify_cohort(z_spacing):
    if 0.7 <= z_spacing <= 0.9: return "Cohorte 2 (0.8mm)"
    elif 2.8 <= z_spacing <= 3.2: return "Cohorte 3 (3.0mm)"
    elif 3.8 <= z_spacing <= 4.2: return "Cohorte 1 (4.0mm)"
    else: return "Otros"

# 1. Limpiar y crear carpetas
if os.path.exists(output_dir):
    print(f"Borrando versión anterior de {output_dir}...")
    shutil.rmtree(output_dir)
os.makedirs(imagesTs_out)
os.makedirs(labelsTs_out)

# 2. Identificar PACIENTES ÚNICOS (Basándonos en las etiquetas)
# En nnU-Net, labelsTr tiene 1 archivo por paciente: "sub-XXX.nii.gz"
label_files = [f for f in os.listdir(labelsTr) if f.endswith(".nii.gz")]
print(f"Total pacientes encontrados en labelsTr: {len(label_files)}")

# 3. Clasificar Pacientes
cohorts = {"Cohorte 2 (0.8mm)": [], "Cohorte 3 (3.0mm)": [], "Cohorte 1 (4.0mm)": [], "Otros": []}

print("Clasificando cohortes según resolución Z...")
for lbl_file in label_files:
    case_id = lbl_file.replace(".nii.gz", "") # ej: sub-101
    
    # Buscamos el canal 0000 para mirar el header (asumimos que todos los canales tienen misma geometria)
    img_0000_path = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
    
    if not os.path.exists(img_0000_path):
        print(f"AVISO: No encuentro imagen base para {case_id}")
        continue
        
    z = get_z_spacing(img_0000_path)
    c = classify_cohort(z)
    cohorts[c].append(case_id)

# 4. Selección y Copia (PACK COMPLETO)
random.seed(seed)
total_selected = 0

print(f"\n--- SELECCIÓN ESTRATIFICADA (Ratio: {ratio*100}%) ---")

for name, patient_list in cohorts.items():
    n_total = len(patient_list)
    if n_total == 0: continue
    
    n_select = max(1, int(n_total * ratio))
    selected_ids = random.sample(patient_list, n_select)
    total_selected += n_select
    
    print(f"[{name}]: {n_total} casos -> Seleccionamos {n_select}")
    
    for case_id in selected_ids:
        # A. Copiar etiqueta
        shutil.copy(os.path.join(labelsTr, f"{case_id}.nii.gz"), 
                    os.path.join(labelsTs_out, f"{case_id}.nii.gz"))
        
        # B. Copiar TODAS las imágenes asociadas (0000, 0001, 0002...)
        # Buscamos cualquier archivo que empiece por case_id + "_"
        associated_images = glob.glob(os.path.join(imagesTr, f"{case_id}_*.nii.gz"))
        
        if len(associated_images) < 3:
            print(f"  ⚠️ ¡OJO! El caso {case_id} tiene menos de 3 imágenes ({len(associated_images)})")
        
        for img_path in associated_images:
            file_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(imagesTs_out, file_name))

print(f"\n¡Hecho! Se han copiado {total_selected} PACIENTES COMPLETOS (con sus 3 canales) a '{output_dir}'.")
