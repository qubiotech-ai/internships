"""
import pandas as pd

def analizar_dataset(path_csv):
    # 1. Sujetos Sanos (Control)
    df =pd.read_csv(path_csv)
    sanos = df[df['NOFINDINGS'] == 1]['Subject ID'].unique()
    
    # 2. Sujetos con rCMBs (Filtrado por MCH y Definite como acordamos)
    # Importante: Agrupamos por Subject ID para no contar varias veces al mismo sujeto
    df_rcmb = df[(df['TYPE'] == 'MCH') & (df['STATUS'] == 'Definite')]
    sujetos_con_cmb = df_rcmb['Subject ID'].unique()
    
    # 3. Estadísticas de rCMBs por sujeto
    # Contamos cuántas filas (CMBs) tiene cada LONI_IMG_ID o Subject ID
    conteo_por_sujeto = df_rcmb.groupby('Subject ID')['LONI_IMG_ID'].count()
    total_rcmbs = len(df_rcmb)
    
    print(f"--- INVENTARIO DEL DATASET ---")
    print(f"Sujetos Sanos (Control): {len(sanos)}")
    print(f"Sujetos con rCMBs Definitivos: {len(sujetos_con_cmb)}")
    print(f"Total de rCMBs detectadas: {total_rcmbs}")
    print(f"\nDistribución de rCMBs por sujeto:")
    print(conteo_por_sujeto.value_counts().sort_index())
    
    return sujetos_con_cmb, sanos

# Ejecución
path = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/ADNI_MCH_Clean_Dataset.csv"
sujetos_pos, sujetos_neg = analizar_dataset(path)


import os
import pandas as pd

def auditar_descargas(path_csv, folder_pos, folder_neg):
    df = pd.read_csv(path_csv)
    
    # IDs en disco (quitando la extensión .nii.gz)
    ids_pos_disco = [f.split('.')[0] for f in os.listdir(folder_pos) if f.endswith('.nii.gz')]
    ids_neg_disco = [f.split('.')[0] for f in os.listdir(folder_neg) if f.endswith('.nii.gz')]

    # 1. Verificar Positivos
    df_pos = df[df['LONI_IMG_ID'].astype(str).isin(ids_pos_disco)]
    incorrectos = df_pos[df_pos['STATUS'] != 'Definite']
    
    print(f"--- AUDITORÍA DE POSITIVOS ({len(ids_pos_disco)} archivos) ---")
    if not incorrectos.empty:
        print(f"ALERTA: Hay {len(incorrectos)} imágenes que NO son 'Definite'.")
        print(incorrectos[['LONI_IMG_ID', 'STATUS', 'TYPE']].head())
    else:
        print("Confirmado: Todos los positivos en disco son 'Definite'.")

    # 2. Verificar Negativos
    df_neg = df[df['LONI_IMG_ID'].astype(str).isin(ids_neg_disco)]
    no_sanos = df_neg[df_neg['NOFINDINGS'] != 1]

    print(f"\n--- AUDITORÍA DE NEGATIVOS ({len(ids_neg_disco)} archivos) ---")
    if not no_sanos.empty:
        print(f"ALERTA: Hay {len(no_sanos)} imágenes marcadas como sanas que tienen hallazgos.")
    else:
        print("Confirmado: Todos los negativos en disco son sujetos sanos.")

# Ejecutar antes de seguir con SynthSeg
path_positives = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/workdir_ADNI_subset/raw/positives"
path_negatives = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/workdir_ADNI_subset/raw/negatives"

auditar_descargas(path, path_positives, path_negatives)

import pandas as pd

def generar_strings_busqueda(path_csv):
    df = pd.read_csv(path_csv)
    # 1. IDs de rCMB Definite
    ids_pos = df[(df['TYPE'] == 'MCH') & (df['STATUS'] == 'Definite')]['LONI_IMG_ID'].unique().astype(str).tolist()
    
    # 2. IDs de Sanos (Control)
    ids_neg = df[df['NOFINDINGS'] == 1]['LONI_IMG_ID'].unique().astype(str).tolist()
    
    todos_los_ids = ids_pos + ids_neg
    print(f"Total de IDs a procesar: {len(todos_los_ids)}")
    
    # Dividir en bloques de 500 para evitar errores en la web
    bloque_size = 500
    bloques = [todos_los_ids[i:i + bloque_size] for i in range(0, len(todos_los_ids), bloque_size)]
    
    for i, bloque in enumerate(bloques):
        print(f"\n--- BLOQUE {i+1} (Pegar en buscador de LONI) ---")
        print(",".join(bloque))

# Ejecución
generar_strings_busqueda(path)

import pandas as pd

def verificar_integridad(path_original, path_maestro):
    # 1. Cargar ambos archivos
    df_orig = pd.read_csv(path_original)
    df_maestro = pd.read_csv(path_maestro)

    # Función para limpiar el ID: quitar 'I'  pasar a string
    def clean_id(x):
        s = str(x).strip()
        return s[1:] if s.startswith('I') else s
    
    df_orig['ID_CLEAN'] = df_orig['LONI_IMG_ID'].apply(clean_id)
    df_maestro['ID_CLEAN'] = df_maestro['LONI_IMG_ID'].apply(clean_id)

    # 2. Definir quiénes DEBERÍAN estar (Criterio estricto)
    # Positivos: MCH + Definite
    mask_pos = (df_orig['TYPE'] == 'MCH') & (df_orig['STATUS'] == 'Definite')
    # Negativos: NOFINDINGS == 1
    mask_neg = (df_orig['NOFINDINGS'] == 1)
    
    ids_esperados = set(df_orig[mask_pos | mask_neg]['ID_CLEAN'].unique())
    ids_maestro = set(df_maestro['ID_CLEAN'].unique())

    # Comparar
    perdidos = ids_esperados - ids_maestro
    sobrantes = ids_maestro - ids_esperados
    
    print(f"--- RESULTADOS AUDITORÍA (Normalizada) ---")
    print(f"Total IDs que deberían estar: {len(ids_esperados)}")
    print(f"Total IDs en tu Maestro:      {len(ids_maestro)}")
    print(f"Diferencia neta:              {len(ids_esperados) - len(ids_maestro)}")
    
    if not perdidos:
        print("\n¡Perfecto! Cruce completo.")
    else:
        print(f"\nFALTAN {len(perdidos)} IDs en tu Maestro.")
        print(f"Ejemplos: {list(perdidos)[:5]}")
        
    if sobrantes:
        print(f"TU MAESTRO TIENE {len(sobrantes)} IDs que no cumplen el filtro.")
# Uso:

path_csv_estudio_original = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/ADNI_original_dataset_downloaded/MAYOADIRL_MRI_MCH_12Feb2026.csv"
path_csv_clean_dataset = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/ADNI_MCH_Clean_Dataset.csv"
verificar_integridad(path_csv_estudio_original, path_csv_clean_dataset)
"""

import pandas as pd

def preparar_descarga_completa(path_orig):
    df_orig = pd.read_csv(path_orig)
    
    clean_id = lambda x: str(x).strip()[1:] if str(x).strip().startswith('I') else str(x).strip()
    
    mask_pos = (df_orig['TYPE'] == 'MCH') & (df_orig['STATUS'] == 'Definite')
    mask_neg = (df_orig['NOFINDINGS'] == 1)
    
    ids_completos = df_orig[mask_pos | mask_neg]['LONI_IMG_ID'].apply(clean_id).unique().tolist()
    
    print(f"Preparando strings para {len(ids_completos)} imágenes...")
    
    # Bloques de 500 para la web de LONI
    for i in range(0, len(ids_completos), 500):
        bloque = ids_completos[i:i+500]
        print(f"\n--- BLOQUE {(i//500)+1} ---")
        print(",".join(bloque))

path_csv_estudio_original = "/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/ADNI_original_dataset_downloaded/MAYOADIRL_MRI_MCH_12Feb2026.csv"
preparar_descarga_completa(path_csv_estudio_original)