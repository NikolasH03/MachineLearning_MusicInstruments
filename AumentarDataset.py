import os
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def aumentar_dataset_acordeones(carpeta_input, carpeta_output, meta_total=1000):
    # 1. Parámetros de Aumento: Configuramos solo lo que NO rompe el propósito
    datagen = ImageDataGenerator(
        rotation_range=20,      # Rotación aleatoria +- 20 grados
        width_shift_range=0.1,  # Desplazamiento horizontal 10%
        height_shift_range=0.1, # Desplazamiento vertical 10%
        shear_range=0.1,        # Distorsión de corte ligera
        zoom_range=0.1,         # Zoom ligero (in/out)
        horizontal_flip=True,   # Reflejo horizontal (espejo)
        fill_mode='nearest'     # Cómo rellenar píxeles nuevos (importante para fondo plano)
    )

    # 2. Conteo actual
    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)
        
    imagenes_actuales = [f for f in os.listdir(carpeta_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_inicial = len(imagenes_actuales)
    
    print(f"Total imágenes iniciales: {total_inicial}")
    
    if total_inicial >= meta_total:
        print("Ya tienes suficientes imágenes.")
        return

    imagenes_necesarias = meta_total - total_inicial
    print(f"Generando {imagenes_necesarias} imágenes aumentadas...")

    # 3. Generación
    count = 0
    # Iteramos sobre las imágenes originales de forma aleatoria para no usar siempre las mismas
    while count < imagenes_necesarias:
        # Elegimos una imagen original al azar
        img_name = random.choice(imagenes_actuales)
        img_path = os.path.join(carpeta_input, img_name)
        
        try:
            # Cargamos y convertimos a array (formato que necesita datagen)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0) # Añadir dimensión de batch
            
            # Generamos UNA imagen aumentada
            # .flow genera imágenes aleatoriamente basadas en la configuración de datagen
            # prefix nos ayuda a identificarlas después
            for batch in datagen.flow(x, batch_size=1, 
                                      save_to_dir=carpeta_output, 
                                      save_prefix='aug_acordeon', 
                                      save_format='jpg'):
                count += 1
                if count % 50 == 0:
                    print(f"Generadas {count}/{imagenes_necesarias}...")
                break # Rompemos el bucle interno para generar solo UNA por imagen original en cada ciclo

        except Exception as e:
            print(f"Error procesando {img_name}: {e}")
            continue

    print(f"\n¡Listo! El proceso terminó. Tienes {meta_total} imágenes disponibles.")

# --- CONFIGURACIÓN ---
# La carpeta donde tienes tus 712 imágenes renombradas
ruta_dataset_original = r'D:\nicolas\University\10 Semestre\Aprendizaje Automatico\MachineLearning_MusicInstruments\dataset_final'

# La carpeta donde quieres guardar las NUEVAS imágenes aumentadas 
# (puedes guardarlas en la misma carpeta o en una nueva y luego juntarlas)
ruta_dataset_aumentado = r'D:\nicolas\University\10 Semestre\Aprendizaje Automatico\MachineLearning_MusicInstruments\dataset_augmentado'

aumentar_dataset_acordeones(ruta_dataset_original, ruta_dataset_aumentado, meta_total=1400)