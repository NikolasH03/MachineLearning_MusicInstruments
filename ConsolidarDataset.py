import os
import shutil

def consolidar_dataset(carpeta_origen, carpeta_destino):
    # Crear la carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"Directorio creado: {carpeta_destino}")

    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    contador = 0

    # os.walk recorre carpetas, subcarpetas y archivos
    for root, dirs, files in os.walk(carpeta_origen):
        for file in files:
            if file.lower().endswith(formatos_validos):
                path_original = os.path.join(root, file)
                
                # Creamos un nombre nuevo: "acordeon_001.jpg", "acordeon_002.jpg", etc.
                # El :04d pone ceros a la izquierda (0001) para que queden ordenadas
                extension = os.path.splitext(file)[1].lower()
                nuevo_nombre = f"acordeon_{contador:04d}{extension}"
                path_destino = os.path.join(carpeta_destino, nuevo_nombre)

                # Copiamos el archivo (usamos copy2 para mantener metadatos si es necesario)
                shutil.copy2(path_original, path_destino)
                
                contador += 1
                if contador % 50 == 0:
                    print(f"Procesadas {contador} imágenes...")

    print(f"\n¡Listo! Se consolidaron {contador} imágenes en: {carpeta_destino}")

# --- CONFIGURACIÓN ---
# Pon aquí la ruta de la carpeta donde tienes todas las subcarpetas con fotos
ruta_padre = r'D:\nicolas\University\10 Semestre\Aprendizaje Automatico\MachineLearning_MusicInstruments\dataset_final'

# Pon aquí la ruta donde quieres que queden todas juntas y renombradas
ruta_final = r'D:\nicolas\University\10 Semestre\Aprendizaje Automatico\MachineLearning_MusicInstruments\dataset_acordeon'

consolidar_dataset(ruta_padre, ruta_final)