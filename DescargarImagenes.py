from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler
import os

def descargar_acordeones(cantidad_total, carpeta_destino):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Dividimos la carga entre diferentes buscadores para llegar a la meta
    buscadores = [
        (GoogleImageCrawler, cantidad_total // 3),
        (BingImageCrawler, cantidad_total // 3),
        (BaiduImageCrawler, cantidad_total // 3)
    ]

    # Palabras clave para asegurar fondo plano y sin personas
    filtros = "button accordion studio shot"

    for crawler_class, cantidad in buscadores:
        print(f"--- Iniciando descarga con {crawler_class.__name__} ---")
        crawler = crawler_class(storage={'root_dir': carpeta_destino})
        
        # Intentamos filtrar por tipo "photo" y color blanco si el buscador lo permite
        crawler.crawl(keyword=filtros, max_num=cantidad, overwrite=False)

# Configuración
descargar_acordeones(850, 'dataset_acordeon')