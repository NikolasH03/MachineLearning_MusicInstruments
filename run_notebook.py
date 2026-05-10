import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import nbformat
from nbclient import NotebookClient

nb_path = 'Trabajo/Parte1_CNN_Instrumentos.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb, timeout=600, kernel_name='ml_music',
    resources={'metadata': {'path': '.'}}
)
client.execute()

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('NOTEBOOK PARTE1 EJECUTADO Y GUARDADO')
