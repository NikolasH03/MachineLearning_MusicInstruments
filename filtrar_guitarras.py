"""
=============================================================
  FILTRADOR AUTOMÁTICO DE IMÁGENES DE GUITARRAS
  Usa CLIP para puntuar qué tan bien se ve la guitarra
  en cada imagen y selecciona las mejores 500.
=============================================================

INSTALACIÓN (ejecutar una vez):
    pip install torch torchvision open-clip-torch Pillow tqdm

USO:
    python filtrar_guitarras.py --input ./Dataset/Imagenes/Guitarra --output ./Dataset/guitarras_filtradas --top 500

OPCIONES:
    --input    Carpeta con las 1500 imágenes originales
    --output   Carpeta donde se copiarán las 500 seleccionadas
    --top      Cuántas imágenes seleccionar (default: 500)
    --umbral   Score mínimo para considerar (default: 0.20)
    --reporte  Generar HTML con vista previa de resultados (default: True)
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime

# ── dependencias ──────────────────────────────────────────
try:
    import torch
    import open_clip
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("\n❌ Faltan dependencias. Instálalas con:")
    print("   pip install torch torchvision open-clip-torch Pillow tqdm\n")
    sys.exit(1)


# ── prompts de texto que describen una guitarra bien visible ──
PROMPTS_POSITIVOS = [
    "a clear photo of a guitar",
    "a guitar instrument visible in the image",
    "acoustic guitar clearly visible",
    "electric guitar clearly visible",
    "a guitar shown prominently",
    "a close up photo of a guitar",
]

PROMPTS_NEGATIVOS = [
    "a photo without any guitar",
    "a blurry or distant guitar",
    "a person holding something unrelated",
    "an image where guitar is barely visible",
    "background scenery without instruments",
]


def cargar_modelo():
    """Carga CLIP en CPU (sin necesidad de GPU)."""
    print("⏳ Cargando modelo CLIP (primera vez descarga ~350MB)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Modelo cargado en {device.upper()}")
    return model, preprocess, tokenizer, device


def codificar_textos(model, tokenizer, device):
    """Codifica los prompts de texto una sola vez."""
    todos = PROMPTS_POSITIVOS + PROMPTS_NEGATIVOS
    tokens = tokenizer(todos).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    positivos = feats[: len(PROMPTS_POSITIVOS)]
    negativos = feats[len(PROMPTS_POSITIVOS):]
    return positivos, negativos


def score_imagen(ruta, model, preprocess, text_pos, text_neg, device):
    """
    Devuelve un score entre 0 y 1 que indica qué tan bien
    se ve una guitarra en la imagen.
    """
    try:
        img = Image.open(ruta).convert("RGB")
    except Exception:
        return 0.0

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_img = model.encode_image(tensor)
        feat_img = feat_img / feat_img.norm(dim=-1, keepdim=True)

    # similitud con prompts positivos y negativos
    sim_pos = (feat_img @ text_pos.T).squeeze().mean().item()
    sim_neg = (feat_img @ text_neg.T).squeeze().mean().item()

    # score neto: más positivo = guitarra más visible
    score = (sim_pos - sim_neg + 1) / 2  # normalizar a [0, 1]
    return round(float(score), 4)


def extensiones_validas(ruta):
    return ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def generar_reporte_html(resultados, ruta_salida, top_n):
    """Genera un reporte HTML con miniaturas de todas las imágenes puntuadas."""
    ruta_html = ruta_salida.parent / "reporte_filtrado.html"

    seleccionadas = [r for r in resultados if r["seleccionada"]]
    descartadas   = [r for r in resultados if not r["seleccionada"]]

    def fila(r, mostrar=True):
        ruta_rel = Path(r["ruta"]).as_posix()
        color = "#d4edda" if r["seleccionada"] else "#f8d7da"
        return (
            f'<div style="display:inline-block;width:160px;margin:6px;'
            f'background:{color};border-radius:8px;padding:6px;'
            f'vertical-align:top;font-size:11px;">'
            f'<img src="{ruta_rel}" style="width:148px;height:110px;'
            f'object-fit:cover;border-radius:4px;" onerror="this.style.display=\'none\'">'
            f'<div style="text-align:center;margin-top:4px;">'
            f'<b>{r["score"]:.3f}</b><br>'
            f'<span style="color:#555;">{Path(r["ruta"]).name[:20]}</span>'
            f'</div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte Filtrado de Guitarras</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .stats {{ background: white; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
  .seccion {{ background: white; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
  h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
</style>
</head>
<body>
<h1>🎸 Reporte de Filtrado Automático de Guitarras</h1>
<div class="stats">
  <b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
  <b>Total analizadas:</b> {len(resultados)}<br>
  <b>Seleccionadas:</b> {len(seleccionadas)} (top {top_n})<br>
  <b>Descartadas:</b> {len(descartadas)}<br>
  <b>Score mínimo seleccionadas:</b> {min(r['score'] for r in seleccionadas):.3f}<br>
  <b>Score máximo descartadas:</b> {max((r['score'] for r in descartadas), default=0):.3f}
</div>

<div class="seccion">
  <h2>✅ Seleccionadas ({len(seleccionadas)})</h2>
  {''.join(fila(r) for r in seleccionadas[:200])}
  {'<p><i>... mostrando primeras 200</i></p>' if len(seleccionadas) > 200 else ''}
</div>

<div class="seccion">
  <h2>❌ Descartadas ({len(descartadas)}) — ordenadas por score desc</h2>
  {''.join(fila(r) for r in descartadas[:200])}
  {'<p><i>... mostrando primeras 200</i></p>' if len(descartadas) > 200 else ''}
</div>
</body>
</html>"""

    ruta_html.write_text(html, encoding="utf-8")
    return ruta_html


def main():
    parser = argparse.ArgumentParser(description="Filtra imágenes de guitarras con CLIP")
    parser.add_argument("--input",   required=True, help="Carpeta con imágenes originales")
    parser.add_argument("--output",  required=True, help="Carpeta destino para las seleccionadas")
    parser.add_argument("--top",     type=int, default=500, help="Cuántas imágenes seleccionar")
    parser.add_argument("--umbral",  type=float, default=0.0, help="Score mínimo (0-1)")
    parser.add_argument("--no-reporte", action="store_true", help="No generar reporte HTML")
    args = parser.parse_args()

    carpeta_in  = Path(args.input)
    carpeta_out = Path(args.output)
    carpeta_out.mkdir(parents=True, exist_ok=True)

    # recopilar imágenes
    imagenes = sorted([p for p in carpeta_in.rglob("*") if extensiones_validas(p)])
    print(f"\n📁 Imágenes encontradas: {len(imagenes)}")
    if not imagenes:
        print("❌ No se encontraron imágenes. Verifica la carpeta --input")
        sys.exit(1)

    # cargar modelo
    model, preprocess, tokenizer, device = cargar_modelo()
    text_pos, text_neg = codificar_textos(model, tokenizer, device)

    # puntuar imágenes
    print(f"\n🔍 Puntuando {len(imagenes)} imágenes...")
    scores = []
    for ruta in tqdm(imagenes, unit="img"):
        s = score_imagen(ruta, model, preprocess, text_pos, text_neg, device)
        scores.append({"ruta": str(ruta), "score": s, "seleccionada": False})

    # ordenar y seleccionar top N
    scores.sort(key=lambda x: x["score"], reverse=True)
    seleccionadas = [r for r in scores if r["score"] >= args.umbral][: args.top]
    for r in seleccionadas:
        r["seleccionada"] = True

    print(f"\n✅ Imágenes seleccionadas: {len(seleccionadas)}")
    print(f"   Score más alto:  {scores[0]['score']:.3f}")
    print(f"   Score más bajo (seleccionado): {seleccionadas[-1]['score']:.3f}")

    # copiar archivos
    print(f"\n📋 Copiando a {carpeta_out}...")
    for r in tqdm(seleccionadas, unit="img"):
        src = Path(r["ruta"])
        dst = carpeta_out / src.name
        # evitar colisiones de nombre
        if dst.exists():
            dst = carpeta_out / f"{src.stem}_{src.parent.name}{src.suffix}"
        shutil.copy2(src, dst)

    # guardar JSON con todos los scores
    json_path = carpeta_out.parent / "scores_todas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"💾 Scores guardados en: {json_path}")

    # reporte HTML
    if not args.no_reporte:
        ruta_html = generar_reporte_html(scores, carpeta_out, args.top)
        print(f"📊 Reporte HTML: {ruta_html}")

    print(f"\n🎸 ¡Listo! {len(seleccionadas)} imágenes copiadas a '{carpeta_out}'")


if __name__ == "__main__":
    main()
