"""
=============================================================
  FILTRADOR AUTOMÁTICO DE IMÁGENES DE INSTRUMENTOS
  Soporta: guitar, accordion, drums
  Versión: open-clip-torch
=============================================================

INSTALACIÓN:
    pip install torch open-clip-torch Pillow tqdm

USO:
    python filtrar_instrumentos.py --instrumento accordion --input ./Dataset/Imagenes/Acordeon --output ./Dataset/acordeones_filtrados --top 500
    python filtrar_instrumentos.py --instrumento drums     --input ./Dataset/Imagenes/Bateria --output ./Dataset/baterias_filtradas  --top 500
    python filtrar_instrumentos.py --instrumento guitar    --input ./mis_imagenes --output ./guitarras_filtradas --top 500

    # Con parámetros extra:
    python filtrar_instrumentos.py --instrumento accordion --input ./mis_imagenes --output ./out --top 500 --umbral 0.50
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    import torch
    import open_clip
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("\n❌ Faltan dependencias. Instálalas con:")
    print("   pip install torch open-clip-torch Pillow tqdm\n")
    sys.exit(1)


# ── Prompts por instrumento ────────────────────────────────
INSTRUMENTOS = {
    "guitar": {
        "nombre": "Guitarra",
        "emoji": "🎸",
        "positivos": [
            "a clear photo of a guitar",
            "a guitar instrument visible in the image",
            "acoustic guitar clearly visible",
            "electric guitar clearly visible",
            "a guitar shown prominently",
            "a close up photo of a guitar",
        ],
        "negativos": [
            "a photo without any guitar",
            "a blurry or distant guitar",
            "a person holding something unrelated",
            "an image where guitar is barely visible",
            "background scenery without instruments",
        ],
    },

    "accordion": {
        "nombre": "Acordeón",
        "emoji": "🪗",
        "positivos": [
            "a clear photo of an accordion",
            "an accordion instrument clearly visible and prominent",
            "a close up of an accordion showing bellows and buttons",
            "a diatonic or chromatic accordion in the foreground",
            "an accordion showing its keyboard and bass buttons",
        ],
        "negativos": [
            "a photo without any accordion",
            "a blurry image with an accordion far in the background",
            "people without any visible musical instrument",
            "a person playing an accordion",
            "a guitar or drum kit without accordion",
            "scenery or crowd without instrument",
            "an accordion barely recognizable in the scene",
        ],
    },

    "drums": {
        "nombre": "Batería",
        "emoji": "🥁",
        "positivos": [
            "a clear photo of a drum kit",
            "a drum set clearly visible with cymbals and drums",
            "a close up of drums and cymbals",
            "a drummer playing a full drum kit",
            "a snare drum bass drum and hi-hat visible",
            "a percussion setup clearly shown",
        ],
        "negativos": [
            "a photo without any drums or drum kit",
            "a blurry image with drums far away",
            "people without visible instruments",
            "a guitar or piano without drums",
            "an empty stage without drum kit",
            "a single drum with no full kit visible",
        ],
    },
}


def cargar_modelo():
    print("⏳ Cargando modelo CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Modelo cargado en {device.upper()}")
    return model, preprocess, tokenizer, device


def codificar_textos(model, tokenizer, prompts_pos, prompts_neg, device):
    todos = prompts_pos + prompts_neg
    tokens = tokenizer(todos).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[: len(prompts_pos)], feats[len(prompts_pos):]


def score_imagen(ruta, model, preprocess, text_pos, text_neg, device):
    try:
        img = Image.open(ruta).convert("RGB")
    except Exception:
        return 0.0
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    sim_pos = (feat @ text_pos.T).squeeze().mean().item()
    sim_neg = (feat @ text_neg.T).squeeze().mean().item()
    score = (sim_pos - sim_neg + 1) / 2
    return round(float(score), 4)


def extensiones_validas(ruta):
    return ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def generar_reporte_html(resultados, ruta_salida, top_n, info):
    ruta_html = ruta_salida.parent / f"reporte_{info['nombre'].lower()}.html"
    seleccionadas = [r for r in resultados if r["seleccionada"]]
    descartadas   = [r for r in resultados if not r["seleccionada"]]

    def fila(r):
        ruta_rel = Path(r["ruta"]).as_posix()
        color = "#d4edda" if r["seleccionada"] else "#f8d7da"
        return (
            f'<div style="display:inline-block;width:160px;margin:6px;'
            f'background:{color};border-radius:8px;padding:6px;vertical-align:top;font-size:11px;">'
            f'<img src="{ruta_rel}" style="width:148px;height:110px;object-fit:cover;border-radius:4px;" '
            f'onerror="this.style.display=\'none\'">'
            f'<div style="text-align:center;margin-top:4px;">'
            f'<b>{r["score"]:.3f}</b><br>'
            f'<span style="color:#555;">{Path(r["ruta"]).name[:22]}</span>'
            f'</div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8">
<title>Reporte {info['nombre']}</title>
<style>
  body{{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5}}
  .box{{background:white;padding:16px;border-radius:8px;margin-bottom:20px}}
  h1{{color:#333}} h2{{border-bottom:2px solid #ddd;padding-bottom:8px}}
</style>
</head><body>
<h1>{info['emoji']} Reporte de Filtrado — {info['nombre']}</h1>
<div class="box">
  <b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
  <b>Total analizadas:</b> {len(resultados)}<br>
  <b>Seleccionadas:</b> {len(seleccionadas)} (top {top_n})<br>
  <b>Descartadas:</b> {len(descartadas)}<br>
  <b>Score corte:</b> {seleccionadas[-1]['score'] if seleccionadas else 'N/A':.3f}
</div>
<div class="box"><h2>✅ Seleccionadas ({len(seleccionadas)})</h2>
{''.join(fila(r) for r in seleccionadas[:200])}
{'<p><i>Mostrando primeras 200</i></p>' if len(seleccionadas) > 200 else ''}
</div>
<div class="box"><h2>❌ Descartadas — top 200 por score</h2>
{''.join(fila(r) for r in descartadas[:200])}
</div>
</body></html>"""

    ruta_html.write_text(html, encoding="utf-8")
    return ruta_html


def main():
    parser = argparse.ArgumentParser(description="Filtra imágenes de instrumentos con CLIP")
    parser.add_argument("--instrumento", required=True,
                        choices=list(INSTRUMENTOS.keys()),
                        help=f"Instrumento a detectar: {', '.join(INSTRUMENTOS.keys())}")
    parser.add_argument("--input",   required=True, help="Carpeta con imágenes originales")
    parser.add_argument("--output",  required=True, help="Carpeta destino")
    parser.add_argument("--top",     type=int,   default=500, help="Cuántas imágenes seleccionar")
    parser.add_argument("--umbral",  type=float, default=0.0, help="Score mínimo (0-1)")
    parser.add_argument("--no-reporte", action="store_true")
    args = parser.parse_args()

    info = INSTRUMENTOS[args.instrumento]
    print(f"\n{info['emoji']}  Instrumento: {info['nombre']}")

    carpeta_in  = Path(args.input)
    carpeta_out = Path(args.output)
    carpeta_out.mkdir(parents=True, exist_ok=True)

    imagenes = sorted([p for p in carpeta_in.rglob("*") if extensiones_validas(p)])
    print(f"📁 Imágenes encontradas: {len(imagenes)}")
    if not imagenes:
        print("❌ No se encontraron imágenes.")
        sys.exit(1)

    model, preprocess, tokenizer, device = cargar_modelo()
    text_pos, text_neg = codificar_textos(
        model, tokenizer, info["positivos"], info["negativos"], device
    )

    print(f"\n🔍 Puntuando {len(imagenes)} imágenes...")
    scores = []
    for ruta in tqdm(imagenes, unit="img"):
        s = score_imagen(ruta, model, preprocess, text_pos, text_neg, device)
        scores.append({"ruta": str(ruta), "score": s, "seleccionada": False})

    scores.sort(key=lambda x: x["score"], reverse=True)
    seleccionadas = [r for r in scores if r["score"] >= args.umbral][: args.top]
    for r in seleccionadas:
        r["seleccionada"] = True

    print(f"\n✅ Seleccionadas: {len(seleccionadas)}")
    if seleccionadas:
        print(f"   Score más alto:          {scores[0]['score']:.3f}")
        print(f"   Score más bajo aceptado: {seleccionadas[-1]['score']:.3f}")

    print(f"\n📋 Copiando a {carpeta_out}...")
    for r in tqdm(seleccionadas, unit="img"):
        src = Path(r["ruta"])
        dst = carpeta_out / src.name
        if dst.exists():
            dst = carpeta_out / f"{src.stem}_{src.parent.name}{src.suffix}"
        shutil.copy2(src, dst)

    json_path = carpeta_out.parent / f"scores_{args.instrumento}.json"
    json_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False))
    print(f"💾 Scores: {json_path}")

    if not args.no_reporte:
        ruta_html = generar_reporte_html(scores, carpeta_out, args.top, info)
        print(f"📊 Reporte HTML: {ruta_html}")

    print(f"\n{info['emoji']}  ¡Listo! {len(seleccionadas)} imágenes en '{carpeta_out}'")


if __name__ == "__main__":
    main()
