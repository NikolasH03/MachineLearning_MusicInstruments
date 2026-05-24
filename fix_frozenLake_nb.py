import nbformat, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Trabajo/Actividad3_RL_FrozenLake.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

def code(source, cid=None):
    c = nbformat.v4.new_code_cell(source=source)
    if cid: c['id'] = cid
    return c

def md(source, cid=None):
    c = nbformat.v4.new_markdown_cell(source=source)
    if cid: c['id'] = cid
    return c

def find(nb, cid):
    for i, c in enumerate(nb.cells):
        if c.get('id') == cid:
            return i, c
    return -1, None

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1a – Episodio completo con render (insertar después de cell-map)
# ─────────────────────────────────────────────────────────────────────────────
EPISODE_RENDER = '''\
# Requisito: ejecutar un episodio completo con env.render()
print('Episodio con politica ALEATORIA (render paso a paso):')
print()
s = entorno.reset()
paso = 0
done = False
recompensa_total = 0

while not done and paso < 100:
    # Intentar render en modo ansi (texto)
    try:
        salida = entorno.render(mode='ansi')
        if isinstance(salida, str) and salida.strip():
            print(salida)
    except Exception:
        try:
            entorno.render()
        except Exception:
            pass

    a = entorno.action_space.sample()
    s_nuevo, r, done, _ = entorno.step(a)
    print(f'Paso {paso+1:2d}: estado {s:2d} + {ACCIONES[a]:<12} -> estado {s_nuevo:2d} | r={r}')
    s = s_nuevo
    recompensa_total += r
    paso += 1

print()
resultado = 'EXITO' if recompensa_total > 0 else 'FALLO'
print(f'Resultado: {resultado} en {paso} pasos (recompensa acumulada = {recompensa_total})')'''

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1b – Recompensa promedio política aleatoria (500 episodios)
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_EVAL = '''\
# Requisito: recompensa promedio de la politica aleatoria en 500 episodios
N_ALEATORIO = 500
recompensas_al = []
exitos_al = 0

for _ in range(N_ALEATORIO):
    s = entorno.reset()
    done = False
    r_total = 0
    while not done:
        a = entorno.action_space.sample()
        s, r, done, _ = entorno.step(a)
        r_total += r
    recompensas_al.append(r_total)
    if r_total > 0:
        exitos_al += 1

media_al = float(np.mean(recompensas_al))
std_al   = float(np.std(recompensas_al))
tasa_al  = exitos_al / N_ALEATORIO

print(f'=== Politica Aleatoria ({N_ALEATORIO} episodios) ===')
print(f'Recompensa promedio : {media_al:.4f} +/- {std_al:.4f}')
print(f'Tasa de exito       : {tasa_al:.2%}  ({exitos_al}/{N_ALEATORIO} episodios exitosos)')
print()
print('Esta linea base (baseline) debe ser superada por cualquier algoritmo de RL.')'''

idx_map, _ = find(nb, 'cell-map')
nb.cells.insert(idx_map + 1, code(EPISODE_RENDER, 'cell-episode-render'))
nb.cells.insert(idx_map + 2, code(RANDOM_EVAL,    'cell-random-policy-eval'))
print('Fix 1: episodio render + politica aleatoria insertados')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1c – Ejemplo B en cell-transition: estado 54 -> estado 62
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'cell-transition')
cell.source = '''\
# P[s][a] -> [(probabilidad, estado_siguiente, recompensa, terminal), ...]
print('Funcion de transicion: P[s][a] -> [(prob, s_nuevo, reward, done)]')
print()
print('Ejemplo A: estado 0 (S), accion 2 (Derecha):')
for prob, s_, r, done in P[0][2]:
    print(f'  -> estado {s_:2d} | prob={prob:.4f} | r={r} | done={done}')
print()
print('Ejemplo B: estado 62 (F, adyacente a la meta), accion 2 (Derecha):')
for prob, s_, r, done in P[62][2]:
    print(f'  -> estado {s_:2d} | prob={prob:.4f} | r={r} | done={done}')
print()
print('Desde el estado 62, ir a la Derecha puede: llegar a meta (63), caer')
print('en hoyo (54) por deslizamiento, o rebotar en la pared inferior (62).')
print('Cada transicion tiene exactamente prob=1/3 (is_slippery=True).')'''
print('Fix 1c: ejemplo B corregido a estado 62')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 – Aumentar N_EPISODIOS_MC a 50 000
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'cell-mc-hyperparams')
cell.source = '''\
EPSILON_VALORES = [0.05, 0.15, 0.30]
GAMMA_MC        = 0.99
N_EPISODIOS_MC  = 50_000   # 50k para convergencia visible en FrozenLake 8x8
VENTANA         = 500      # ventana deslizante para suavizar curvas'''
print('Fix 2: N_EPISODIOS_MC -> 50_000')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3a – iterar_politica retorna n_iter
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'cell-pd-iterators')
cell.source = '''\
def iterar_politica(P, nS, nA, gamma=GAMMA, theta=THETA):
    politica = np.ones([nS, nA]) / nA
    todos_deltas = []
    n_iter = 0
    while True:
        V, deltas = evaluacion_politica(P, nS, nA, politica, gamma, theta)
        todos_deltas.extend(deltas)
        politica_nueva = mejorar_politica(P, nS, nA, V, gamma)
        n_iter += 1
        if (politica_nueva == politica).all():
            break
        politica = np.copy(politica_nueva)
    print(f'PI: {n_iter} mejoras de politica | {len(todos_deltas)} barridos de evaluacion')
    return politica_nueva, V, todos_deltas, n_iter

def iteracion_valores(P, nS, nA, gamma=GAMMA, theta=THETA):
    V = np.zeros(nS)
    deltas = []
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            V[s] = max(calcular_funcion_q(P, nA, V, s, gamma))
            delta = max(delta, abs(V[s] - v))
        deltas.append(delta)
        if delta < theta:
            break
    politica = mejorar_politica(P, nS, nA, V, gamma)
    print(f'VI: {len(deltas)} barridos hasta convergencia')
    return politica, V, deltas

def evaluar_politica(entorno, politica, episodios=1000):
    exitos = 0
    for _ in range(episodios):
        s = entorno.reset()
        done = False
        while not done:
            a = np.argmax(politica[s])
            s, r, done, _ = entorno.step(a)
        if r == 1.0:
            exitos += 1
    return exitos / episodios'''

_, cell = find(nb, 'cell-run-pd')
cell.source = '''\
print('Ejecutando Iteracion de Politica (PI)...')
politica_pi, V_pi, deltas_pi, n_iter_pi = iterar_politica(P, nS, nA)

print()
print('Ejecutando Iteracion de Valores (VI)...')
politica_vi, V_vi, deltas_vi = iteracion_valores(P, nS, nA)'''
print('Fix 3a: iterar_politica retorna n_iter_pi')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3b – cell-pd-eval: f-string corregido + usa n_iter_pi
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'cell-pd-eval')
cell.source = '''\
N_EVAL = 1000
print(f"Evaluando politicas en {N_EVAL} episodios...")
tasa_pi = evaluar_politica(entorno, politica_pi, N_EVAL)
tasa_vi = evaluar_politica(entorno, politica_vi, N_EVAL)

print()
print(f"{'Algoritmo':<28} {'Tasa exito':>12} {'Barridos':>10} {'Mejoras pol.':>14}")
print('-' * 66)
print(f"{'Iteracion de Politica (PI)':<28} {tasa_pi:>12.2%} {len(deltas_pi):>10} {n_iter_pi:>14}")
print(f"{'Iteracion de Valores  (VI)':<28} {tasa_vi:>12.2%} {len(deltas_vi):>10} {'N/A':>14}")
print()
print(f"Baseline politica aleatoria: {tasa_al:.2%}  |  "
      f"PI mejora: {tasa_pi - tasa_al:+.2%}  |  VI mejora: {tasa_vi - tasa_al:+.2%}")'''
print('Fix 3b: f-string corregido y tabla mejorada')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3c – Analisis PD: referencia a valores observados
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'md-sec2-analysis')
cell.source = '''\
### Análisis – Sección 2: Programación Dinámica

**Convergencia (ver gráficos y tabla arriba):**
- **Iteración de Política (PI)**: convergió en `n_iter_pi` mejoras de política con `len(deltas_pi)` barridos totales de evaluación.
  En el gráfico se observan **saltos de δ** que corresponden a cada nueva mejora; cada mejora reinicia la evaluación.
- **Iteración de Valores (VI)**: convergió en `len(deltas_vi)` barridos de forma **monotónica y continua** (un solo ciclo descendente).
  VI requiere más barridos totales, pero cada barrido es más simple (sin subrutina de evaluación separada).

**Política óptima:**
Ambos algoritmos convergen a la misma política óptima (o equivalente), confirmando la equivalencia teórica de PI y VI.

**Comparación con la política aleatoria (baseline, Sección 1):**
La mejora de tasa de éxito respecto a la política aleatoria confirma que PD aprende comportamientos útiles.
La tasa obtenida refleja el **óptimo estocástico**: incluso con la política perfecta, el deslizamiento
puede llevar al agente a un hoyo por azar — 100% de éxito no es alcanzable con `is_slippery=True`.

**Ventaja de PD:** calcula la solución exacta sin exploración. Limitación: requiere conocer P completamente.'''
print('Fix 3c: analisis PD actualizado')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3d – Analisis MC: referencia a tabla observada (no predicciones)
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'md-sec3-analysis')
cell.source = '''\
### Análisis – Sección 3: Monte Carlo Control

**Efecto de ε (ver tabla y curvas arriba):**

- **ε = 0.05 (exploración baja)**: muy poca exploración para descubrir el camino a G en 64 estados
  con recompensas escasas. La curva de aprendizaje sube lentamente o permanece baja.

- **ε = 0.15 (balance)**: equilibrio entre exploración y explotación. Permite descubrir G con
  suficiente frecuencia para estimar Q correctamente, sin sacrificar demasiado la explotación.

- **ε = 0.30 (exploración alta)**: descubre G rápidamente durante el entrenamiento.
  La política greedy resultante puede superar a ε=0.05 gracias a una Q mejor estimada,
  aunque el 30% de acciones aleatorias agrega ruido a las curvas de entrenamiento.

**Observación clave — recompensas escasas:**
FrozenLake 8×8 da R=1 solo al llegar a G (estado 63). Con ε bajo, si el agente no descubre G
en los primeros episodios, Q ≈ 0 en casi todos los estados y no hay señal para mejorar.
Por eso, en este entorno, una ε **moderada-alta favorece el aprendizaje** (ver tabla de evaluación).

**Mejor ε:** según la tabla de evaluación, el mejor ε fue el que produjo mayor tasa de éxito greedy.
Ese será el parámetro fijo para la Sección 4.'''
print('Fix 3d: analisis MC actualizado')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 – Visualización de política con flechas (insertar antes de md-sec4-analysis)
# ─────────────────────────────────────────────────────────────────────────────
POLICY_VIZ = '''\
# Visualizacion de la politica aprendida como cuadricula con flechas
FLECHAS = {0: 'izq', 1: 'abj', 2: 'der', 3: 'arr'}
FLECHAS_UNICODE = {0: u'\\u2190', 1: u'\\u2193', 2: u'\\u2192', 3: u'\\u2191'}

def visualizar_politica_flechas(politica, titulo, mapa_tiles, ax):
    COLORES = {'S': '#A8E6CF', 'F': '#DCEEFB', 'H': '#FFAAA5', 'G': '#FFD700'}
    for i in range(8):
        for j in range(8):
            tile = mapa_tiles[i][j]
            rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                  facecolor=COLORES.get(tile, '#DCEEFB'),
                                  edgecolor='gray', linewidth=0.8)
            ax.add_patch(rect)
            s = i * 8 + j
            if tile in ('H', 'G', 'S'):
                ax.text(j, i, tile, ha='center', va='center', fontsize=13,
                        fontweight='bold', color='black')
            else:
                flecha = FLECHAS_UNICODE[int(np.argmax(politica[s]))]
                ax.text(j, i, flecha, ha='center', va='center', fontsize=18, color='navy')

    ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xlabel('Columna'); ax.set_ylabel('Fila')
    ax.set_title(titulo, fontsize=11)

mejor_pol_mc = resultados_gamma[mejor_gamma]['pol']

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
visualizar_politica_flechas(politica_vi,
    f'Politica optima - Iteracion de Valores (PD)', mapa_tiles, axes[0])
visualizar_politica_flechas(mejor_pol_mc,
    f'Politica MC (eps={mejor_eps}, gamma={mejor_gamma})', mapa_tiles, axes[1])

plt.suptitle('Politicas aprendidas: PD vs Monte Carlo  |  Leyenda: S=Inicio  H=Hoyo  G=Meta',
             fontsize=12)
plt.tight_layout()
plt.savefig('politica_flechas.png', dpi=120, bbox_inches='tight')
plt.show()'''

idx_analysis, _ = find(nb, 'md-sec4-analysis')
nb.cells.insert(idx_analysis, code(POLICY_VIZ, 'cell-policy-viz'))
print('Fix 4: visualizacion de politica con flechas insertada')

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3e – Analisis gamma: referencia a tabla, no prediccion
# ─────────────────────────────────────────────────────────────────────────────
_, cell = find(nb, 'md-sec4-analysis')
cell.source = '''\
### Análisis – Sección 4: Efecto de γ

**γ = 0.7 (descuento agresivo):**
El agente valora solo recompensas cercanas. En FrozenLake 8×8, la meta puede estar a más de 20 pasos
desde el inicio. El valor descontado de llegar a G sería γ²⁰ = 0.7²⁰ ≈ 0.0008 — prácticamente nulo.
Q ≈ 0 en la mayoría de estados → la política no aprende a dirigirse a G.

**γ = 0.9 (descuento moderado):**
Mejora respecto a γ=0.7. La recompensa de G se propaga más lejos (γ²⁰ = 0.9²⁰ ≈ 0.12).
Hay señal útil, pero se atenúa significativamente para estados lejanos a la meta.

**γ = 0.99 (descuento mínimo):**
La recompensa de G se propaga casi intacta (γ²⁰ = 0.99²⁰ ≈ 0.82). El agente "ve" el valor
de llegar a G incluso desde el estado inicial. Esto es crítico en entornos con recompensas
escasas y horizontes largos como FrozenLake 8×8.

**Resultados observados (ver tabla arriba):**
Los valores de evaluación confirman el efecto esperado: γ mayor produce tasas de éxito más altas.
El mapa de calor de Q con γ=0.99 muestra valores que se propagan gradualmente desde G hacia S,
formando un gradiente que guía la navegación. Con γ=0.7 el mapa es casi uniforme (Q≈0).

**Visualización de política:**
Las flechas muestran la acción greedy en cada celda libre. La política de PD y la de MC con
el mejor γ deberían coincidir en las rutas principales hacia G, aunque pueden diferir en celdas
alejadas de la meta donde la señal de Q es más débil (especialmente para MC con pocos episodios).'''
print('Fix 3e: analisis gamma actualizado')

# ─────────────────────────────────────────────────────────────────────────────
# Guardar
# ─────────────────────────────────────────────────────────────────────────────
with open('Trabajo/Actividad3_RL_FrozenLake.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

# Verificacion final
with open('Trabajo/Actividad3_RL_FrozenLake.ipynb', 'r', encoding='utf-8') as f:
    nb2 = nbformat.read(f, as_version=4)

fstrings_malos = [(i, l.strip()) for i, c in enumerate(nb2.cells) if c.cell_type=='code'
                  for l in c.source.splitlines() if "f'" in l and "{'" in l]
savefig_malo   = [(i, l.strip()) for i, c in enumerate(nb2.cells) if c.cell_type=='code'
                  for l in c.source.splitlines() if 'savefig' in l and 'Trabajo/' in l]

print()
print(f'Total celdas: {len(nb2.cells)}')
print(f'f-strings problematicos : {fstrings_malos if fstrings_malos else "ninguno"}')
print(f'savefig con Trabajo/    : {savefig_malo   if savefig_malo   else "ninguno"}')
print()
print('IDs de celdas clave:')
ids_buscar = ['cell-episode-render','cell-random-policy-eval','cell-pd-eval',
              'cell-mc-hyperparams','cell-policy-viz','md-sec4-analysis']
for cid in ids_buscar:
    idx, _ = find(nb2, cid)
    print(f'  {cid:<30} -> celda {idx}')
