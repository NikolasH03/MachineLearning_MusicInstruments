---
name: rl-exploracion-entorno
description: >
  Skill para implementar y documentar correctamente el Punto 1 — Exploración del Entorno
  en actividades de Aprendizaje por Refuerzo con Diferencia Temporal (SARSA / Q-Learning).
  Usar SIEMPRE que el usuario necesite: describir el espacio de estados y acciones de un
  entorno RL, ejecutar episodios con política aleatoria, calcular una línea base (baseline),
  analizar el entorno Blackjack-v1 o CliffWalking-v0, justificar TD vs Monte Carlo vs
  Programación Dinámica, o responder las preguntas formales de exploración de entorno en
  una guía o laboratorio de RL. También aplica cuando el usuario menciona "exploración del
  entorno", "política aleatoria", "línea base RL", "espacio de estados Blackjack",
  "cuántos estados tiene" o "ventajas de TD".
---

# Skill: Exploración del Entorno — Punto 1 RL (TD / SARSA / Q-Learning)

## Propósito

Este skill codifica el desglose completo del **Punto 1** de una guía de Aprendizaje por
Refuerzo con algoritmos de Diferencia Temporal. Cubre cuatro bloques entregables que deben
resolverse en orden y con precisión conceptual antes de entrenar ningún agente.

---

## Bloques de Entregables (en orden obligatorio)

### BLOQUE A — Descripción formal del entorno

Responder con texto + código de soporte las cuatro preguntas estructurales:

1. **Espacio de estados** → tamaño total + significado de cada componente
2. **Espacio de acciones** → cantidad + efecto concreto de cada acción
3. **Función de recompensa** → valor en cada escenario (positivo / nulo / negativo)
4. **Condición de terminación** → qué evento cierra el episodio

**Código mínimo requerido:**
```python
import gymnasium as gym
env = gym.make('Blackjack-v1', sab=False)
print("Espacio de observacion:", env.observation_space)
print("Espacio de acciones   :", env.action_space)
obs, _ = env.reset()
print("Observacion ejemplo   :", obs)
```

**Tabla de descripción formal para Blackjack-v1:**

| Componente | Tipo | Rango | Significado |
|---|---|---|---|
| `player_sum` | Discrete(32) | [0, 31] | Suma de cartas del jugador |
| `dealer_card` | Discrete(11) | [1, 10] | Carta visible del dealer (As=1) |
| `usable_ace` | Discrete(2) | {0, 1} | 1 = el jugador tiene un As contado como 11 sin pasarse |
| **Acción 0** | — | — | **Stand**: plantarse, el dealer juega |
| **Acción 1** | — | — | **Hit**: pedir una carta adicional |

**Función de recompensa Blackjack:**

| Resultado | Recompensa |
|---|---|
| Jugador gana | +1 |
| Empate | 0 |
| Jugador pierde | −1 |
| Jugador se pasa de 21 | −1 (episodio termina inmediatamente) |

**Condición de terminación:**
- El jugador se pasa de 21 (bust)
- El jugador elige Stand → el dealer juega hasta ≥17 → se compara

---

### BLOQUE B — Episodio con política aleatoria (mínimo 3 pasos)

**Patrón de implementación requerido:**

```python
env = gym.make('Blackjack-v1', sab=False)
obs, _ = env.reset(seed=42)

print(f"{'Paso':<6} {'Estado (sum,dealer,ace)':<28} {'Accion':<10} {'Reward':<10} {'Done'}")
print("-" * 68)
print(f"{'0':<6} {str(obs):<28} {'—':<10} {'—':<10} —")

step = 0
done = False
while not done:
    action = env.action_space.sample()           # politica aleatoria
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    accion_str = 'Hit' if action == 1 else 'Stand'
    print(f"{step+1:<6} {str(next_obs):<28} {accion_str:<10} {reward:<10} {done}")
    obs = next_obs
    step += 1
env.close()
```

**Requisitos de output:**
- Mostrar estado ANTES de cada acción
- Mostrar acción como texto legible (Hit/Stand, no 0/1)
- Mostrar recompensa y flag `done` en cada paso
- El episodio debe mostrar al menos 3 transiciones

---

### BLOQUE C — Línea base: 1000 episodios con política aleatoria

**Este valor es la referencia de comparación para SARSA y Q-Learning.**  
Debe fijarse una semilla para reproducibilidad.

**Implementación requerida:**

```python
import numpy as np

N_BASELINE = 1000
SEED_BASE   = 42

env = gym.make('Blackjack-v1', sab=False)
wins, draws, losses, rewards_ep = 0, 0, 0, []

for ep in range(N_BASELINE):
    obs, _ = env.reset(seed=SEED_BASE + ep)
    done = False; ep_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
    rewards_ep.append(ep_reward)
    if ep_reward > 0:    wins   += 1
    elif ep_reward == 0: draws  += 1
    else:                losses += 1
env.close()

rewards_arr = np.array(rewards_ep)
print("=" * 45)
print("  LINEA BASE — Politica Aleatoria (1000 eps)")
print("=" * 45)
print(f"  Recompensa promedio : {rewards_arr.mean():.4f}")
print(f"  Desv. estandar      : {rewards_arr.std():.4f}")
print(f"  Tasa de victoria    : {wins/N_BASELINE:.3f}  ({wins}/1000)")
print(f"  Tasa de empate      : {draws/N_BASELINE:.3f}  ({draws}/1000)")
print(f"  Tasa de derrota     : {losses/N_BASELINE:.3f}  ({losses}/1000)")
print(f"\n  ⚠ 'Exito' = victoria (reward > 0). Empates NO se cuentan.")
print(f"  Esta recompensa promedio ({rewards_arr.mean():.4f}) es el umbral")
print(f"  que SARSA y Q-Learning deben superar para justificarse.")
```

**Aclaración obligatoria en el notebook:**  
"Éxito" se define como `reward > 0` (victoria). Los empates (`reward == 0`) NO
se cuentan como éxito. Se reportan las tres categorías por separado.

---

### BLOQUE D — Análisis específico Blackjack (5 sub-puntos)

#### D1. Ejemplo concreto del estado

Siempre incluir un ejemplo real con interpretación:

```
Estado: (16, 7, 0)
  → player_sum  = 16 : el jugador tiene cartas que suman 16 (ej: 9 + 7)
  → dealer_card = 7  : la carta visible del dealer es un 7
  → usable_ace  = 0  : el jugador NO tiene un As usable como 11
  Interpretación: situación comprometida — 16 contra 7 del dealer,
  sin as de seguridad. La decisión estratégica es difícil.

Estado: (18, 10, 1)
  → player_sum  = 18 : suma actual es 18
  → dealer_card = 10 : dealer muestra un 10 (carta fuerte)
  → usable_ace  = 1  : el jugador TIENE un As que cuenta como 11
                       (ej: As + 7 = 18, si pide carta y se pasa
                        el As pasa a valer 1 → no bust)
```

#### D2. Conteo de estados: teóricos vs alcanzables

```python
# Estados TEORICOS (producto cartesiano de los espacios)
estados_teoricos = 32 * 11 * 2
print(f"Estados teoricos  : {estados_teoricos}")  # 704

# Estados ALCANZABLES (suma jugador relevante: 4-21, dealer: 1-10)
# Una mano de Blackjack nunca empieza con suma < 4 (minimo 2+2)
# Ni se registran estados con suma > 21 (episodio ya termino)
estados_alcanzables = len(range(4, 22)) * 10 * 2
print(f"Estados alcanzables aprox: {estados_alcanzables}")  # 360

# Estados con decision real (el jugador aun puede actuar: suma 12-21)
# Con suma < 12 siempre conviene pedir (no hay riesgo de pasarse)
estados_con_decision = len(range(12, 22)) * 10 * 2
print(f"Estados con decision real: {estados_con_decision}")  # 200
```

**Punto clave a redactar:** El espacio de Blackjack NO es una cuadrícula porque:
- No hay "vecindad espacial" entre estados (el estado (16,7,0) no es "adyacente" a (17,7,0) en ningún sentido geográfico)
- Las transiciones dependen de probabilidades de cartas (estocástico), no de movimientos deterministas
- No se puede usar indexación matricial directa → se usan diccionarios (`defaultdict`) o tablas indexadas por tupla
- El espacio relevante es mucho menor que el teórico → muchas tuplas son inalcanzables

#### D3. ¿Por qué no manejarlo como cuadrícula?

```
Cuadrícula (ej: FrozenLake, CliffWalking):          Blackjack:
  - Estado = posición (fila, columna)                 - Estado = (suma, carta, as)
  - Vecindad clara: arriba/abajo/izq/der              - Sin vecindad espacial
  - Transiciones deterministas o casi                 - Transiciones estocásticas (cartas)
  - Índice matricial directo: Q[fila][col]            - Índice por tupla: Q[(suma,carta,as)]
  - Todos los estados son alcanzables                 - Muchos estados teóricos inalcanzables
  - Política visual en el mapa                        - Política como tabla de tuplas
```

#### D4. ¿TD mejor, peor o igual que Monte Carlo? (ANTES de entrenar)

**Respuesta argumentativa requerida — sin datos experimentales:**

```
Hipótesis: TD sera levemente mejor que MC en velocidad de convergencia,
con resultado final similar en terminos de politica optima.

Argumentos a favor de TD >= MC en Blackjack:
  1. TD actualiza tras CADA carta recibida, no al final del episodio.
     Aunque la recompensa solo llega al final, el bootstrapping permite
     propagar informacion antes de que el episodio termine.
  2. TD tiene menor varianza que MC porque no acumula todas las
     recompensas del episodio — usa estimaciones intermedias.
  3. Con pocos episodios, TD converge mas rapido por el punto 1 y 2.

Argumento honesto de por que la diferencia sera pequena:
  - Los episodios de Blackjack son MUY cortos (tipicamente 2-6 pasos).
  - La ventaja de TD sobre MC se magnifica en episodios LARGOS.
  - Con episodios tan cortos, MC puede calcular el retorno real con
    poca varianza adicional respecto a TD.
  - La politica optima final deberia ser similar en ambos.

Conclusion anticipada: TD ≈ MC en Blackjack, con TD potencialmente
mas estable en las primeras etapas del entrenamiento.
```

#### D5. ¿Por qué aplicar TD? Ventajas vs DP y vs MC

**Celda markdown estructurada requerida:**

```
POR QUE TD ES APLICABLE EN BLACKJACK:

vs Programacion Dinamica (DP):
  ✗ DP requiere conocer P(s'|s,a) — la probabilidad de pasar
    al estado s' al tomar accion a desde s.
  ✗ En Blackjack, P(s'|s,a) depende de la distribucion del mazo,
    que no es directamente accesible como modelo explicito.
  ✓ TD aprende directamente de experiencia (episodios jugados)
    sin necesitar el modelo del entorno. Model-free.

vs Monte Carlo (MC):
  ✗ MC espera al FINAL del episodio para actualizar Q(s,a).
    Requiere episodios completos.
  ✓ TD actualiza Q(s,a) tras CADA paso (bootstrapping).
    Mas eficiente en uso de experiencia.
  ✓ TD puede aplicarse en entornos continuos (no episodicos).
    MC no puede (nunca termina el episodio).
  ✓ TD tiene menor varianza en la estimacion del retorno.

VENTAJA CONCRETA EN BLACKJACK:
  El entorno es episodico pero stochastico. No tenemos el modelo
  del mazo (DP imposible). Los episodios son cortos (MC aceptable
  pero TD mas eficiente). TD es la eleccion natural: sin modelo,
  actualizacion online, convergencia garantizada con alpha decreciente.
```

---

## Reglas de Calidad para el Punto 1

### ✅ Checklist de entregables completos

- [ ] `env.observation_space` y `env.action_space` impresos con código
- [ ] Tabla de recompensas con los 4 escenarios de Blackjack
- [ ] Episodio aleatorio con ≥3 pasos visibles como texto estructurado
- [ ] Baseline de 1000 episodios con: mean, std, win rate, draw rate, loss rate
- [ ] "Éxito" definido explícitamente como `reward > 0`
- [ ] Semilla fijada para reproducibilidad
- [ ] Ejemplo concreto del estado con interpretación de los 3 componentes
- [ ] Conteo: estados teóricos (704) vs alcanzables (~360) vs con decisión (~200)
- [ ] Explicación de por qué Blackjack ≠ cuadrícula (al menos 3 diferencias)
- [ ] Hipótesis TD vs MC argumentada antes de entrenar (sin datos)
- [ ] Argumento TD vs DP (model-free)
- [ ] Argumento TD vs MC (step-by-step vs episodio completo)

### ⚠️ Errores comunes a evitar

| Error | Corrección |
|---|---|
| Usar `env.render()` en Blackjack esperando output visual | Blackjack no tiene render útil — usar prints estructurados |
| Contar empates como éxito en la tasa de éxito | Éxito = `reward > 0` únicamente; reportar las 3 categorías |
| Decir "704 estados posibles" sin distinguir alcanzables | Diferenciar teóricos (704) vs alcanzables (~360) vs relevantes (~200) |
| Responder D4 (TD vs MC) con datos de entrenamiento | D4 debe ser argumento teórico previo al entrenamiento |
| Omitir la semilla en el baseline | `env.reset(seed=SEED + ep)` para reproducibilidad |
| Usar `WidthType.PERCENTAGE` en tablas de reporte | Siempre DXA si se genera DOCX |

### 📐 Estructura de celdas en el notebook

```
[Markdown] Punto 1 — Título y contexto
[Code]     Bloque A: imports + print espacio estados/acciones
[Markdown] Interpretación formal del espacio
[Code]     Bloque B: episodio con política aleatoria (≥3 pasos)
[Markdown] Interpretación del episodio mostrado
[Code]     Bloque C: baseline 1000 episodios + métricas
[Markdown] Bloque D1-D3: estado concreto + conteo + vs cuadrícula
[Markdown] Bloque D4: hipótesis TD vs MC (argumentativa, sin código)
[Markdown] Bloque D5: ventajas TD vs DP y vs MC
```

---

## Valores de referencia esperados (Blackjack-v1, política aleatoria)

Estos valores son aproximados y sirven para validar que la implementación es correcta:

| Métrica | Valor esperado |
|---|---|
| Recompensa promedio | −0.25 a −0.35 |
| Win rate | 0.28 a 0.32 |
| Draw rate | 0.08 a 0.12 |
| Loss rate | 0.55 a 0.65 |
| Estados únicos visitados (1000 eps) | 150 a 250 |

Si los valores difieren significativamente, verificar:
- `sab=False` en `gym.make` (reglas estándar, no "sab" variant)
- Que la semilla esté fijada correctamente
- Que se esté usando `Blackjack-v1` y no una versión distinta

---

## Glosario rápido de términos del Punto 1

| Término | Definición breve |
|---|---|
| **Espacio de estados** | Conjunto de todas las observaciones posibles del entorno |
| **Espacio de acciones** | Conjunto de todas las decisiones que puede tomar el agente |
| **Política aleatoria** | Selección uniforme de acciones sin aprendizaje |
| **Línea base (baseline)** | Rendimiento de referencia sin entrenamiento; umbral mínimo a superar |
| **Bootstrapping** | Usar estimaciones actuales de Q para calcular el target de actualización |
| **Model-free** | No requiere P(s'\|s,a) ni R(s,a) explícitos del entorno |
| **As usable** | As que cuenta como 11 sin que el jugador se pase de 21 |
| **Bust** | Suma del jugador supera 21 → derrota automática |
| **On-policy** | El agente aprende la Q de la política que está siguiendo (SARSA) |
| **Off-policy** | El agente aprende la Q óptima independiente de su política (Q-Learning) |
