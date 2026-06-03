# Guía de Sustentación — Actividad 2: SARSA y Q-Learning en Blackjack

---

## 1. EL MAPA GRANDE — Dónde encaja esta actividad

```
Aprendizaje por Refuerzo
│
├── Programación Dinámica (PD)
│   ✗ Requiere modelo del entorno P(s'|s,a)
│   ✗ No aplicable a Blackjack
│
├── Monte Carlo (MC)  ← actividad anterior (Blackjack-v0)
│   ✗ Actualiza al FINAL del episodio
│   ✓ No requiere modelo
│
└── Diferencia Temporal (TD)  ← ESTA ACTIVIDAD
    ✓ Actualiza en CADA PASO
    ✓ No requiere modelo
    │
    ├── SARSA (on-policy)
    └── Q-Learning (off-policy)
```

**Frase para la sustentación:**
> "TD es el equilibrio entre PD y MC: como MC no necesita el modelo, pero como PD actualiza
> en cada paso sin esperar el final del episodio."

---

## 2. TÉRMINOS CLAVE con definición simple

### Función Q — `Q(s, a)`
Número que representa **cuánto beneficio espero obtener** si estoy en estado `s` y tomo acción `a`, siguiendo una política desde ese punto.

- En Blackjack: Q((16, 7, 0), Hit) = ¿cuánto gano en promedio si tengo 16 contra 7 del dealer y pido carta?
- Empieza en 0. Se ajusta episodio a episodio.

---

### Política ε-greedy
Regla para elegir acciones:
```
Con probabilidad (1-ε) → tomo la mejor acción conocida (greedy / explotación)
Con probabilidad  ε    → tomo una acción aleatoria (exploración)
```
- ε = 0 → siempre greedy, nunca explora
- ε = 1 → siempre aleatorio, nunca aprende

**Analogía:** es como estudiar para un examen. ε alto = leer temas nuevos (explorar). ε bajo = repasar lo que ya sé (explotar).

---

### Alpha `α` — Tasa de aprendizaje
Controla cuánto "peso" le doy a la información nueva vs la que ya tenía.

```
Q_nuevo = Q_viejo + α × (error_TD)
```
- α = 0.9 → el agente "olvida" casi todo lo anterior y cree ciegamente la nueva experiencia
- α = 0.1 → actualiza muy suavemente, promedia muchas experiencias
- α muy alto → curva de entrenamiento ruidosa, puede no converger
- α muy bajo → converge pero necesita muchos episodios

**Analogía:** α es como la "terquedad" del agente. α alto = cambia de opinión rápido. α bajo = necesita muchas evidencias para cambiar.

---

### Gamma `γ` — Factor de descuento
Controla cuánto valora el agente las recompensas **futuras** vs las inmediatas.

```
γ = 0.7  → recompensa en 4 pasos vale: 0.7⁴ ≈ 0.24 del valor original
γ = 0.99 → recompensa en 4 pasos vale: 0.99⁴ ≈ 0.96 del valor original
```

**¿Por qué γ alto en Blackjack?**
Porque la única recompensa (+1/-1) llega al FINAL del episodio. Con γ bajo, los estados previos a ganar reciben un valor artificialmente pequeño → el agente no aprende a estrategizar.

**Analogía:** γ es la "paciencia" del agente. γ bajo = quiere resultados ya. γ alto = puede pensar a largo plazo.

---

### Bootstrapping
Actualizar Q usando estimaciones actuales de Q (no el valor real).

```
TD: Q[s,a] += α × (r + γ·Q[s',a'] - Q[s,a])
                        ↑
              Esto es bootstrapping: Q[s'] es una estimación, no el valor real
```

Monte Carlo NO hace bootstrapping: espera el retorno real G_t al final del episodio.

---

### Ventana deslizante (sliding window)
Técnica para suavizar la curva de entrenamiento. Promedia los últimos N episodios.

```
Sin suavizar: -1, +1, -1, +1, -1, -1, +1...  (imposible ver tendencia)
Ventana 500: -0.21, -0.18, -0.15, ...          (tendencia visible)
```

---

## 3. LA DISTINCIÓN CENTRAL — SARSA vs Q-Learning

### La diferencia está en UNA línea de código

```python
# SARSA (ON-POLICY)
a_ = e_greedy(s_, Q, epsilon)          # acción siguiente = e-greedy
Q[s][a] += α × (r + γ·Q[s_][a_] - Q[s][a])
#                         ↑ usa la acción que REALMENTE tomaré

# Q-LEARNING (OFF-POLICY)
a_ = np.argmax(Q[s_])                  # acción siguiente = la MEJOR posible
Q[s][a] += α × (r + γ·Q[s_][a_] - Q[s][a])
#                         ↑ usa la acción ÓPTIMA, no la que realmente tomará
```

### Consecuencias de esa diferencia

| | SARSA | Q-Learning |
|---|---|---|
| ¿Qué aprende Q? | La política ε-greedy que está siguiendo | La política óptima (greedy) |
| Al evaluar con ε=0 | Puede estar "entrenado" para entorno exploratorio | Preparado exactamente para ε=0 |
| Tipo | **On-policy** | **Off-policy** |
| Comportamiento | Más conservador | Más agresivo/optimista |
| Sensibilidad a ε | ε afecta DIRECTAMENTE la política final | ε solo afecta velocidad de convergencia |

### Analogía para explicarlo
> SARSA es como un conductor que aprende **mientras maneja con un GPS imperfecto** — su
> aprendizaje incluye el costo de las veces que el GPS lo mandó mal.
>
> Q-Learning es como un conductor que siempre aprende asumiendo que tomará **la ruta perfecta**,
> aunque en la práctica a veces se equivoque. Al final, su mapa mental es óptimo.

---

## 4. BLACKJACK — Conceptos específicos

### El estado es una tupla (no un número)
```
Estado = (player_sum, dealer_card, usable_ace)
         (    16    ,      7     ,      0    )

player_sum  = suma de cartas del jugador (0–31)
dealer_card = carta visible del dealer (1–10, As=1)
usable_ace  = 1 si el jugador tiene un As que cuenta como 11 sin pasarse
```

**Ejemplo concreto:** `(16, 7, 0)`
- Tengo cartas que suman 16 (ej: 9+7)
- El dealer muestra un 7
- No tengo As usable
- Situación difícil: si pido carta y saco >5, me paso (bust)

**Ejemplo con As:** `(18, 10, 1)`
- Tengo As+7 = 18 (el As vale 11)
- Si pido carta y saco >3, el As se convierte en 1 → suma pasaría a 9+carta (no bust)
- El As actúa como "red de seguridad"

---

### ¿Por qué 704 estados, no una cuadrícula?

```
Estados teóricos:  32 × 11 × 2 = 704
Estados alcanzables:  18 × 10 × 2 = 360   (suma 4-21, dealer 1-10)
Estados con decisión:  10 × 10 × 2 = 200  (suma 12-21, donde hay riesgo real)
```

**¿Por qué NO manejarlo como cuadrícula?**
- En FrozenLake: estado (2,3) tiene vecinos claros (1,3), (3,3), (2,2), (2,4)
- En Blackjack: estado (16,7,0) NO tiene vecinos. No hay "ir a la izquierda" ni "una celda abajo"
- Las transiciones dependen del mazo (probabilísticas, no deterministas)
- Por eso Q es un `defaultdict(tupla → array)`, no una matriz numpy

---

### ¿Por qué no PD en Blackjack?
PD necesita conocer `P(s'|s,a)`: la probabilidad de pasar al estado s' al pedir carta en estado s.
Esto requiere saber la distribución de cartas restantes en el mazo — información que el entorno **no expone** directamente.

---

### ¿Por qué no es directamente comparable con el MC de la actividad anterior?
| Aspecto | MC anterior | Esta actividad |
|---|---|---|
| Versión | Blackjack-**v0** | Blackjack-**v1** |
| Reward por blackjack natural | **+1.5** | **+1.0** |
| Métrica reportada | Función V (valor de estado) | Win rate (tasa de victorias) |
| Episodios | 100,000 | 50,000 |
| Comparación directa | ✗ No válida en absoluto | — |

---

## 5. PREGUNTAS FRECUENTES EN SUSTENTACIÓN

### "¿Qué es SARSA?"
> "SARSA es un algoritmo de Diferencia Temporal on-policy. On-policy significa que aprende la
> función Q de la misma política que está ejecutando — la ε-greedy. La actualización usa la
> siguiente acción **real** que tomará el agente: Q[s,a] += α(r + γ·Q[s',a'] - Q[s,a]), donde
> a' se elige con ε-greedy."

---

### "¿Qué diferencia hay con Q-Learning?"
> "Una sola diferencia: en Q-Learning, a' no es la acción que realmente tomará sino la
> **acción greedy** (argmax Q[s']). Esto hace que Q-Learning aprenda la política óptima
> independientemente de cómo explore, por eso es off-policy. SARSA aprende la política
> ε-greedy que está siguiendo, que es ligeramente subóptima al evaluarla sin exploración."

---

### "¿Por qué usaron defaultdict en vez de una matriz?"
> "Porque el estado de Blackjack es una tupla — no un índice numérico directo. No podemos
> hacer Q[16][7][0] como si fuera una matriz 3D porque muchas combinaciones son inalcanzables
> y desperdiciaríamos memoria. El defaultdict inicializa a cero solo los estados que
> realmente se visitan durante el entrenamiento."

---

### "¿Qué pasa si ε = 0?"
> "Con ε=0 el agente nunca explora. En SARSA esto es doblemente problemático: no visita
> estados nuevos Y aprende una política 100% greedy que puede quedar atrapada en óptimos
> locales. En Q-Learning también hay problema de cobertura — si un estado nunca se visita,
> su Q nunca se actualiza. Necesitamos algo de exploración para garantizar que todos los
> estados relevantes reciban actualizaciones."

---

### "¿Por qué eligieron γ = 0.999?"
> "Porque en Blackjack la única recompensa (+1 o -1) llega al final del episodio, después
> de 2 a 6 pasos. Con γ=0.999, esa recompensa llega casi sin descuento (0.999^6 ≈ 0.994).
> Con γ=0.7, la misma recompensa en 4 pasos valdría solo 0.24 → el agente subestimaría
> el valor de los estados previos y no aprendería a estrategizar a largo plazo."

---

### "¿TD es mejor que Monte Carlo?"
> "Para Blackjack son comparables, con TD siendo más eficiente en uso de episodios.
> MC actualizó Q una vez por episodio (al final). TD actualiza en cada paso — con episodios
> de 4 pasos promedio, TD hace 4× más actualizaciones por episodio con la misma experiencia.
> En nuestra actividad, TD convergió con 50k episodios a un nivel comparable al MC que
> necesitó 100k episodios."

---

### "¿Qué significa la brecha entre win rate de entrenamiento y win rate greedy?"
> "Durante entrenamiento el agente actúa con ε-greedy: un % de acciones son aleatorias
> y reducen el rendimiento. Al evaluar con ε=0 (greedy puro), siempre elige la mejor
> acción conocida. La brecha es proporcional a ε. Para SARSA la brecha es mayor porque
> Q aprendió asumiendo que habrá exploración — para Q-Learning la brecha es puramente
> el costo de las acciones aleatorias durante el entrenamiento, no un defecto de Q."

---

### "¿Cuándo usarían SARSA en la vida real?"
> "Cuando el agente aprende en el entorno real y no puede darse el lujo de tomar acciones
> exploratorias peligrosas. SARSA aprende a ser conservador porque su Q incorpora el costo
> de explorar. Ejemplo: un robot físico — si explora agresivamente puede dañarse o dañar
> personas. Con SARSA el agente aprende una política 'segura' desde las primeras etapas."

---

### "¿Y Q-Learning?"
> "Cuando se puede simular sin consecuencias reales. Q-Learning busca la política óptima
> sin importar cómo explore. Videojuegos, simulaciones logísticas, entornos de prueba.
> En Blackjack (simulado) preferimos Q-Learning porque puede fallar mil veces sin costo
> real y converge más rápido a la política óptima."

---

## 6. TABLA RESUMEN FINAL — Para tener a la vista

| Concepto | SARSA | Q-Learning |
|---|---|---|
| Tipo | On-policy | Off-policy |
| Aprende | Política ε-greedy | Política óptima Q* |
| Actualización target | Q[s', **a'**] donde a' ~ ε-greedy | **max** Q[s', :] |
| Sensibilidad a ε | Alta (afecta política final) | Baja (solo velocidad) |
| Política resultante | Conservadora | Agresiva/óptima |
| Mejor para | Entornos reales con riesgo | Simulaciones |

| Paradigma | Necesita modelo | Actualiza cuándo |
|---|---|---|
| PD | ✓ Sí | Por barrido completo |
| MC | ✗ No | Al FINAL del episodio |
| SARSA | ✗ No | Cada PASO |
| Q-Learning | ✗ No | Cada PASO |

---

## 7. NÚMEROS CLAVE DEL EXPERIMENTO (completar al correr)

| Métrica | Valor |
|---|---|
| Baseline (política aleatoria) | ~0.28–0.32 win rate |
| K episodios de entrenamiento | 50,000 |
| Ventana deslizante | 500 episodios |
| Evaluación greedy | 5,000 episodios |
| Mejor α SARSA | *(ver tabla 2.2)* |
| Mejor ε SARSA | *(ver tabla 2.3)* |
| Mejor γ SARSA | *(ver tabla 2.4)* |
| Win rate final SARSA | *(ver tabla 5)* |
| Win rate final Q-Learning | *(ver tabla 5)* |
| MC actividad anterior | 100k eps, ε=0.2, γ=0.9, Blackjack-v0 |
