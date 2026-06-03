# Skill: Análisis On-Policy vs Off-Policy (SARSA vs Q-Learning) en Blackjack-v1

## 🎯 Objetivo
Analizar empírica y conceptualmente la diferencia entre SARSA (on-policy) y Q-Learning (off-policy) más allá de la convergencia, mediante la visualización y comparación de las políticas aprendidas.

---

## 🧠 1. Concepto Clave

- SARSA (on-policy): aprende la política que ejecuta (incluye exploración)
- Q-Learning (off-policy): aprende la política óptima independientemente de la exploración

👉 Diferencia clave:
- SARSA → política más conservadora
- Q-Learning → política más agresiva/optimista

---

## ♠️ 2. Representación de la Política en Blackjack

Estado:
(player_sum, dealer_card, usable_ace)

### Dimensiones de visualización:
- Filas: jugador (12 a 21)
- Columnas: dealer (1 a 10)
- Valor celda:
  - 'P' = pedir carta (hit)
  - 'S' = plantarse (stick)

---

## 📊 3. Tablas requeridas

Deben generarse **4 tablas en total**:

| Algoritmo | usable_ace | Tabla |
|----------|------------|------|
| SARSA | True | 1 |
| SARSA | False | 1 |
| Q-Learning | True | 1 |
| Q-Learning | False | 1 |

---

## ⚙️ 4. Construcción de la Política

A partir de Q:

```python
policy[state] = argmax_a Q[state]
```

Luego mapear:
- 0 → 'S'
- 1 → 'P'

⚠️ Solo usar política greedy (sin exploración)

---

## 🔍 5. Comparación Visual

### Preguntas obligatorias:

1. ¿Las políticas son iguales?
2. ¿Dónde difieren?
3. ¿Qué patrones aparecen?
4. ¿Cuál se parece más a la estrategia básica de Blackjack?

---

## 🧠 6. Interpretación Esperada

### SARSA:
- Más conservador
- Menos riesgo
- Evita estados peligrosos

### Q-Learning:
- Más agresivo
- Busca maximizar reward esperado
- Puede tomar más riesgos

---

## 🃏 7. Conexión con Estrategia Básica

- Estrategia básica = política casi óptima
- Q-Learning suele aproximarse mejor
- SARSA puede diferir por incorporar exploración en aprendizaje

---

## ⚠️ 8. Análisis Conceptual (epsilon = 0)

### Durante entrenamiento:

- SARSA:
  - No explora
  - Aprende solo la trayectoria inicial
  - Puede quedar atrapado en políticas subóptimas

- Q-Learning:
  - También sin exploración
  - Pero el target usa max
  - Puede propagar valores óptimos indirectamente

👉 Diferencia: Q-Learning sigue siendo más robusto

---

### Durante evaluación:

- Ambos usan política greedy
- Se comportan igual en ejecución

👉 Diferencia desaparece (solo se evalúa política final)

---

## ⚠️ Errores Comunes

- No separar usable_ace True/False
- No usar política greedy
- Confundir índices de acciones
- No analizar visualmente
- No conectar con teoría

---

## ✅ Checklist

- [ ] Generación de las 4 tablas
- [ ] Mapeo correcto de acciones (P/S)
- [ ] Uso de política greedy
- [ ] Comparación visual clara
- [ ] Identificación de diferencias
- [ ] Conexión con estrategia básica
- [ ] Respuesta conceptual sobre epsilon=0

---

## 🧠 Insight Final

Este punto evalúa comprensión profunda:

👉 SARSA aprende una política "segura"
👉 Q-Learning aprende una política "óptima"

La diferencia no está en el código, está en el comportamiento aprendido.
