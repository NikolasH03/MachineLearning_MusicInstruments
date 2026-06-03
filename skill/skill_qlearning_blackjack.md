# Skill: Implementación y Análisis de Q-Learning en Blackjack-v1

## 🎯 Objetivo
Implementar Q-Learning (off-policy), compararlo rigurosamente con SARSA bajo condiciones controladas y analizar diferencias en aprendizaje, convergencia y sensibilidad a hiperparámetros.

---

## 🔗 0. Dependencias del Punto 2 (CRÍTICO)

Variables necesarias provenientes de SARSA:

```python
BEST_ALPHA_SARSA
BEST_EPSILON_SARSA
BEST_GAMMA_SARSA
K_EPISODES
```

✅ Estos valores deben estar definidos y justificados previamente.

---

## 🧠 1. Comprensión Conceptual

### Q-Learning (off-policy)
- Aprende la política óptima independientemente de la política de exploración
- Usa el valor máximo futuro como referencia

Ecuación:
```text
Q(s,a) ← Q(s,a) + α [r + γ max_a Q(s',a) − Q(s,a)]
```

---

## 🔥 Diferencia clave con SARSA

| Algoritmo | Tipo | Target |
|----------|------|--------|
| SARSA | On-policy | Q(s', a') |
| Q-Learning | Off-policy | max Q(s', a) |

👉 Q-Learning es más agresivo / optimista
👉 SARSA es más conservador / estable

---

## 🧩 2. Implementación de Q-Learning

### Firma OBLIGATORIA
```python
def qlearning(env, Alpha, gamma, epsilon, K):
```

⚠️ No modificar nombres

### Componentes requeridos:
- defaultdict para Q
- política epsilon-greedy
- loop episodios
- loop pasos
- actualización TD con max
- acumulación de recompensas

### Actualización clave:
```python
best_next = np.max(Q[next_state])
td_target = reward + gamma * best_next
```

---

## 📊 3.1 Comparación Directa con SARSA

### Configuración (idéntica):
- Alpha = BEST_ALPHA_SARSA
- epsilon = BEST_EPSILON_SARSA
- gamma = BEST_GAMMA_SARSA
- K = mismo número de episodios

### Condiciones experimentales:
- Misma semilla
- Mismo entorno
- Misma política

### Visualización:
- Una sola gráfica
- Ambas curvas
- Ventana suavizado: 500

### Pregunta:
- ¿Q-Learning converge más rápido, lento o igual?

### Análisis esperado:
- Velocidad inicial
- Estabilidad
- Nivel final

---

## ⚙️ 3.2 Ajuste de Alpha

### Valores:
```python
[0.1, 0.3, 0.6, 0.9]
```

### Condiciones:
- gamma fijo
- epsilon fijo

### Visualización:
- Subplots 2x2
- Ventana 500

### Preguntas:
1. ¿Mejor Alpha coincide con SARSA?
2. ¿Por qué diferiría?

### Explicación mecanística:
- Q-Learning: usa max → sobreestimación
- Targets más variables
- Alpha alto → oscilaciones

---

## 🎯 3.3 Ajuste de Epsilon

### Valores:
```python
[0.05, 0.15, 0.30]
```

### Condiciones:
- Usar BEST_ALPHA_Q
- gamma fijo

### Visualización:
- Una figura con 3 curvas o subplot 1x3

### Evaluación adicional:
- Política greedy
- 1000 episodios

### Preguntas:
- ¿Qué epsilon aprende más rápido?
- ¿Cuál da mejor política final?
- ¿Coincide con SARSA?

---

## ⚠️ Gaps y Ajustes

| Problema | Solución |
|---------|--------|
| Dependencia Punto 2 | Definir variables explícitas |
| Firma estricta | Respetar nombres |
| 3.3 incompleto | Añadir evaluación |
| Comparación justa | Fijar semilla |

---

## ✅ Buenas Prácticas

- Experimentos controlados
- Separar entrenamiento y evaluación
- Usar smoothing consistente
- Documentar decisiones
- Explicar causalidad

---

## ❌ Errores Comunes

- Cambiar hiperparámetros entre algoritmos
- No usar misma semilla
- Implementar mal el target
- No comparar correctamente
- Falta de análisis

---

## ✅ Checklist

- [ ] Implementación correcta Q-Learning
- [ ] Firma exacta respetada
- [ ] Comparación directa válida
- [ ] Gráfica única en 3.1
- [ ] Análisis de convergencia
- [ ] Barrido alpha completo
- [ ] Explicación mecanística
- [ ] Barrido epsilon completo
- [ ] Evaluación política final
- [ ] Comparación con SARSA

---

## 🧠 Insight Final

Punto 3 = experimentación controlada + análisis profundo.

Si cambias condiciones → invalida resultados.
Si no explicas → pierdes valor del ejercicio.
