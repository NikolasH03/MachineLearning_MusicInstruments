# Skill: Evaluación Final y Comparación Global (SARSA vs Q-Learning) en Blackjack-v1

## 🎯 Objetivo
Realizar una comparación final rigurosa entre SARSA y Q-Learning usando sus **mejores hiperparámetros individuales**, evaluando desempeño, velocidad de convergencia, sensibilidad a parámetros y aplicabilidad en escenarios reales.

---

## 🔗 0. Dependencias Previas

Variables requeridas de puntos anteriores:

```python
BEST_ALPHA_SARSA
BEST_EPSILON_SARSA
BEST_GAMMA_SARSA

BEST_ALPHA_Q
BEST_EPSILON_Q
BEST_GAMMA_Q

K_EPISODES
```

✅ Cada valor debe estar **justificado previamente** en puntos 2 y 3

---

## 🧠 1. Concepto Clave de este Punto

A diferencia del Punto 3:

- ❌ Ya no se comparan en iguales condiciones
- ✅ Se comparan en su mejor versión

👉 Pregunta implícita:
> ¿Qué algoritmo rinde mejor cuando está correctamente optimizado?

---

## ⚙️ 2. Entrenamiento Final

### Requisitos:
- Entrenar SARSA con sus mejores hiperparámetros
- Entrenar Q-Learning con los suyos

⚠️ Condición crítica:
- Los hiperparámetros **pueden ser distintos**

---

## 📊 3. Evaluación de Políticas

### Requisito obligatorio:

- Evaluar cada política en:
```text
1000 episodios
```

### Condición:
- ε = 0 (sin exploración)

👉 Se evalúa la política greedy

---

## 📈 4. Gráfica Comparativa Final

### Requisitos:
- UNA única figura
- Curvas:
  - Mejor SARSA
  - Mejor Q-Learning
- Ventana de suavizado: 500
- Leyenda clara

---

## 🔍 5. Interpretación de Resultados

Debe permitir analizar:

- Desempeño final (reward promedio)
- Velocidad de aprendizaje
- Estabilidad

---

## 🧠 6. Preguntas Obligatorias (Análisis)

---

### 🔴 6.1 Desempeño Final

- ¿Cuál algoritmo obtuvo mejor reward promedio?

✅ Requiere:
- Uso de evaluación (1000 episodios)
- Comparación cuantitativa

---

### 🔴 6.2 Velocidad de Convergencia

- ¿Cuál converge más rápido?
- ¿En qué iteración se estabiliza cada uno?

✅ Requiere:
- Uso de gráfica suavizada
- Identificación de punto de estabilización

---

### 🔴 6.3 Sensibilidad a Alpha

- ¿Cuál algoritmo fue más sensible?

✅ Interpretación esperada:

- Q-Learning:
  - usa max → mayor varianza
  - más sensible a alpha

- SARSA:
  - más estable

---

### 🔴 6.4 Aplicaciones Reales

#### SARSA:
- Preferido cuando:
  - el riesgo durante entrenamiento importa
  - el entorno es real

Ejemplos:
- robótica
- conducción autónoma

#### Q-Learning:
- Preferido cuando:
  - se puede simular sin riesgo
  - se busca optimalidad

Ejemplos:
- videojuegos
- simulaciones

---

### 🔴 6.5 Comparación con otros métodos

#### vs Monte Carlo

Ventajas:
- no espera final del episodio
- menor varianza
- aprendizaje más rápido

---

#### vs Programación Dinámica

Ventajas:
- no requiere modelo del entorno
- aplicable a entornos desconocidos

---

## ⚠️ Errores Comunes

- Usar mismos hiperparámetros
- No evaluar con ε = 0
- No justificar respuestas
- No usar resultados previos
- Respuestas genéricas sin análisis

---

## ✅ Checklist

- [ ] Uso de mejores hiperparámetros individuales
- [ ] Entrenamiento correcto de ambos agentes
- [ ] Evaluación con 1000 episodios
- [ ] Política greedy en evaluación
- [ ] Gráfica única con ventana 500
- [ ] Comparación cuantitativa clara
- [ ] Análisis de convergencia
- [ ] Explicación de sensibilidad a alpha
- [ ] Aplicación en escenarios reales
- [ ] Comparación con MC y DP

---

## 🧠 Insight Final

Este punto evalúa pensamiento avanzado en RL:

👉 Comparar algoritmos en su mejor configuración
👉 Interpretar comportamiento más allá del código
👉 Conectar teoría con aplicaciones reales

