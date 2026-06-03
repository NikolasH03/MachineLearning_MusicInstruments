# Skill: Conexión con Programación Dinámica y Monte Carlo en Blackjack-v1

## 🎯 Objetivo
Integrar y comparar los paradigmas de Programación Dinámica (PD), Monte Carlo (MC) y Diferencia Temporal (TD: SARSA y Q-Learning), usando evidencia experimental en Blackjack.

---

## 🔗 0. Dependencias Previas (CRÍTICO)

Debe existir:

```python
MC_SUCCESS_RATE
MC_MEAN_REWARD

BEST_SARSA_REWARDS
BEST_Q_REWARDS

RANDOM_POLICY_REWARD
PD_REFERENCE (si aplica)
```

✅ Los valores de Monte Carlo deben provenir de la actividad anterior

---

## 🧠 1. Contexto Conceptual

| Paradigma | Característica clave |
|-----------|---------------------|
| PD | Requiere modelo del entorno |
| MC | Aprende al final del episodio |
| TD | Aprende en cada paso |

👉 Este punto evalúa integración y comparación global

---

## 📊 2. Gráfica Comparativa Multinivel

### Requisitos:

Graficar:
- Mejor SARSA
- Mejor Q-Learning

---

### ➕ Líneas horizontales punteadas:

- Política aleatoria
- Monte Carlo
- Programación Dinámica (si disponible)

---

### ⚙️ Detalles técnicos:

- Ventana: 500 episodios
- Figura única
- Leyenda clara

Ejemplo:
```python
plt.axhline(y=MC_MEAN_REWARD, linestyle='--', label='Monte Carlo')
```

---

## 🎯 3. Propósito de la Gráfica

Responder:

> ¿Cuántos episodios necesita TD para alcanzar los niveles de MC o baseline?

---

## 🔍 4. Análisis Esperado

- Velocidad de convergencia
- Episodios necesarios para alcanzar referencias
- Comparación SARSA vs Q-Learning

---

## 🧠 Insight esperado

- TD converge más rápido que MC
- TD no requiere modelo (ventaja vs PD)
- Balance óptimo entre eficiencia y aplicabilidad

---

## 📋 5. Tabla Comparativa

| Criterio | PD | Monte Carlo | SARSA | Q-Learning |
|----------|----|-------------|--------|------------|
| Necesita modelo | Sí | No | No | No |
| Aprende de experiencia | No | Sí | Sí | Sí |
| Actualiza al final del episodio | No | Sí | No | No |
| Actualiza en cada paso | Sí | No | Sí | Sí |
| On-policy / Off-policy | N/A | On | On | Off |
| Tasa de éxito | (valor real) | (valor real) | (valor real) | (valor real) |
| Recompensa media final | (valor real) | (valor real) | (valor real) | (valor real) |

---

### ⚠️ Requisito crítico

- Usar valores reales obtenidos
- No estimaciones

---

## 🧠 6. Interpretación de la Tabla

Debe incluir:
- Diferencias estructurales
- Impacto en resultados
- Relación con Blackjack

---

## ✍️ 7. Conclusión Final

Responder:

> ¿Cuándo usar cada paradigma?

---

### PD
- Cuando el modelo es conocido
- Problemas pequeños

### Monte Carlo
- Episodios completos
- Simulación

### SARSA
- Riesgo durante entrenamiento importa

### Q-Learning
- Simulación segura
- Búsqueda de optimalidad

---

## ⚠️ 8. Errores Comunes

- No incluir referencias MC
- No graficar líneas horizontales
- Tabla incompleta
- Valores inventados

---

## ✅ 9. Checklist

- [ ] Resultados MC correctamente integrados
- [ ] Gráfica con líneas de referencia
- [ ] Ventana de 500 episodios
- [ ] Tabla completada con datos reales
- [ ] Análisis comparativo
- [ ] Conclusión aplicada a problemas reales

---

## 🧠 Insight Final

PD → teórico
MC → simple
TD → práctico

👉 TD es el mejor compromiso para problemas reales

