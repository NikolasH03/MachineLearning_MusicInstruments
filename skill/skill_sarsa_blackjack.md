# Skill: Implementación y Análisis de SARSA en Blackjack-v1

## Objetivo
Implementar, evaluar y analizar el algoritmo SARSA (on-policy) usando aprendizaje por Diferencia Temporal en el entorno Blackjack-v1, cumpliendo criterios técnicos y conceptuales de experimentación.

---

## 1. Comprensión Conceptual

- SARSA es un algoritmo **on-policy**.
- Actualiza la función Q usando la acción que realmente sigue la política epsilon-greedy.
- Pertenece a **Diferencia Temporal (TD)**:
  - No requiere modelo del entorno
  - Actualiza en cada paso (no al final del episodio)

Ecuación:
Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]

---

## 2. Representación del Problema (Blackjack)

- Estado: tupla `(player_sum, dealer_card, usable_ace)`
- Espacio discreto, no indexable directamente

### Decisión técnica obligatoria:
- Usar `defaultdict`:
```python
Q = defaultdict(lambda: np.zeros(n_actions))
```

### Justificación esperada:
- Manejo dinámico de estados no vistos
- Evita inicializaciones manuales

---

## 3. Implementación de la Función SARSA

Firma requerida:
```python
def sarsa(env, alpha, gamma, epsilon, K):
```

### Debe incluir:
- Inicialización de Q
- Política epsilon-greedy
- Loop por episodios
- Loop por pasos
- Actualización TD (SARSA)
- Acumulación de recompensa por episodio

### Output:
- Tabla Q
- Lista de retornos por episodio

---

## 4. Primer Entrenamiento (2.1)

### Requisitos:
- Hiperparámetros arbitrarios (documentados)
- Entrenamiento completo
- Cálculo de retornos

### Visualización:
- Curva suavizada
- Ventana deslizante: 500 episodios

### Análisis esperado:
- ¿El agente mejora?
- ¿La curva es estable?
- ¿Hay ruido?

---

## 5. Ajuste de Alpha (2.2)

### Valores a probar:
- 0.1, 0.3, 0.6, 0.9

### Condiciones:
- Mantener gamma y epsilon fijos

### Visualización:
- Subplots 2x2
- Todas las curvas suavizadas

### Análisis esperado:
- Velocidad de aprendizaje
- Estabilidad
- Convergencia

### Interpretación clave:
- Alpha bajo → estable pero lento
- Alpha alto → rápido pero inestable

---

## 6. Ajuste de Epsilon (2.3)

### Valores:
- 0.05, 0.15, 0.30

### Requisitos:
- Mismo número de episodios
- Evaluación adicional

### Evaluación:
- Política greedy (sin exploración)
- 1000 episodios

### Análisis esperado:
- Velocidad de aprendizaje
- Calidad de política final

### Interpretación:
- Bajo epsilon → poca exploración
- Alto epsilon → aprendizaje ruidoso

---

## 7. Ajuste de Gamma (2.4)

### Valores:
- 0.7, 0.9, 0.99

### Requisitos:
- Usar mejor alpha y epsilon
- Evaluación en 1000 episodios

### Análisis conceptual:
- Gamma controla importancia del futuro

### Insight de Blackjack:
- Episodios cortos
- Recompensa al final

### Expectativa:
- Gamma tendrá impacto limitado

---

## 8. Buenas Prácticas

- Separar entrenamiento y evaluación
- Usar smoothing (moving average)
- Mantener experimentos controlados
- Documentar decisiones
- Interpretar resultados (no solo graficar)

---

## 9. Errores Comunes a Evitar

- Usar matriz en vez de diccionario
- Implementar Q-learning en lugar de SARSA
- Cambiar múltiples hiperparámetros simultáneamente
- No evaluar política final
- No justificar decisiones técnicas

---

## 10. Checklist de Cumplimiento

- [ ] Implementación correcta de SARSA
- [ ] Uso de defaultdict justificado
- [ ] Primer entrenamiento documentado
- [ ] Gráfica con ventana de 500
- [ ] Experimento con alpha completo
- [ ] Experimento con epsilon + evaluación
- [ ] Experimento con gamma + análisis conceptual
- [ ] Interpretación clara y justificada

