# CNN para la Detección de Patologías en Plantas

Este proyecto implementa una CNN para clasificar enfermedades foliares en manzanos, utilizando el dataset de [Plant Pathology 2020](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/overview).

El objetivo era construir un modelo robusto para diferenciar 4 clases de hojas (Sana, Múltiples Enfermedades, Roya, Sarna). El modelo final alcanzó una **precisión de 63.80%** en el set de pruebas (30% de los datos).

* **Resultados de Experimentos:** [Weights & Biases Dashboard](https://wandb.ai/emilio-/Plant-Pathology-CNN?nw=nwuseremiliosoto)

---

## Proceso de Experimentación

El desarrollo se centró en encontrar la mejor combinación de arquitectura y estrategia de *fine-tuning*.

### Intento 1: EfficientNetV2S

Se probó inicialmente con `EfficientNetV2S`. El entrenamiento fue problemático:
1.  El uso de `class_weight` para balancear las clases "envenenaba" el entrenamiento y no permitía la convergencia.
2.  Incluso sin `class_weight`, el modelo se estancó rápidamente con una precisión de validación de solo **~51%**, que se descartó por su bajo rendimiento.

### Intento 2: VGG16 

Se cambió a una arquitectura `VGG16`, que demostró ser mucho más prometedora.

1.  **Entrenamiento del Clasificador (Head):** Se entrenó el modelo base congelado. El análisis en W&B mostró que alcanzó un pico de `val_accuracy` de **~61%**, pero el entrenamiento continuó 100 épocas sin `EarlyStopping`, sobreajustando y guardando un modelo final con solo **59.22%**.
2.  **Diagnóstico de Fine-Tuning:** Un intento previo de *fine-tuning* (registrado en W&B) falló por completo; el `val_loss` subió, mostrando olvido por un *learning rate* demasiado alto.
3.  **Estrategia Final (Fine-Tuning exitoso):** Se cargó el modelo sobreajustado de 59.22%, se descongeló la red completa (`model.trainable = True`) y se re-compiló con un LR ultra-bajo de `1e-6`.
    * Este método permitió al modelo pulir sus pesos lentamente sin destruir el conocimiento previo.
    * El entrenamiento fue lento pero estable, superando el pico anterior y alcanzando un nuevo `val_accuracy` estable.

---

## Evaluación Final

Se tomó el mejor modelo `VGG16` (el último descrito) y se evaluó contra el **set de `test` (30%)** que nunca había visto.

**Resultados Finales:**
* **Pérdida (Loss) en Test: 0.9501**
* **Precisión (Accuracy) en Test: 63.80%**
