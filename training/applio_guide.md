# Guía de Entrenamiento en Applio
## Configuración exacta para RTX 5060 8GB

---

## Paso 1 — Preparar el dataset

Después de ejecutar el pipeline:
```
output/dataset/
├── audio_001.wav
├── audio_002.wav
├── ...
└── dataset_info.txt
```

Todos los audios ya están en el formato correcto (WAV, 44100Hz, mono).

---

## Paso 2 — Abrir Applio

```bash
# Si instalaste Applio con el instalador automático:
cd Applio
python app.py

# O desde el acceso directo en el escritorio
```

Interfaz en: `http://localhost:7897`

---

## Paso 3 — Configuración de entrenamiento

Ve a la pestaña **Train** y configura:

### Sección "Model Information"
| Campo | Valor |
|---|---|
| Model Name | `nombre_de_tu_modelo` |
| Sampling Rate | `40k` (si grabaste a 44100Hz) |

### Sección "Dataset"
| Campo | Valor |
|---|---|
| Dataset Path | `ruta/completa/a/output/dataset/` |

### Sección "Preprocessing"
| Campo | Valor |
|---|---|
| F0 Method | **RMVPE** ← imprescindible |
| Embedder Model | **ContentVec** (o Spin si lo tienes) |

### Sección "Training"
| Campo | Valor | Por qué |
|---|---|---|
| Total Epochs | `500-1000` | Ver TensorBoard para parar antes |
| Batch Size | `8` | Óptimo para 8GB VRAM con 30+ audios |
| Save Every Epoch | `10` | Para elegir el mejor checkpoint |
| Pretrained G | `f0G40k.pth` (incluido en Applio) | Base G generador |
| Pretrained D | `f0D40k.pth` (incluido en Applio) | Base D discriminador |

> **Si tienes <30 audios**: usa Batch Size = 4

---

## Paso 4 — TensorBoard (crucial)

Abre TensorBoard ANTES de empezar el entrenamiento:

```bash
# En otra terminal, desde la carpeta de Applio:
tensorboard --logdir logs/nombre_de_tu_modelo
```

Abre en el navegador: `http://localhost:6006`

### Qué observar:
- **Gráfico `loss/g/total`** → debe bajar y estabilizarse
- **Gráfico `loss/d/total`** → debe bajar en paralelo
- **Punto óptimo**: cuando la curva se aplana y deja de mejorar
- **Sobreentrenamiento**: si la curva sube después de bajar → para ahí

### Cuándo parar:
```
Épocas aproximadas según dataset:
- 10 audios  (~10 min) → ~600-800 épocas
- 30 audios  (~30 min) → ~400-600 épocas  
- 60 audios  (~60 min) → ~300-500 épocas
```

---

## Paso 5 — Elegir el mejor checkpoint

En Applio, una vez entrenado:
1. Ve a **Voice Conversion**
2. Prueba distintos checkpoints: `nombre_G_400.pth`, `nombre_G_500.pth`, etc.
3. Elige el que suene más natural y sin artefactos

---

## Paso 6 — Archivos que necesitas

Una vez entrenado, copia estos dos archivos a `output/models/`:

```
logs/nombre_de_tu_modelo/
├── nombre_G_XXXX.pth    ← el modelo generador (el que usas)
└── nombre.index         ← índice de características vocales
```

**Ambos archivos son necesarios** para W-Okada y para conversión batch.

---

## Solución de problemas comunes

### "CUDA out of memory"
```
→ Reduce Batch Size a 4
→ Cierra otras aplicaciones GPU (navegador, Discord con GPU acelerada)
```

### "Loss no baja después de 100 épocas"
```
→ Verifica que los audios tengan buena calidad
→ Asegúrate de que UVR5 eliminó bien el ruido
→ Revisa que todos los audios sean de la misma persona
```

### "La voz suena robótica"
```
→ El modelo puede estar sobreentrenado → prueba un checkpoint anterior
→ O puede estar subentrenado → sigue entrenando
→ Usa el índice .index con ratio 0.75 en W-Okada
```
