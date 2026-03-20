# Guía de Tiempo Real con W-Okada
## Tu voz entra → Modelo PTH procesa → Voz clonada sale

---

## Requisitos previos

1. **Tu modelo entrenado**: `output/models/nombre_G_XXXX.pth` + `nombre.index`
2. **W-Okada Voice Changer**: https://github.com/w-okada/voice-changer
3. **VB-Audio Virtual Cable**: https://vb-audio.com/Cable/ (gratis)

---

## Paso 1 — Instalar VB-Audio Virtual Cable

1. Descarga desde https://vb-audio.com/Cable/
2. Ejecuta el instalador como administrador
3. Reinicia el computador
4. Verifica en Panel de Control → Sonido:
   - Input: `CABLE Output (VB-Audio Virtual Cable)`
   - Output: `CABLE Input (VB-Audio Virtual Cable)`

---

## Paso 2 — Instalar W-Okada

```bash
# Opción A: descarga el release precompilado (más fácil)
# Ir a: https://github.com/w-okada/voice-changer/releases
# Descargar: voice-changer_win_nvidia.zip
# Extraer y ejecutar: start_http.bat

# Opción B: desde código fuente
git clone https://github.com/w-okada/voice-changer.git
cd voice-changer
# Seguir el README del repo
```

Interfaz en: `http://localhost:18888`

---

## Paso 3 — Cargar tu modelo en W-Okada

1. Abre `http://localhost:18888`
2. En la sección **Model Setting**:
   - **Framework**: `PyTorch` (para .pth) o `ONNX` (para .onnx)
   - **Model (.pth)**: arrastra tu `nombre_G_XXXX.pth`
   - **Index (.index)**: arrastra tu `nombre.index`
3. Clic en **Upload**

---

## Paso 4 — Configuración de rendimiento (RTX 5060)

### Parámetros críticos de latencia:

| Parámetro | Valor recomendado | Descripción |
|---|---|---|
| **CHUNK** | `256` | Tamaño del buffer de audio procesado |
| **EXTRA** | `8192` | Contexto adicional para mejor calidad |
| **F0 Method** | `RMVPE` | Debe coincidir con el usado en entrenamiento |
| **Index Ratio** | `0.75` | Balance timbre original/timbre modelo |
| **Noise Reduction** | `10-20` | Elimina ruido del micrófono |

> **Latencia esperada en RTX 5060**: ~100-150ms con CHUNK=256
> Si hay cortes o artefactos: sube CHUNK a 512 (más latencia pero más estable)

---

## Paso 5 — Configurar audio I/O

### Entrada (tu micrófono):
```
Input Device → tu micrófono físico
```

### Salida (cable virtual):
```
Output Device → CABLE Input (VB-Audio Virtual Cable)
```

---

## Paso 6 — Routing según uso

### Para Discord / TeamSpeak:
```
Discord → Configuración → Voz → Dispositivo de entrada:
→ Seleccionar "CABLE Output (VB-Audio Virtual Cable)"
```

### Para OBS Studio:
```
OBS → Fuentes → Captura de entrada de audio:
→ Seleccionar "CABLE Output (VB-Audio Virtual Cable)"
```

### Para grabar directamente:
```
Cualquier DAW (Audacity, FL Studio, etc.)
→ Input: "CABLE Output (VB-Audio Virtual Cable)"
```

---

## Paso 7 — Activar y probar

1. Clic en **START** en W-Okada
2. Habla por tu micrófono
3. Deberías escuchar tu voz convertida en tiempo real

---

## Monitoreo de rendimiento

W-Okada muestra en tiempo real:
- **Latency**: latencia actual en ms (objetivo: <200ms)
- **Response Time**: tiempo de procesamiento por chunk
- **GPU Usage**: uso de VRAM

Si la latencia es alta:
```
1. Cierra otras aplicaciones GPU
2. Sube CHUNK a 512 o 1024
3. Baja EXTRA a 4096
4. Si persiste → exporta a .onnx (ver utils/export_onnx.py)
```

---

## Exportar a .ONNX desde W-Okada (para menor latencia)

1. Con el modelo .PTH cargado
2. Clic en **Export to ONNX**
3. Guarda el archivo resultante
4. Carga el .onnx en lugar del .pth
5. En Framework selecciona **ONNX**

> El .onnx suele ser 20-40% más rápido pero con ligera pérdida de calidad.
> El .pth tiene mejor calidad para la mayoría de voces.

---

## Flujo completo resumido

```
Micrófono físico
      ↓
W-Okada Voice Changer (procesa con tu .PTH en GPU)
      ↓
CABLE Input (VB-Audio Virtual Cable)
      ↓
Discord / OBS / DAW escuchan "CABLE Output"
      ↓
Todos escuchan la voz del modelo clonado
```
