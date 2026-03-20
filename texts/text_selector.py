"""
texts/text_selector.py
Selector de textos para el dataset de entrenamiento RVC.
Soporta dos fuentes:
  - es_dataset.txt  : frases cortas variadas (cobertura fonetica amplia)
  - es_guiones.txt  : guiones narrativos de ~1 minuto (tono y ritmo natural)

Modos de seleccion:
  auto    -> guiones para 10/30, mixto para 60
  frases  -> solo frases cortas
  guiones -> solo guiones narrativos
  mixto   -> 2/3 guiones + 1/3 frases (mejor cobertura total)
"""

import random
from pathlib import Path
from rich.console import Console

console = Console()

TEXTS_DIR = Path(__file__).parent

# Archivos por idioma
LANG_FRASES  = {"ES": "es_dataset.txt",  "EN": "en_dataset.txt"}
LANG_GUIONES = {"ES": "es_guiones.txt",  "EN": "es_guiones.txt"}  # fallback ES si no hay EN

# Fonemas clave espanol para cobertura
ES_KEY_PHONEMES = list("aeiouabcdfghjklmnopqrstvwxyz")


# ─── Carga ────────────────────────────────────────────────────────────────────

def _load_file(filepath: Path) -> list:
    if not filepath.exists():
        return []
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(line)
    console.print(f"  [dim]Cargado:[/dim] [green]{len(items)} textos[/green] desde {filepath.name}")
    return items


def _load_frases(language: str = "ES") -> list:
    fname = LANG_FRASES.get(language, "es_dataset.txt")
    texts = _load_file(TEXTS_DIR / fname)
    if not texts:
        texts = _load_file(TEXTS_DIR / "es_dataset.txt")  # fallback
    return texts


def _load_guiones(language: str = "ES") -> list:
    fname = LANG_GUIONES.get(language, "es_guiones.txt")
    texts = _load_file(TEXTS_DIR / fname)
    if not texts:
        texts = _load_file(TEXTS_DIR / "es_guiones.txt")  # fallback
    return texts


# ─── Selector principal ───────────────────────────────────────────────────────

def select_texts(language: str = "ES", count: int = 30, mode: str = "auto") -> list:
    """
    Selecciona textos para el dataset de entrenamiento.

    Args:
        language : ES, EN, FR, ZH, JP, KR
        count    : 10, 30 o 60 audios
        mode     : 'auto' | 'frases' | 'guiones' | 'mixto'

    Returns:
        Lista de textos ordenados de menor a mayor longitud
        (el modelo RVC aprende mejor empezando con textos cortos)
    """
    frases  = _load_frases(language)
    guiones = _load_guiones(language)

    # Fallback si falta alguno
    if not frases and not guiones:
        console.print("  [yellow]Usando textos de emergencia[/yellow]")
        return _fallback(count)
    if not frases:
        frases = guiones[:]
    if not guiones:
        guiones = frases[:]

    # Resolver modo auto
    if mode == "auto":
        mode = "mixto" if count >= 60 else "guiones"

    # Seleccion segun modo
    if mode == "frases":
        pool = frases
        selected = _pick(pool, count)

    elif mode == "guiones":
        pool = guiones
        selected = _pick(pool, count)

    elif mode == "mixto":
        n_g = (count * 2) // 3        # 2/3 guiones
        n_f = count - n_g              # 1/3 frases
        selected = _pick(guiones, n_g) + _pick(frases, n_f)
        random.shuffle(selected)

    else:
        # fallback: todo junto
        selected = _pick(guiones + frases, count)

    # Ordenar: cortos primero, largos al final
    selected = sorted(selected, key=len)

    # Log cobertura fonetica
    cov = _coverage(selected, language)
    console.print(
        f"  [dim]Modo:[/dim] [cyan]{mode}[/cyan]  "
        f"[dim]Cobertura fonética:[/dim] [green]{cov:.0%}[/green]"
    )

    return selected


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pick(pool: list, n: int) -> list:
    """Elige n elementos sin repetir. Si n > pool, permite repeticion controlada."""
    if not pool:
        return []
    if n <= len(pool):
        return random.sample(pool, n)
    # Completar con vueltas adicionales
    result = []
    while len(result) < n:
        batch = pool[:]
        random.shuffle(batch)
        result.extend(batch)
    return result[:n]


def _coverage(texts: list, language: str) -> float:
    """Porcentaje de fonemas clave cubiertos por el conjunto de textos."""
    target = ES_KEY_PHONEMES  # usar espanol para todos por ahora
    combined = " ".join(texts).lower()
    covered = sum(1 for p in target if p in combined)
    return covered / len(target)


def _fallback(count: int) -> list:
    base = [
        "Buenos dias, como estas hoy.",
        "El tiempo libre es muy valioso para todos nosotros.",
        "Necesito confirmar los detalles de la reunion de manana.",
        "La tecnologia avanza a pasos enormes cada dia que pasa.",
        "Estoy muy contento con los resultados que hemos obtenido.",
        "El proyecto esta avanzando muy bien segun lo planificado.",
        "Manana tenemos una presentacion muy importante para el equipo.",
        "Los datos confirman las hipotesis que planteamos al inicio.",
        "Gracias por tu tiempo y tu dedicacion constante al trabajo.",
        "Cuando era pequeno, siempre me gustaba dibujar y sonar.",
    ]
    return _pick(base, count)


# ─── Info ────────────────────────────────────────────────────────────────────

def dataset_info(language: str = "ES") -> dict:
    """Estadisticas del dataset disponible."""
    f = _load_frases(language)
    g = _load_guiones(language)
    return {
        "frases_disponibles":    len(f),
        "guiones_disponibles":   len(g),
        "total":                 len(f) + len(g),
        "duracion_estimada_min": round(len(g) * 1.0 + len(f) * 0.12, 1),
    }