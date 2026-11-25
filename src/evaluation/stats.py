"""Bootstrap (solo biblioteca estándar) para cuantificar la incertidumbre con pocas preguntas."""
from __future__ import annotations

import random
from statistics import fmean


def _resample_means(values: list[float], n_resamples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return sorted(fmean(rng.choices(values, k=n)) for _ in range(n_resamples))


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))]


def bootstrap_ci(values: list[float], *, n_resamples: int = 2000, seed: int = 0,
                 confidence: float = 0.95) -> tuple[float, float, float]:
    """(media, límite inferior, límite superior) por bootstrap percentil sobre las preguntas."""
    if not values:
        raise ValueError("No hay valores para estimar")
    means = _resample_means(values, n_resamples, seed)
    alpha = (1 - confidence) / 2
    return fmean(values), _percentile(means, alpha), _percentile(means, 1 - alpha)


def paired_bootstrap(a: list[float], b: list[float], *, n_resamples: int = 2000, seed: int = 0,
                     confidence: float = 0.95) -> dict[str, float]:
    """Compara dos métodos sobre las MISMAS preguntas: diferencia media a-b, su IC y `p`, la fracción de
    remuestreos en que a no supera a b (p chico = evidencia de que a es mejor)."""
    if len(a) != len(b) or not a:
        raise ValueError("Las series deben tener el mismo largo y no estar vacías")
    diffs = [x - y for x, y in zip(a, b)]
    means = _resample_means(diffs, n_resamples, seed)
    alpha = (1 - confidence) / 2
    return {
        "diff": fmean(diffs),
        "lo": _percentile(means, alpha),
        "hi": _percentile(means, 1 - alpha),
        "p": sum(1 for m in means if m <= 0) / len(means),
    }
