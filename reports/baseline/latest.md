# Reporte de evaluación

k=5 · preguntas: 48 (dev=24, test=24) · bootstrap=2000 remuestreos, semilla=0 · IC 95 % entre corchetes.

## Ajuste de pesos (solo `dev`, n=24)

| Peso BM25 | Peso vector | MRR | Recall | nDCG |
|---|---|---|---|---|
| 0.1 | 0.9 | 0.958 | 1.00 | 0.886 |
| 0.2 | 0.8 | 0.958 | 1.00 | 0.886 |
| 0.3 | 0.7 | 0.938 | 1.00 | 0.875 |
| 0.4 | 0.6 | 0.958 | 1.00 | 0.881 |
| 0.5 | 0.5 | 0.958 | 1.00 | 0.888 | ←
| 0.6 | 0.4 | 0.958 | 1.00 | 0.871 |
| 0.7 | 0.3 | 0.917 | 1.00 | 0.843 |
| 0.8 | 0.2 | 0.917 | 1.00 | 0.840 |
| 0.9 | 0.1 | 0.896 | 1.00 | 0.831 |

## Resultados en `test` (n=24, no usado para ajustar nada)

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 normalizada |
|---|---|---|---|---|---|
| Solo BM25 (n=24) | 0.88 [0.75, 1.00] | 0.71 [0.55, 0.86] | 0.49 [0.37, 0.62] | 0.30 [0.22, 0.39] | 0.49 [0.36, 0.64] |
| Solo vectorial (n=24) | 0.96 [0.88, 1.00] | 0.84 [0.72, 0.95] | 0.70 [0.59, 0.80] | 0.47 [0.37, 0.58] | 0.73 [0.60, 0.84] |
| Híbrido 0.5/0.5 (n=24) | 1.00 [1.00, 1.00] | 0.83 [0.70, 0.95] | 0.66 [0.54, 0.77] | 0.42 [0.33, 0.52] | 0.67 [0.56, 0.79] |
| Híbrido 0.5/0.5 + filtro categoría* (n=24) | 1.00 [1.00, 1.00] | 0.86 [0.73, 0.97] | 0.68 [0.56, 0.79] | 0.43 [0.33, 0.54] | 0.69 [0.57, 0.80] |
| Solo vectorial + filtro categoría* (n=24) | 1.00 [1.00, 1.00] | 0.89 [0.79, 0.98] | 0.75 [0.64, 0.85] | 0.49 [0.39, 0.60] | 0.75 [0.64, 0.86] |

## Todo el set (n=48) — métodos sin ajuste

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 normalizada |
|---|---|---|---|---|---|
| Solo BM25 (n=48) | 0.92 [0.83, 0.98] | 0.79 [0.68, 0.88] | 0.65 [0.55, 0.74] | 0.37 [0.30, 0.44] | 0.67 [0.56, 0.77] |
| Solo vectorial (n=48) | 0.98 [0.94, 1.00] | 0.90 [0.82, 0.96] | 0.80 [0.72, 0.86] | 0.47 [0.39, 0.55] | 0.81 [0.73, 0.89] |
| Híbrido 0.5/0.5 (n=48) | 1.00 [1.00, 1.00] | 0.89 [0.82, 0.96] | 0.77 [0.69, 0.85] | 0.44 [0.38, 0.51] | 0.79 [0.71, 0.87] |
| Híbrido 0.5/0.5 + filtro categoría* (n=48) | 1.00 [1.00, 1.00] | 0.91 [0.84, 0.97] | 0.79 [0.71, 0.85] | 0.45 [0.38, 0.52] | 0.80 [0.73, 0.88] |
| Solo vectorial + filtro categoría* (n=48) | 1.00 [1.00, 1.00] | 0.92 [0.86, 0.98] | 0.83 [0.76, 0.90] | 0.49 [0.41, 0.57] | 0.84 [0.76, 0.91] |

\* categoría conocida de antemano (oráculo): cota superior del filtro, no uso real.

## Comparaciones pareadas (mismas preguntas)

Δ = a − b. `p` = fracción de remuestreos en que a no supera a b (p bajo ⇒ evidencia de que a es mejor).

| Subconjunto | a | b | Métrica | Δ | IC 95 % | p |
|---|---|---|---|---|---|---|
| test | Híbrido 0.5/0.5 | Solo vectorial | recall | +0.042 | [+0.000, +0.125] | 0.356 |
| test | Híbrido 0.5/0.5 | Solo vectorial | mrr | -0.013 | [-0.150, +0.115] | 0.576 |
| test | Híbrido 0.5/0.5 | Solo BM25 | recall | +0.125 | [+0.000, +0.250] | 0.036 |
| test | Híbrido 0.5/0.5 | Solo BM25 | mrr | +0.126 | [+0.021, +0.235] | 0.011 |
| all | Híbrido 0.5/0.5 | Solo vectorial | recall | +0.021 | [+0.000, +0.062] | 0.352 |
| all | Híbrido 0.5/0.5 | Solo vectorial | mrr | -0.003 | [-0.075, +0.069] | 0.528 |
| all | Híbrido 0.5/0.5 | Solo BM25 | recall | +0.083 | [+0.021, +0.167] | 0.018 |
| all | Híbrido 0.5/0.5 | Solo BM25 | mrr | +0.105 | [+0.030, +0.189] | 0.002 |

## Por tipo de pregunta (todo el set)

| Tipo | n | Solo BM25 (R / MRR) | Solo vectorial (R / MRR) | Híbrido 0.5/0.5 (R / MRR) |
|---|---|---|---|---|
| ambigua | 10 | 0.90 / 0.67 | 1.00 / 0.88 | 1.00 / 0.82 |
| exacto | 22 | 0.95 / 0.87 | 1.00 / 0.95 | 1.00 / 0.91 |
| parafrasis | 16 | 0.88 / 0.75 | 0.94 / 0.83 | 1.00 / 0.92 |

## Preguntas que el híbrido 0.5/0.5 no resuelve en el top-5 (0)

- ninguna
