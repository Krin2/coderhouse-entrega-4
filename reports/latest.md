# Reporte de evaluación

k=5 · preguntas: 48 (dev=24, test=24) · bootstrap=2000 remuestreos, semilla=0 · IC 95 % entre corchetes.

## Ajuste de pesos (solo `dev`, n=24)

| Peso BM25 | Peso vector | MRR | Recall | nDCG |
|---|---|---|---|---|
| 0.1 | 0.9 | 0.979 | 1.00 | 0.888 |
| 0.2 | 0.8 | 0.979 | 1.00 | 0.889 |
| 0.3 | 0.7 | 0.979 | 1.00 | 0.889 |
| 0.4 | 0.6 | 0.979 | 1.00 | 0.891 |
| 0.5 | 0.5 | 0.979 | 1.00 | 0.897 | ←
| 0.6 | 0.4 | 0.958 | 1.00 | 0.885 |
| 0.7 | 0.3 | 0.938 | 1.00 | 0.876 |
| 0.8 | 0.2 | 0.917 | 1.00 | 0.860 |
| 0.9 | 0.1 | 0.896 | 1.00 | 0.851 |

## Resultados en `test` (n=24, no usado para ajustar nada)

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 normalizada |
|---|---|---|---|---|---|
| Solo BM25 (n=24) | 0.83 [0.67, 0.96] | 0.69 [0.52, 0.83] | 0.52 [0.39, 0.64] | 0.30 [0.21, 0.39] | 0.55 [0.41, 0.69] |
| Solo vectorial (n=24) | 1.00 [1.00, 1.00] | 0.88 [0.77, 0.97] | 0.73 [0.64, 0.83] | 0.43 [0.34, 0.52] | 0.76 [0.66, 0.85] |
| Híbrido 0.5/0.5 (n=24) | 1.00 [1.00, 1.00] | 0.83 [0.70, 0.95] | 0.68 [0.56, 0.79] | 0.39 [0.30, 0.49] | 0.72 [0.59, 0.83] |
| Híbrido 0.5/0.5 + filtro categoría* (n=24) | 1.00 [1.00, 1.00] | 0.89 [0.78, 0.98] | 0.75 [0.65, 0.84] | 0.44 [0.35, 0.54] | 0.78 [0.69, 0.88] |
| Solo vectorial + filtro categoría* (n=24) | 1.00 [1.00, 1.00] | 0.90 [0.79, 0.98] | 0.77 [0.67, 0.87] | 0.45 [0.36, 0.55] | 0.79 [0.69, 0.88] |

## Todo el set (n=48) — métodos sin ajuste

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 normalizada |
|---|---|---|---|---|---|
| Solo BM25 (n=48) | 0.90 [0.81, 0.98] | 0.77 [0.66, 0.87] | 0.67 [0.57, 0.75] | 0.36 [0.29, 0.43] | 0.70 [0.61, 0.80] |
| Solo vectorial (n=48) | 1.00 [1.00, 1.00] | 0.93 [0.86, 0.97] | 0.81 [0.74, 0.87] | 0.42 [0.36, 0.49] | 0.82 [0.75, 0.88] |
| Híbrido 0.5/0.5 (n=48) | 1.00 [1.00, 1.00] | 0.90 [0.83, 0.97] | 0.79 [0.70, 0.86] | 0.40 [0.35, 0.47] | 0.81 [0.73, 0.88] |
| Híbrido 0.5/0.5 + filtro categoría* (n=48) | 1.00 [1.00, 1.00] | 0.93 [0.86, 0.98] | 0.82 [0.75, 0.88] | 0.43 [0.38, 0.50] | 0.84 [0.78, 0.91] |
| Solo vectorial + filtro categoría* (n=48) | 1.00 [1.00, 1.00] | 0.94 [0.88, 0.98] | 0.85 [0.78, 0.91] | 0.45 [0.38, 0.52] | 0.85 [0.78, 0.91] |

\* categoría conocida de antemano (oráculo): cota superior del filtro, no uso real.

## Comparaciones pareadas (mismas preguntas)

Δ = a − b. `p` = fracción de remuestreos en que a no supera a b (p bajo ⇒ evidencia de que a es mejor).

| Subconjunto | a | b | Métrica | Δ | IC 95 % | p |
|---|---|---|---|---|---|---|
| test | Híbrido 0.5/0.5 | Solo vectorial | recall | +0.000 | [+0.000, +0.000] | 1.000 |
| test | Híbrido 0.5/0.5 | Solo vectorial | mrr | -0.046 | [-0.175, +0.079] | 0.771 |
| test | Híbrido 0.5/0.5 | Solo BM25 | recall | +0.167 | [+0.042, +0.333] | 0.013 |
| test | Híbrido 0.5/0.5 | Solo BM25 | mrr | +0.144 | [+0.062, +0.235] | 0.001 |
| all | Híbrido 0.5/0.5 | Solo vectorial | recall | +0.000 | [+0.000, +0.000] | 1.000 |
| all | Híbrido 0.5/0.5 | Solo vectorial | mrr | -0.023 | [-0.085, +0.041] | 0.761 |
| all | Híbrido 0.5/0.5 | Solo BM25 | recall | +0.104 | [+0.021, +0.188] | 0.008 |
| all | Híbrido 0.5/0.5 | Solo BM25 | mrr | +0.134 | [+0.061, +0.212] | 0.000 |

## Por tipo de pregunta (todo el set)

| Tipo | n | Solo BM25 (R / MRR) | Solo vectorial (R / MRR) | Híbrido 0.5/0.5 (R / MRR) |
|---|---|---|---|---|
| ambigua | 10 | 0.90 / 0.62 | 1.00 / 0.95 | 1.00 / 0.87 |
| exacto | 22 | 0.91 / 0.86 | 1.00 / 0.95 | 1.00 / 0.91 |
| parafrasis | 16 | 0.88 / 0.73 | 1.00 / 0.88 | 1.00 / 0.92 |

## Preguntas que el híbrido 0.5/0.5 no resuelve en el top-5 (0)

- ninguna
