# Reporte de evaluación

k=5 · preguntas: 48 (dev=24, test=24) · bootstrap=2000 remuestreos, semilla=0 · IC 95 % entre corchetes.

## Ajuste de pesos (solo `dev`, n=24)

| Peso BM25 | Peso vector | MRR | Recall | nDCG |
|---|---|---|---|---|
| 0.1 | 0.9 | 0.972 | 1.00 | 0.865 |
| 0.2 | 0.8 | 0.979 | 1.00 | 0.867 |
| 0.3 | 0.7 | 0.979 | 1.00 | 0.873 | ←
| 0.4 | 0.6 | 0.979 | 1.00 | 0.857 |
| 0.5 | 0.5 | 0.979 | 1.00 | 0.865 |
| 0.6 | 0.4 | 0.958 | 1.00 | 0.833 |
| 0.7 | 0.3 | 0.979 | 1.00 | 0.832 |
| 0.8 | 0.2 | 0.938 | 1.00 | 0.813 |
| 0.9 | 0.1 | 0.917 | 1.00 | 0.801 |

## Resultados en `test` (n=24, no usado para ajustar nada)

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 normalizada |
|---|---|---|---|---|---|
| Solo BM25 (n=24) | 0.92 [0.79, 1.00] | 0.70 [0.54, 0.85] | 0.48 [0.36, 0.60] | 0.36 [0.27, 0.46] | 0.48 [0.36, 0.60] |
| Solo vectorial (n=24) | 0.92 [0.79, 1.00] | 0.88 [0.73, 0.98] | 0.72 [0.59, 0.84] | 0.53 [0.41, 0.63] | 0.68 [0.54, 0.81] |
| Híbrido 0.5/0.5 (n=24) | 0.96 [0.88, 1.00] | 0.83 [0.69, 0.94] | 0.64 [0.52, 0.76] | 0.48 [0.38, 0.58] | 0.62 [0.50, 0.73] |
| Híbrido ajustado en dev (0.3/0.7) (n=24) | 0.96 [0.88, 1.00] | 0.82 [0.67, 0.94] | 0.68 [0.56, 0.80] | 0.53 [0.43, 0.63] | 0.68 [0.56, 0.80] |
| Híbrido 0.5/0.5 + filtro categoría* (n=24) | 1.00 [1.00, 1.00] | 0.87 [0.75, 0.97] | 0.68 [0.57, 0.79] | 0.51 [0.40, 0.62] | 0.68 [0.56, 0.80] |
| Solo vectorial + filtro categoría* (n=24) | 0.96 [0.88, 1.00] | 0.88 [0.76, 0.98] | 0.75 [0.62, 0.87] | 0.57 [0.44, 0.68] | 0.72 [0.59, 0.84] |

## Todo el set (n=48) — métodos sin ajuste

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 normalizada |
|---|---|---|---|---|---|
| Solo BM25 (n=48) | 0.96 [0.90, 1.00] | 0.80 [0.71, 0.89] | 0.63 [0.54, 0.72] | 0.43 [0.35, 0.50] | 0.64 [0.54, 0.72] |
| Solo vectorial (n=48) | 0.96 [0.90, 1.00] | 0.92 [0.85, 0.99] | 0.79 [0.71, 0.87] | 0.52 [0.44, 0.60] | 0.76 [0.67, 0.85] |
| Híbrido 0.5/0.5 (n=48) | 0.98 [0.94, 1.00] | 0.90 [0.83, 0.97] | 0.75 [0.67, 0.83] | 0.50 [0.43, 0.56] | 0.74 [0.65, 0.81] |
| Híbrido 0.5/0.5 + filtro categoría* (n=48) | 1.00 [1.00, 1.00] | 0.92 [0.86, 0.98] | 0.77 [0.69, 0.84] | 0.51 [0.44, 0.58] | 0.76 [0.68, 0.84] |
| Solo vectorial + filtro categoría* (n=48) | 0.98 [0.94, 1.00] | 0.93 [0.86, 0.99] | 0.83 [0.75, 0.90] | 0.56 [0.48, 0.65] | 0.81 [0.72, 0.88] |

\* categoría conocida de antemano (oráculo): cota superior del filtro, no uso real.

## Comparaciones pareadas (mismas preguntas)

Δ = a − b. `p` = fracción de remuestreos en que a no supera a b (p bajo ⇒ evidencia de que a es mejor).

| Subconjunto | a | b | Métrica | Δ | IC 95 % | p |
|---|---|---|---|---|---|---|
| test | Híbrido 0.5/0.5 | Solo vectorial | recall | +0.042 | [-0.083, +0.208] | 0.383 |
| test | Híbrido 0.5/0.5 | Solo vectorial | mrr | -0.049 | [-0.177, +0.062] | 0.778 |
| test | Híbrido 0.5/0.5 | Solo BM25 | recall | +0.042 | [+0.000, +0.125] | 0.360 |
| test | Híbrido 0.5/0.5 | Solo BM25 | mrr | +0.124 | [-0.020, +0.267] | 0.045 |
| all | Híbrido 0.5/0.5 | Solo vectorial | recall | +0.021 | [-0.042, +0.104] | 0.378 |
| all | Híbrido 0.5/0.5 | Solo vectorial | mrr | -0.021 | [-0.083, +0.036] | 0.739 |
| all | Híbrido 0.5/0.5 | Solo BM25 | recall | +0.021 | [+0.000, +0.062] | 0.362 |
| all | Híbrido 0.5/0.5 | Solo BM25 | mrr | +0.099 | [+0.009, +0.185] | 0.015 |
| test | Híbrido ajustado en dev (0.3/0.7) | Solo vectorial | recall | +0.042 | [-0.083, +0.208] | 0.383 |
| test | Híbrido ajustado en dev (0.3/0.7) | Solo vectorial | mrr | -0.056 | [-0.184, +0.056] | 0.814 |
| test | Híbrido ajustado en dev (0.3/0.7) | Híbrido 0.5/0.5 | recall | +0.000 | [+0.000, +0.000] | 1.000 |
| test | Híbrido ajustado en dev (0.3/0.7) | Híbrido 0.5/0.5 | mrr | -0.007 | [-0.021, +0.000] | 1.000 |

## Por tipo de pregunta (todo el set)

| Tipo | n | Solo BM25 (R / MRR) | Solo vectorial (R / MRR) | Híbrido 0.5/0.5 (R / MRR) |
|---|---|---|---|---|
| ambigua | 10 | 0.90 / 0.78 | 1.00 / 0.88 | 1.00 / 0.85 |
| exacto | 22 | 0.95 / 0.90 | 1.00 / 0.98 | 0.95 / 0.92 |
| parafrasis | 16 | 1.00 / 0.69 | 0.88 / 0.88 | 1.00 / 0.91 |

## Preguntas que el híbrido 0.5/0.5 no resuelve en el top-5 (1)

- `tutorial/cookie-params` (exacto, test): How do I read a Cookie parameter in a path operation?
