# Ablación: chunks fusionados vs chunks originales

Δ = chunks fusionados − chunks originales sobre las mismas preguntas (bootstrap pareado, IC 95 %). `p` = fracción de remuestreos en que no hay mejora (p bajo ⇒ la mejora es creíble).

| Subconjunto | Método | Métrica | chunks originales | chunks fusionados | Δ | IC 95 % | p |
|---|---|---|---|---|---|---|---|
| all (n=48) | Solo BM25 | recall | 0.917 | 0.896 | -0.021 | [-0.062, +0.000] | 1.000 |
| all (n=48) | Solo BM25 | mrr | 0.790 | 0.770 | -0.020 | [-0.055, +0.005] | 0.924 |
| all (n=48) | Solo BM25 | ndcg | 0.648 | 0.666 | +0.018 | [-0.016, +0.050] | 0.156 |
| all (n=48) | Solo vectorial | recall | 0.979 | 1.000 | +0.021 | [+0.000, +0.062] | 0.352 |
| all (n=48) | Solo vectorial | mrr | 0.898 | 0.927 | +0.030 | [+0.000, +0.068] | 0.043 |
| all (n=48) | Solo vectorial | ndcg | 0.796 | 0.811 | +0.015 | [-0.006, +0.042] | 0.094 |
| all (n=48) | Híbrido 0.5/0.5 | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| all (n=48) | Híbrido 0.5/0.5 | mrr | 0.895 | 0.904 | +0.009 | [-0.024, +0.050] | 0.386 |
| all (n=48) | Híbrido 0.5/0.5 | ndcg | 0.774 | 0.787 | +0.013 | [-0.022, +0.048] | 0.236 |
| all (n=48) | Híbrido 0.5/0.5 + filtro categoría* | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| all (n=48) | Híbrido 0.5/0.5 + filtro categoría* | mrr | 0.909 | 0.931 | +0.022 | [-0.003, +0.057] | 0.076 |
| all (n=48) | Híbrido 0.5/0.5 + filtro categoría* | ndcg | 0.786 | 0.823 | +0.037 | [+0.007, +0.070] | 0.006 |
| all (n=48) | Solo vectorial + filtro categoría* | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| all (n=48) | Solo vectorial + filtro categoría* | mrr | 0.923 | 0.938 | +0.015 | [+0.000, +0.044] | 0.126 |
| all (n=48) | Solo vectorial + filtro categoría* | ndcg | 0.833 | 0.845 | +0.012 | [-0.004, +0.028] | 0.059 |
| test (n=24) | Solo BM25 | recall | 0.875 | 0.833 | -0.042 | [-0.125, +0.000] | 1.000 |
| test (n=24) | Solo BM25 | mrr | 0.706 | 0.685 | -0.020 | [-0.071, +0.016] | 0.787 |
| test (n=24) | Solo BM25 | ndcg | 0.493 | 0.520 | +0.027 | [-0.021, +0.074] | 0.121 |
| test (n=24) | Solo vectorial | recall | 0.958 | 1.000 | +0.042 | [+0.000, +0.125] | 0.356 |
| test (n=24) | Solo vectorial | mrr | 0.844 | 0.875 | +0.031 | [+0.000, +0.083] | 0.136 |
| test (n=24) | Solo vectorial | ndcg | 0.701 | 0.734 | +0.032 | [-0.004, +0.076] | 0.050 |
| test (n=24) | Híbrido 0.5/0.5 | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| test (n=24) | Híbrido 0.5/0.5 | mrr | 0.831 | 0.829 | -0.002 | [-0.065, +0.060] | 0.648 |
| test (n=24) | Híbrido 0.5/0.5 | ndcg | 0.660 | 0.677 | +0.016 | [-0.047, +0.082] | 0.298 |
| test (n=24) | Híbrido 0.5/0.5 + filtro categoría* | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| test (n=24) | Híbrido 0.5/0.5 + filtro categoría* | mrr | 0.859 | 0.889 | +0.030 | [+0.000, +0.081] | 0.042 |
| test (n=24) | Híbrido 0.5/0.5 + filtro categoría* | ndcg | 0.677 | 0.746 | +0.070 | [+0.019, +0.127] | 0.001 |
| test (n=24) | Solo vectorial + filtro categoría* | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| test (n=24) | Solo vectorial + filtro categoría* | mrr | 0.894 | 0.896 | +0.002 | [+0.000, +0.006] | 0.356 |
| test (n=24) | Solo vectorial + filtro categoría* | ndcg | 0.754 | 0.773 | +0.020 | [-0.003, +0.044] | 0.049 |
