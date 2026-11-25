# Ablación: con código vs sin código

Δ = con código − sin código sobre las mismas preguntas (bootstrap pareado, IC 95 %). `p` = fracción de remuestreos en que no hay mejora (p bajo ⇒ la mejora es creíble).

| Subconjunto | Método | Métrica | sin código | con código | Δ | IC 95 % | p |
|---|---|---|---|---|---|---|---|
| all (n=48) | Solo BM25 | recall | 0.917 | 0.958 | +0.042 | [+0.000, +0.104] | 0.144 |
| all (n=48) | Solo BM25 | mrr | 0.790 | 0.804 | +0.014 | [-0.044, +0.074] | 0.319 |
| all (n=48) | Solo BM25 | ndcg | 0.648 | 0.634 | -0.014 | [-0.061, +0.029] | 0.729 |
| all (n=48) | Solo vectorial | recall | 0.979 | 0.958 | -0.021 | [-0.062, +0.000] | 1.000 |
| all (n=48) | Solo vectorial | mrr | 0.898 | 0.924 | +0.026 | [-0.026, +0.083] | 0.194 |
| all (n=48) | Solo vectorial | ndcg | 0.796 | 0.792 | -0.004 | [-0.045, +0.035] | 0.570 |
| all (n=48) | Híbrido 0.5/0.5 | recall | 1.000 | 0.979 | -0.021 | [-0.062, +0.000] | 1.000 |
| all (n=48) | Híbrido 0.5/0.5 | mrr | 0.895 | 0.903 | +0.008 | [-0.030, +0.048] | 0.338 |
| all (n=48) | Híbrido 0.5/0.5 | ndcg | 0.774 | 0.753 | -0.021 | [-0.060, +0.016] | 0.865 |
| all (n=48) | Híbrido 0.5/0.5 + filtro categoría* | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| all (n=48) | Híbrido 0.5/0.5 + filtro categoría* | mrr | 0.909 | 0.923 | +0.014 | [+0.000, +0.038] | 0.128 |
| all (n=48) | Híbrido 0.5/0.5 + filtro categoría* | ndcg | 0.786 | 0.774 | -0.012 | [-0.044, +0.020] | 0.768 |
| all (n=48) | Solo vectorial + filtro categoría* | recall | 1.000 | 0.979 | -0.021 | [-0.062, +0.000] | 1.000 |
| all (n=48) | Solo vectorial + filtro categoría* | mrr | 0.923 | 0.928 | +0.005 | [-0.042, +0.052] | 0.446 |
| all (n=48) | Solo vectorial + filtro categoría* | ndcg | 0.833 | 0.826 | -0.007 | [-0.042, +0.024] | 0.642 |
| test (n=24) | Solo BM25 | recall | 0.875 | 0.917 | +0.042 | [+0.000, +0.125] | 0.367 |
| test (n=24) | Solo BM25 | mrr | 0.706 | 0.702 | -0.003 | [-0.092, +0.092] | 0.561 |
| test (n=24) | Solo BM25 | ndcg | 0.493 | 0.481 | -0.012 | [-0.083, +0.052] | 0.637 |
| test (n=24) | Solo vectorial | recall | 0.958 | 0.917 | -0.042 | [-0.125, +0.000] | 1.000 |
| test (n=24) | Solo vectorial | mrr | 0.844 | 0.875 | +0.031 | [-0.073, +0.146] | 0.296 |
| test (n=24) | Solo vectorial | ndcg | 0.701 | 0.722 | +0.020 | [-0.051, +0.093] | 0.265 |
| test (n=24) | Híbrido 0.5/0.5 | recall | 1.000 | 0.958 | -0.042 | [-0.125, +0.000] | 1.000 |
| test (n=24) | Híbrido 0.5/0.5 | mrr | 0.831 | 0.826 | -0.005 | [-0.032, +0.024] | 0.643 |
| test (n=24) | Híbrido 0.5/0.5 | ndcg | 0.660 | 0.642 | -0.019 | [-0.080, +0.040] | 0.717 |
| test (n=24) | Híbrido 0.5/0.5 + filtro categoría* | recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.000 |
| test (n=24) | Híbrido 0.5/0.5 + filtro categoría* | mrr | 0.859 | 0.866 | +0.007 | [+0.000, +0.021] | 0.364 |
| test (n=24) | Híbrido 0.5/0.5 + filtro categoría* | ndcg | 0.677 | 0.683 | +0.006 | [-0.044, +0.058] | 0.408 |
| test (n=24) | Solo vectorial + filtro categoría* | recall | 1.000 | 0.958 | -0.042 | [-0.125, +0.000] | 1.000 |
| test (n=24) | Solo vectorial + filtro categoría* | mrr | 0.894 | 0.883 | -0.010 | [-0.094, +0.073] | 0.640 |
| test (n=24) | Solo vectorial + filtro categoría* | ndcg | 0.754 | 0.754 | +0.000 | [-0.068, +0.061] | 0.478 |
