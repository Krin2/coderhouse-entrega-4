# Sistema RAG escalable con Pinecone

Módulo de recuperación híbrido (BM25 + vectorial) sobre Pinecone Serverless, con la documentación Markdown de FastAPI como corpus.

## Arquitectura

- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dim, cosine).
- **Índice:** Pinecone Serverless (AWS `us-east-1`), namespace `docs-fastapi`.
- **Chunking:** `RecursiveCharacterTextSplitter` por tokens (600 tokens, overlap 80); los chunks de menos de 200 tokens se fusionan con su vecino hasta un máximo de 800.
- **Metadata por chunk:** `text` (contenido original), `source`, `category`, `doc_id`, `chunk_id`, `chunk_index` (posición en el documento) y `section` (encabezado Markdown del chunk).
- **Híbrido:** `EnsembleRetriever` (BM25 0.5 + Pinecone 0.5). El corpus BM25 se reconstruye desde la metadata de Pinecone (caché en `data/cache/`, que se invalida sola: `ingest` la borra y, además, se descarta si su tamaño no coincide con el namespace de Pinecone), sin otra base de datos.

```
src/
├─ settings.py          Settings (dataclass inmutable; el entorno se lee con Settings.from_env())
├─ infra/index.py       cliente Pinecone y creación idempotente del índice Serverless
├─ ingestion/
│  ├─ loaders.py        MarkdownLoader · PdfLoader · JsonLoader · CompositeLoader (protocolo Loader)
│  ├─ chunking.py       chunks por tokens + metadata (chunk_id, chunk_index, section)
│  └─ pipeline.py       IngestPipeline: loader -> chunking -> reset namespace -> Pinecone
├─ retrieval/
│  ├─ corpus.py         PineconeCorpus: corpus BM25 desde Pinecone, con caché que se invalida sola
│  ├─ lexical.py        BM25 + tokenizador
│  └─ hybrid.py         HybridRetriever (EnsembleRetriever, filtros por categoría; agnóstico de Pinecone)
├─ rag_system.py        RAGSystem: fachada lista para usar (Settings + Pinecone + OpenAI)
├─ vectorstore.py       fábrica del PineconeVectorStore
├─ evaluation/          metrics.py (Recall, Precision, MRR, nDCG) · stats.py (bootstrap) · runner.py · report.py · ablation.py
└─ cli.py               python -m src.cli {setup,ingest,query,evaluate}
evaluate.py             punto de entrada de la consigna (= `python -m src.cli evaluate`)
tests/                  pytest: unitarios sin red + 1 de integración
```

## Formatos de entrada

`ingest` recorre `data/raw` (o `--raw-dir`) y elige el loader por extensión. Un mismo corpus puede mezclar formatos.

| Formato | Unidad | Metadata propia |
|---|---|---|
| `.md` | un documento por archivo | `section` (encabezado del chunk), `category` = carpeta |
| `.pdf` | un documento por archivo; los chunks pueden cruzar páginas | `page` = página donde **empieza** el chunk (1-based), `total_pages`, `title` si el PDF la trae |
| `.json` / `.jsonl` | un documento por registro | `category`, `tags` y cualquier campo escalar o lista de str del registro |

```json
[{"id": "pwd", "text": "Para restablecer tu clave...", "category": "cuentas", "tags": ["login", "clave"]}]
```

- `doc_id` es la ruta sin extensión (`runbooks/operaciones`); en JSON, `<archivo>/<id>`. Si dos archivos producen el mismo `doc_id` (p. ej. `guia.md` y `guia.pdf`) la ingesta falla con un error explícito.
- Los chunks **cruzan páginas** (el salto de página es un salto de línea más) y `page` es la página donde empieza el chunk: uno que mezcla el final de la 3 con el comienzo de la 4 lleva `page: 3`. Las páginas sin texto (PDF escaneados, sin OCR) se omiten y quedan en `PdfLoader.skipped_pages`; un PDF cifrado con contraseña falla.
- Los registros JSON inválidos (sin texto, ids repetidos, estructura que no es lista de objetos) fallan indicando `archivo[posición]`. Los campos anidados no se indexan (Pinecone solo admite escalares y listas de texto), y `source`, `doc_id`, `chunk_id`, `chunk_index` y `text` no se pueden sobrescribir.
- Verificado contra Pinecone real con un PDF de 3 páginas y un JSON con etiquetas: la consulta *"how do I fail over the database"* devuelve la página 2 del PDF en primer lugar, y `--category cuentas` filtra por el campo del JSON. Los tests de los loaders generan los PDF al vuelo, sin archivos binarios en el repo.
## Replicar el índice (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env        # completar PINECONE_API_KEY y OPENAI_API_KEY

# Dataset: docs en Markdown de FastAPI (tutorial + advanced + 2 sueltos)
git clone --depth 1 https://github.com/fastapi/fastapi $env:TEMP\fastapi_src
$s = "$env:TEMP\fastapi_src\docs\en\docs"
foreach ($d in 'tutorial','advanced') { New-Item -ItemType Directory -Force data\raw\$d | Out-Null; Copy-Item "$s\$d\*.md" data\raw\$d }
Copy-Item "$s\async.md","$s\features.md" data\raw

# Opcional: código referenciado por la documentación (docs_src), para `ingest --with-code`
New-Item -ItemType Directory -Force data\code | Out-Null
Copy-Item "$env:TEMP\fastapi_src\docs_src\*" data\code -Recurse

python -m src.cli setup            # crea el índice (idempotente)
python -m src.cli ingest           # 75 documentos -> 224 chunks
python -m src.cli query "lifespan events" --category advanced   # prueba rápida
python evaluate.py                 # reporte de métricas (= python -m src.cli evaluate)
```

`.env` nunca se sube al repo (está en `.gitignore`); la plantilla es `.env.example`.

## Evaluación

**Golden set** (`eval/golden_set.json`): 48 preguntas con documento fuente conocido, en tres tipos y repartidas en dos particiones estratificadas:

| Tipo | n | Qué mide |
|---|---|---|
| `exacto` | 22 | la pregunta nombra un identificador técnico (`WebSocketDisconnect`, `ORJSONResponse`...) |
| `parafrasis` | 16 | misma intención, sin las palabras clave del documento |
| `ambigua` | 10 | conceptual y cercana a otros documentos (p. ej. extras en query, cookies y headers) |

La partición `dev` (24) se usa **solo** para elegir los pesos del híbrido; el resultado se informa en `test` (24), que no participó en ningún ajuste. Un chunk es relevante si su `doc_id` es el esperado.

**Métricas:** Recall@5, MRR, nDCG@5, Precision@5 y Precision@5 normalizada (dividida por el máximo alcanzable: un documento de un solo chunk no puede pasar de 0.2). Cada valor lleva un intervalo de confianza del 95 % por bootstrap sobre las preguntas, y los métodos se comparan con bootstrap pareado.

```powershell
python evaluate.py                         # imprime el reporte y lo guarda en reports/latest.{md,json}
python -m src.cli evaluate --bootstrap 5000 --seed 1 --no-report
python -m src.cli evaluate --namespace otro-ns --report-dir reports/otro   # otro namespace del mismo índice
```

### Resultados en `test` (n=24, IC 95 %)

| Método | Recall@5 | MRR | nDCG@5 | Precision@5 | Precision@5 norm. |
|---|---|---|---|---|---|
| Solo BM25 | 0.83 [0.67, 0.96] | 0.69 [0.52, 0.83] | 0.52 [0.39, 0.64] | 0.30 | 0.55 |
| Solo vectorial | **1.00** [1.00, 1.00] | **0.88** [0.77, 0.97] | **0.73** [0.64, 0.83] | **0.43** | **0.76** |
| Híbrido 0.5/0.5 | **1.00** [1.00, 1.00] | 0.83 [0.70, 0.95] | 0.68 [0.56, 0.79] | 0.39 | 0.72 |

En todo el set (n=48): Recall@5 0.90 (BM25), 1.00 (vectorial) y 1.00 (híbrido); MRR 0.77, 0.93 y 0.90. Reporte completo, con barrido de pesos, desglose por tipo y comparaciones, en [reports/latest.md](reports/latest.md).

### Qué se puede y qué no se puede afirmar

- **El híbrido supera claramente a BM25 solo** (en las 48 preguntas: Recall +0.10, IC [+0.02, +0.19], p=0.008; MRR +0.13, IC [+0.06, +0.21], p<0.001).
- **El híbrido no supera al vectorial solo.** Ambos aciertan en las 48 preguntas (Recall 1.00) y el MRR del híbrido es incluso algo menor (−0.023, IC [−0.085, +0.041], p=0.76): indistinguibles. Con este corpus, BM25 no aporta una mejora medible sobre la búsqueda vectorial; el híbrido solo gana en las preguntas parafraseadas (MRR 0.92 frente a 0.88) y pierde en las ambiguas (0.87 frente a 0.95).
- **El barrido de pesos en `dev` es plano hasta 0.5** (MRR 0.979 con pesos de BM25 de 0.1 a 0.5) y baja a 0.90-0.96 con 0.6-0.9. Se eligió 0.5/0.5 por tener el mejor nDCG entre los empatados.
- **BM25 no destaca en las preguntas `exacto`** (MRR 0.86 frente a 0.95 del vectorial). Indexar el código de los ejemplos no lo explica (ver abajo).
- **El filtro por categoría**, con la categoría conocida de antemano (oráculo, cota superior), sube el MRR del híbrido de 0.90 a 0.93. Solo es útil si la aplicación conoce la categoría. No se hizo prueba pareada.
- Con n=24 en `test`, un intervalo de ±0.1 es esperable: las diferencias menores no son concluyentes.

### Experimento: tamaño de los chunks

El splitter es voraz y deja chunks diminutos cuando la sección vecina ya está casi llena (590 + 33 > 600): con `chunk_size=600`, 51 de 271 chunks (19 %) tenían menos de 200 tokens y el mínimo era 6. Ahora los chunks de menos de `min_chunk_tokens=200` se fusionan con su vecino, sin pasar de `max_chunk_tokens=800`. La fusión recompone el tramo del texto original, así que el solapamiento no se duplica, y `page` y `section` siguen siendo los del inicio del chunk.

| | Antes | Ahora |
|---|---|---|
| Chunks | 271 | 224 |
| Menos de 200 tokens | 51 | 4 |
| Mediana / p75 / máximo (tokens) | 406 / 518 / 596 | 453 / 552 / 770 |
| Entre 500 y 800 tokens | 80 (30 %) | 85 (38 %) |

Los dos chunks que quedan bajo 150 tokens son documentos enteros de un solo chunk corto. Impacto medido con las mismas 48 preguntas y bootstrap pareado ([reports/ablation-chunks.md](reports/ablation-chunks.md); versión anterior en [reports/baseline/](reports/baseline/latest.md)): el híbrido no cambia (Recall 1.00 → 1.00, MRR +0.009, p=0.39), el vectorial solo mejora algo (Recall 0.98 → 1.00, MRR +0.030, p=0.04, una de ~30 comparaciones, así que sin mucho peso) y BM25 solo empeora levemente (Recall −0.02). No hay regresión del método por defecto. Solo ~38 % de los chunks cae en el rango 500-800 de la consigna: la mediana es menor porque muchas secciones de la documentación son cortas, y fusionarlas más (mínimo 300) subiría los chunks por encima de lo que conviene a la precisión.

Al implementarlo apareció un defecto de LangChain: `add_start_index` devuelve -1 para el 9 % de los chunks (24 de 271, en 18 de 75 documentos), porque busca desde «inicio anterior + largo anterior − chunk_overlap» tratando el solapamiento (en tokens) como caracteres. Por eso las posiciones se calculan en `locate_chunks`. Esto también corrige un efecto latente sobre `page` en los PDF.
### Experimento: indexar el código de los ejemplos

La documentación referencia ~350 fragmentos de código (`{* ../../docs_src/... *}`). Por defecto se descartan; con `ingest --with-code` se incrustan como bloques de código (respetando `ln[a:b]` y sin repetir el mismo fragmento en un documento): 271 → 363 chunks. Se midió en un namespace aparte del mismo índice (`docs-fastapi-code`), con las mismas 48 preguntas y semillas, y bootstrap pareado ([reports/ablation-code.md](reports/ablation-code.md)):

| Método (n=48) | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 | 0.92 → 0.96 (p=0.14) | 0.79 → 0.80 (p=0.32) | 0.65 → 0.63 |
| Vectorial | 0.98 → 0.96 | 0.90 → 0.92 (p=0.19) | 0.80 → 0.79 |
| Híbrido 0.5/0.5 | 1.00 → 0.98 | 0.89 → 0.90 (p=0.34) | 0.77 → 0.75 |

**Conclusión: resultado nulo.** Ninguna diferencia es creíble (todos los IC incluyen o tocan el 0) y el híbrido pierde una pregunta (`cookie-params`: la palabra `Cookie` aparece ahora en el código de otros documentos y compite con el correcto). Precision@5 sube (0.44 → 0.50) solo porque hay más chunks por documento: la normalizada baja (0.79 → 0.74). Por eso el código **no se indexa por defecto**; queda como opción documentada (medido con la versión anterior del chunking).

## Filtro por categoría

```python
rag = RAGSystem()
rag.categories                                   # ['advanced', 'tutorial', 'general']
rag.retrieve("lifespan events", category="advanced")```n
El filtro se aplica a ambos recuperadores: BM25 se construye sobre el subcorpus de la categoría y Pinecone recibe `filter={'category': {'$eq': ...}}`. Una categoría inexistente lanza `ValueError`.

## Tests

```powershell
pytest                    # 121 tests unitarios, sin red (corpus y vectorstore falsos)
pytest -m integration   # prueba contra Pinecone/OpenAI reales (requiere .env)
```

## Calidad

```powershell
ruff check .             # lint (configurado en pyproject.toml)
```n