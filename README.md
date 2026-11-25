# Sistema RAG escalable con Pinecone

Módulo de recuperación híbrido (BM25 + vectorial) sobre Pinecone Serverless, con la documentación Markdown de FastAPI como corpus.

## Arquitectura

- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dim, cosine).
- **Índice:** Pinecone Serverless (AWS `us-east-1`), namespace `docs-fastapi`.
- **Chunking:** `RecursiveCharacterTextSplitter` por tokens (600 tokens, overlap 80).
- **Metadata por chunk:** `text` (contenido original), `source`, `category`, `doc_id`, `chunk_id`, `chunk_index` (posición en el documento) y `section` (encabezado Markdown del chunk).
- **Híbrido:** `EnsembleRetriever` (BM25 0.5 + Pinecone 0.5). El corpus BM25 se reconstruye desde la metadata de Pinecone (caché en `data/cache/`, que se invalida sola: `ingest` la borra y, además, se descarta si su tamaño no coincide con el namespace de Pinecone), sin otra base de datos.

```
src/config.py          configuración y validaciones
src/pinecone_setup.py  crea el índice si no existe y valida la dimensión
src/ingest.py          carga -> chunking -> embeddings -> upsert
src/vectorstore.py     PineconeVectorStore / retriever semántico
src/rag_system.py      clase RAGSystem (EnsembleRetriever)
evaluate.py            Recall@5 y Precision@5 sobre eval/golden_set.json
```

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

python -m src.pinecone_setup       # crea el índice (idempotente)
python -m src.ingest               # 75 documentos -> 271 chunks
python -m src.rag_system           # prueba rápida del recuperador
python evaluate.py                 # reporte de métricas
```

`.env` nunca se sube al repo (está en `.gitignore`); la plantilla es `.env.example`.

## Evaluación

Golden set de 19 preguntas con documento fuente conocido (`eval/golden_set.json`): 15 con términos técnicos exactos (`WebSocketDisconnect`, `ORJSONResponse`, `convert_underscores`...) y 4 parafraseadas sin la palabra clave. Un chunk es relevante si su `doc_id` es el esperado. `evaluate.py` compara BM25 solo, vectorial solo y el híbrido con tres combinaciones de pesos.

| Método | Recall@5 | Precision@5 |
|---|---|---|
| Solo BM25 | 0.95 | 0.41 |
| Solo vectorial | 0.95 | 0.46 |
| Híbrido bm25=0.3 / vec=0.7 | **1.00** | **0.46** |
| Híbrido bm25=0.5 / vec=0.5 (por defecto) | **1.00** | 0.45 |
| Híbrido bm25=0.7 / vec=0.3 | 0.95 | 0.42 |

**Lectura:** el híbrido es el único que recupera el documento correcto en las 19 preguntas (cada método por separado falla una distinta), lo que respalda combinar ambas señales. En Precision@5 la mejora es marginal (0.45-0.46 vs 0.46 vectorial): el techo es bajo porque muchos documentos tienen pocos chunks y no pueden llenar 5 posiciones. Con n=19 la diferencia de un solo acierto en Recall no es estadísticamente concluyente; conviene ampliar el set antes de fijar los pesos. Los pesos se cambian con `RAGSystem(weights=(bm25, vector))`.
