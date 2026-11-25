# Sistema RAG escalable con Pinecone

Módulo de recuperación híbrido (BM25 + vectorial) sobre Pinecone Serverless, con la documentación Markdown de FastAPI como corpus.

## Arquitectura

- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dim, cosine).
- **Índice:** Pinecone Serverless (AWS `us-east-1`), namespace `docs-fastapi`.
- **Chunking:** `RecursiveCharacterTextSplitter` por tokens (600 tokens, overlap 80).
- **Metadata por chunk:** `text` (contenido original), `source`, `page` (índice del chunk en el documento), `category`, `doc_id`, `chunk_id`.
- **Híbrido:** `EnsembleRetriever` (BM25 0.5 + Pinecone 0.5). El corpus BM25 se reconstruye desde la metadata de Pinecone (cache en `data/cache/`), sin otra base de datos.

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

Golden set de 5 preguntas con documento fuente conocido (`eval/golden_set.json`). Un chunk es relevante si su `doc_id` es el esperado.

| Método | Recall@5 | Precision@5 |
|---|---|---|
| Solo vectorial | 1.00 | 0.52 |
| Híbrido (BM25 + vector) | 1.00 | 0.48 |

**Lectura:** ambos recuperan el documento correcto en las 5 preguntas. El híbrido no mejora en este set: las preguntas ya contienen términos que el modelo semántico resuelve bien y el set es muy pequeño (5 preguntas), por lo que la diferencia de 0.04 no es significativa. Precision@5 tiene un techo bajo en documentos de pocos chunks (p. ej. `static-files` tiene muy pocos, así que no puede llenar 5 posiciones). Para ver el aporte de BM25 conviene ampliar el golden set con identificadores exactos y ajustar los pesos en `RAGSystem(weights=...)`.
