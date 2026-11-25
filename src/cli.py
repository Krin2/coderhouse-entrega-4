"""CLI: `python -m src.cli {setup,ingest,query,evaluate,compare-reports}`."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from src.retrieval.rerank import DEFAULT_MODEL as DEFAULT_RERANK_MODEL
from src.settings import Settings


def _setup(settings: Settings, _args) -> None:
    from src.infra.index import ensure_index
    settings.require_keys()
    ensure_index(settings)


def _ingest(settings: Settings, args) -> None:
    from src.ingestion.pipeline import IngestPipeline
    IngestPipeline.from_settings(settings, with_code=args.with_code).run()


def _make_reranker(settings: Settings, args):
    """PineconeReranker con caché en disco, o None si no se pidió `--rerank`."""
    if not args.rerank:
        return None
    from src.infra.index import get_client
    from src.retrieval.rerank import PineconeReranker
    settings.require_keys()
    return PineconeReranker(get_client(settings), args.rerank_model, max_requests=args.max_rerank_requests,
                            cache_path=settings.cache_dir / f"rerank-{args.rerank_model}.jsonl")


def _query(settings: Settings, args) -> None:
    from src.rag_system import RAGSystem
    rag = RAGSystem(settings, k=args.k, reranker=_make_reranker(settings, args))
    for d in rag.retrieve(args.text, category=args.category, rerank=args.rerank):
        page = f" p.{int(d.metadata['page'])}" if "page" in d.metadata else ""
        print(f"{d.metadata['chunk_id']:<48}{page:<7}| {d.page_content[:60]!r}")


def _evaluate(settings: Settings, args) -> None:
    from src.evaluation.report import to_markdown, write_reports
    from src.evaluation.runner import load_golden, run_comparison
    from src.rag_system import RAGSystem
    golden = load_golden(settings.golden_set)
    if args.rerank and args.report_dir is None:                  # no pisar el reporte sin reranking
        settings = replace(settings, report_dir=Path("reports/rerank"))
    rag = RAGSystem(settings, k=args.k, cache_queries=True, reranker=_make_reranker(settings, args))
    results = run_comparison(rag, golden, args.k, n_bootstrap=args.bootstrap, seed=args.seed)
    print(to_markdown(results))
    if not args.no_report:
        md, js = write_reports(results, settings.report_dir)
        print(f"Reporte guardado en {md} y {js}")


def _compare_reports(_settings: Settings, args) -> None:
    from src.evaluation.ablation import compare_runs, load_results, to_markdown
    rows = compare_runs(load_results(args.base), load_results(args.other),
                        n_bootstrap=args.bootstrap, seed=args.seed)
    text = to_markdown(rows, args.base_label, args.other_label)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Guardado en {args.out}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--namespace", default=None, help="namespace de Pinecone (por defecto, el de .env)")

    rerank = argparse.ArgumentParser(add_help=False)
    rerank.add_argument("--rerank", action="store_true",
                        help="reordena los candidatos con el reranker de Pinecone (consume cuota de la API)")
    rerank.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    rerank.add_argument("--max-rerank-requests", type=int, default=150,
                        help="tope de solicitudes reales al reranker; la caché local no cuenta")

    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", parents=[common], help="crea el índice Serverless si no existe").set_defaults(run=_setup)

    i = sub.add_parser("ingest", parents=[common], help="documentos -> chunks -> embeddings -> Pinecone")
    i.add_argument("--with-code", action="store_true",
                   help="incrusta el código referenciado por la documentación (data/code)")
    i.add_argument("--raw-dir", default=None, help="carpeta con .md/.pdf/.json/.jsonl (por defecto, data/raw)")
    i.set_defaults(run=_ingest)

    q = sub.add_parser("query", parents=[common, rerank], help="consulta el recuperador híbrido")
    q.add_argument("text")
    q.add_argument("--category", default=None, help="restringe BM25 y Pinecone a una categoría")
    q.add_argument("-k", type=int, default=Settings.top_k)
    q.set_defaults(run=_query)

    e = sub.add_parser("evaluate", parents=[common, rerank], help="métricas, IC y comparaciones sobre el golden set")
    e.add_argument("-k", type=int, default=Settings.top_k)
    e.add_argument("--bootstrap", type=int, default=2000, help="remuestreos para los intervalos de confianza")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--report-dir", default=None, help="carpeta del reporte (por defecto, reports/)")
    e.add_argument("--no-report", action="store_true", help="no escribir latest.{md,json}")
    e.set_defaults(run=_evaluate)

    c = sub.add_parser("compare-reports", help="ablación pareada entre dos reportes JSON")
    c.add_argument("base")
    c.add_argument("other")
    c.add_argument("--base-label", default="base")
    c.add_argument("--other-label", default="otro")
    c.add_argument("--bootstrap", type=int, default=2000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default=None, help="archivo .md de salida")
    c.set_defaults(run=_compare_reports)
    return parser


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    overrides = {}
    if getattr(args, "namespace", None):
        overrides["namespace"] = args.namespace
    if getattr(args, "raw_dir", None):
        overrides["raw_dir"] = Path(args.raw_dir)
    if getattr(args, "report_dir", None):
        overrides["report_dir"] = Path(args.report_dir)
    args.run(replace(settings, **overrides) if overrides else settings, args)


if __name__ == "__main__":
    main()
