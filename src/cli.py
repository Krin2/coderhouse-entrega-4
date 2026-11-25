"""CLI: `python -m src.cli {setup,ingest,query,evaluate}`."""
from __future__ import annotations

import argparse
import sys

from src.settings import Settings


def _setup(settings: Settings, _args) -> None:
    from src.infra.index import ensure_index
    settings.require_keys()
    ensure_index(settings)


def _ingest(settings: Settings, _args) -> None:
    from src.ingestion.pipeline import IngestPipeline
    IngestPipeline.from_settings(settings).run()


def _query(settings: Settings, args) -> None:
    from src.rag_system import RAGSystem
    for d in RAGSystem(settings, k=args.k).retrieve(args.text, category=args.category):
        print(f"{d.metadata['chunk_id']:<48} | {d.page_content[:60]!r}")


def _evaluate(settings: Settings, args) -> None:
    from src.evaluation.report import to_markdown, write_reports
    from src.evaluation.runner import load_golden, run_comparison
    from src.rag_system import RAGSystem
    golden = load_golden(settings.golden_set)
    rag = RAGSystem(settings, k=args.k, cache_queries=True)
    results = run_comparison(rag, golden, args.k, n_bootstrap=args.bootstrap, seed=args.seed)
    print(to_markdown(results))
    if not args.no_report:
        md, js = write_reports(results, settings.report_dir)
        print(f"Reporte guardado en {md} y {js}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="crea el índice Serverless si no existe").set_defaults(run=_setup)
    sub.add_parser("ingest", help="documentos -> chunks -> embeddings -> Pinecone").set_defaults(run=_ingest)
    q = sub.add_parser("query", help="consulta el recuperador híbrido")
    q.add_argument("text")
    q.add_argument("--category", default=None, help="restringe BM25 y Pinecone a una categoría")
    q.add_argument("-k", type=int, default=Settings.top_k)
    q.set_defaults(run=_query)
    e = sub.add_parser("evaluate", help="Recall@k y Precision@k sobre el golden set")
    e.add_argument("-k", type=int, default=Settings.top_k)
    e.add_argument("--bootstrap", type=int, default=2000, help="remuestreos para los intervalos de confianza")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--no-report", action="store_true", help="no escribir reports/latest.{md,json}")
    e.set_defaults(run=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.run(Settings.from_env(), args)


if __name__ == "__main__":
    main()
