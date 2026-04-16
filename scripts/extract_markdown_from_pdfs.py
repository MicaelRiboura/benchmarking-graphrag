#!/usr/bin/env python3
"""Extrai texto em Markdown de PDFs usando pymupdf4llm."""

from __future__ import annotations

import argparse
from pathlib import Path

from pymupdf4llm import to_markdown


def extract_markdown_from_pdfs(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"Nenhum PDF encontrado em: {input_dir}")
        return

    for pdf_path in pdf_files:
        markdown_text = to_markdown(str(pdf_path))
        output_path = output_dir / f"{pdf_path.stem}.txt"
        output_path.write_text(markdown_text, encoding="utf-8")
        print(f"Gerado: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai Markdown de PDFs e salva em arquivos .txt."
    )
    parser.add_argument(
        "--input-dir",
        default="pdfs",
        help="Diretório com os arquivos PDF de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        default="pdfs_txt",
        help="Diretório de saída dos arquivos .txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_markdown_from_pdfs(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
