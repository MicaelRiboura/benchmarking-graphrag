#!/usr/bin/env python3
"""Gera PDFs PHMSA preenchidos a partir de um arquivo TSV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


RectRef = Tuple[int, fitz.Rect]


def int_to_rgb(color: int) -> Tuple[int, int, int]:
    return (color >> 16) & 255, (color >> 8) & 255, color & 255


def is_red_color(color: int) -> bool:
    r, g, b = int_to_rgb(color)
    return r >= 150 and g <= 120 and b <= 120


def rect_intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def union_rects(rects: List[fitz.Rect]) -> fitz.Rect:
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] or "registro"


def fit_text_to_rect(rect: fitz.Rect, text: str, fontsize: float) -> str:
    if not text:
        return ""
    return text


def write_value_in_rect(page: fitz.Page, rect: fitz.Rect, value: str) -> bool:
    """Escreve o valor no retângulo, com fallback para caixas pequenas."""
    text = fit_text_to_rect(rect, value, fontsize=7)
    if not text:
        return False

    # Tenta primeiro com textbox reduzindo fonte para caber.
    for fontsize in (7, 6, 5, 4, 3, 2):
        rc = page.insert_textbox(
            rect,
            text,
            fontsize=fontsize,
            fontname="helv",
            color=(0, 0, 1),
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )
        if rc >= 0:
            return True

    # Fallback: escreve na linha de base para não perder o conteúdo.
    baseline_y = max(rect.y0 + 4, rect.y1 - 1)
    page.insert_text(
        fitz.Point(rect.x0, baseline_y),
        text,
        fontsize=2,
        fontname="helv",
        color=(0, 0, 1),
        overlay=True,
    )
    return True


def map_placeholders(template_path: Path, columns: List[str]) -> Dict[str, List[RectRef]]:
    placeholders: Dict[str, List[RectRef]] = defaultdict(list)
    doc = fitz.open(template_path)
    try:
        for page_idx, page in enumerate(doc):
            raw = page.get_text("rawdict")
            char_boxes: List[Tuple[fitz.Rect, int]] = []
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_color = int(span.get("color", 0))
                        for ch in span.get("chars", []):
                            bbox = ch.get("bbox")
                            if bbox:
                                char_boxes.append((fitz.Rect(bbox), span_color))

            for column in columns:
                found = page.search_for(column)
                for rect in found:
                    red_area = 0.0
                    total_area = 0.0
                    red_char_rects: List[fitz.Rect] = []
                    for ch_rect, ch_color in char_boxes:
                        overlap = rect_intersection_area(rect, ch_rect)
                        if overlap <= 0:
                            continue
                        total_area += overlap
                        if is_red_color(ch_color):
                            red_area += overlap
                            red_char_rects.append(ch_rect)

                    if total_area <= 0:
                        continue
                    red_ratio = red_area / total_area
                    # Substitui somente placeholders predominante vermelhos.
                    if red_ratio >= 0.55 and red_char_rects:
                        tight_rect = union_rects(red_char_rects)
                        # Encolhe levemente o retângulo para evitar apagar texto vizinho.
                        tight_rect = fitz.Rect(
                            tight_rect.x0 + 0.5,
                            tight_rect.y0 + 0.3,
                            tight_rect.x1 - 0.5,
                            tight_rect.y1 - 0.3,
                        )
                        placeholders[column].append((page_idx, tight_rect))
    finally:
        doc.close()
    return placeholders


def generate_pdfs(
    data_path: Path,
    template_path: Path,
    output_dir: Path,
    max_records: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encodings = ("utf-8-sig", "cp1252", "latin-1")
    last_error: Exception | None = None
    selected_encoding: str | None = None
    for enc in encodings:
        try:
            with data_path.open("r", encoding=enc, newline="") as probe:
                probe.readline()
            selected_encoding = enc
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if selected_encoding is None:
        raise RuntimeError(f"Não foi possível ler o arquivo de dados: {last_error}")

    with data_path.open("r", encoding=selected_encoding, newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Arquivo TSV sem cabeçalho.")
        columns = [col.strip() for col in reader.fieldnames]
        placeholders = map_placeholders(template_path, columns)

        missing_cols = [col for col in columns if col not in placeholders]
        if missing_cols:
            print(
                f"Aviso: {len(missing_cols)} colunas não foram encontradas no template e serão ignoradas."
            )

        for index, row in enumerate(reader, start=1):
            if max_records is not None and index > max_records:
                break

            doc = fitz.open(template_path)
            try:
                replacements_by_page: Dict[int, List[Tuple[fitz.Rect, str]]] = defaultdict(list)
                for col, raw_value in row.items():
                    if col not in placeholders:
                        continue
                    value = (raw_value or "").strip()
                    for page_idx, rect in placeholders[col]:
                        replacements_by_page[page_idx].append((rect, value))

                for page_idx, replacements in replacements_by_page.items():
                    page = doc[page_idx]
                    for rect, _ in replacements:
                        page.add_redact_annot(rect, fill=None)
                    page.apply_redactions()

                    for rect, value in replacements:
                        write_value_in_rect(page, rect, value)

                report_number = (row.get("REPORT_NUMBER") or "").strip()
                suffix = sanitize_filename(report_number) if report_number else f"registro_{index:05d}"
                output_path = output_dir / f"phmsa_{suffix}.pdf"
                doc.save(output_path, garbage=4, deflate=True)
                print(f"[{index}] PDF gerado: {output_path}")
            finally:
                doc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera PDFs preenchidos a partir de um TSV PHMSA e de um PDF template."
    )
    parser.add_argument(
        "--data",
        default="dataset_phmsa/accident_hazardous_liquid_jan2010_present.txt",
        help="Caminho do arquivo TSV de dados.",
    )
    parser.add_argument(
        "--template",
        default="dataset_phmsa/Hazardous Liquid Accident PHMSA F7000 1 Rev 3-2021 Data fields.pdf",
        help="Caminho do template PDF com placeholders.",
    )
    parser.add_argument(
        "--output-dir",
        default="pdfs",
        help="Diretório de saída dos PDFs gerados.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limita a quantidade de registros processados (útil para teste).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_pdfs(
        data_path=Path(args.data),
        template_path=Path(args.template),
        output_dir=Path(args.output_dir),
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
