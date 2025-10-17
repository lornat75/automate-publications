#!/usr/bin/env python3
"""
Parse an EndNote XML export and output a normalized list of publication dicts.

Note: EndNote's native .enl library is a proprietary container; please export to
EndNote XML (File -> Export -> XML) and pass that .xml here.

Schema aligns with parse_publications_html.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _normalize_title_for_id(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s


def _author_to_surname_initials(a: str) -> str:
    a = _norm_space(a)
    if "," in a:
        last, rest = a.split(",", 1)
        rest = _norm_space(rest)
    else:
        parts = a.split()
        last, rest = (parts[-1], " ".join(parts[:-1])) if parts else (a, "")
    initials = " ".join([p[0].upper() + "." for p in rest.split() if p])
    res = f"{_norm_space(last)}, {_norm_space(initials)}".strip()
    return res.rstrip(",")


def _elem_text_recursive(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    # Gather all text including child nodes (e.g., <style> wrappers)
    txt = "".join(elem.itertext())
    return _norm_space(txt)


def _get_first_text(node: ET.Element, paths: List[str]) -> Optional[str]:
    for p in paths:
        found = node.find(p)
        t = _elem_text_recursive(found)
        if t:
            return t
    return None


def _get_all_texts(node: ET.Element, path: str) -> List[str]:
    vals: List[str] = []
    for x in node.findall(path):
        t = _elem_text_recursive(x)
        if t:
            vals.append(t)
    return vals


def _extract_doi(texts: List[str]) -> Optional[str]:
    for t in texts:
        m = DOI_RE.search(t)
        if m:
            return m.group(1)
    return None


@dataclass
class Publication:
    id: str
    year: Optional[int]
    authors: List[str]
    title: str
    venue: str
    doi: Optional[str]
    preprint_url: Optional[str]
    fulltext_url: Optional[str]
    bibtex_url: Optional[str]
    links: Dict[str, str] = field(default_factory=dict)
    source: str = "endnote-xml"


def parse_endnote_xml(path: str) -> List[Dict]:
    tree = ET.parse(path)
    root = tree.getroot()

    # EndNote exports typically have <records><record>...</record></records>
    records = root.findall(".//record") if root.tag != "record" else [root]
    pubs: List[Publication] = []

    # Track DOI duplicates to avoid collisions that break matching
    doi_title_fp: Dict[str, str] = {}
    doi_counts: Dict[str, int] = {}

    # Track collisions for reporting (doi -> list of (id, title))
    collisions: Dict[str, List[tuple]] = {}

    for rec in records:
        title = _get_first_text(
            rec,
            [
                "./titles/title",
                "./titles/Title",  # some exports capitalize
            ],
        ) or ""

        # Venue: journal/book title; try secondary-title and periodical/full-title
        venue = _get_first_text(
            rec,
            [
                "./titles/secondary-title",
                "./titles/SecondaryTitle",
                "./periodical/full-title",
                "./periodical/FullTitle",
            ],
        ) or ""

        # Year
        year_s = _get_first_text(rec, ["./dates/year", "./year"]) or ""
        m = re.search(r"(19|20|21)\d{2}", year_s)
        year = int(m.group(0)) if m else None

        # Authors
        authors_raw = _get_all_texts(rec, "./contributors/authors/author")
        authors = [_author_to_surname_initials(a) for a in authors_raw]

        # URLs
        related_urls = _get_all_texts(rec, "./urls/related-urls/url")
        pdf_urls = _get_all_texts(rec, "./urls/pdf-urls/url")
        all_urls = related_urls + pdf_urls

        # DOI extraction: explicit fields or from URLs/texts
        doi_candidates: List[str] = []
        # explicit field sometimes named 'electronic-resource-num' or 'doi'
        for tag in ["./electronic-resource-num", "./doi", "./DOI"]:
            t = _get_first_text(rec, [tag])
            if t:
                doi_candidates.append(t)
        # consider venue/title/url text as well
        doi = _extract_doi([title, venue] + all_urls + doi_candidates)

        # best effort fulltext
        fulltext_url = None
        if pdf_urls:
            fulltext_url = pdf_urls[0]
        elif related_urls:
            fulltext_url = related_urls[0]

        links: Dict[str, str] = {}
        if fulltext_url:
            links["url"] = fulltext_url
        if pdf_urls:
            links["pdf"] = pdf_urls[0]
        if doi:
            links["doi"] = f"https://doi.org/{doi}"

        if doi:
            base_id = f"doi:{doi.lower()}"
            fp = _normalize_title_for_id(title)
            if base_id in doi_title_fp:
                # If same fingerprint/title, treat as exact duplicate: skip adding a second identical record
                if doi_title_fp[base_id] == fp:
                    # Skip silent duplicate
                    continue
                # Different title sharing identical DOI: assign a suffixed id
                doi_counts[base_id] = doi_counts.get(base_id, 1) + 1
                pid = f"{base_id}#{doi_counts[base_id]}"
                # record collision (store both existing and new if first time we notice)
                if base_id not in collisions:
                    # need to find previously stored publication for base_id to include its title
                    prev_title_fp = doi_title_fp[base_id]
                    # attempt to locate earlier publication object (linear search acceptable for small lists)
                    for existing in pubs:
                        if existing.doi and existing.doi.lower() == doi.lower() and _normalize_title_for_id(existing.title) == prev_title_fp:
                            collisions.setdefault(base_id, []).append((existing.id, existing.title))
                            break
                collisions.setdefault(base_id, []).append((pid, title))
            else:
                doi_title_fp[base_id] = fp
                doi_counts[base_id] = 1
                pid = base_id
        else:
            pid = f"title:{_normalize_title_for_id(title)}:{year or ''}"

        pubs.append(
            Publication(
                id=pid,
                year=year,
                authors=authors,
                title=title,
                venue=venue,
                doi=doi,
                preprint_url=None,
                fulltext_url=fulltext_url,
                bibtex_url=None,
                links=links,
                source="endnote-xml",
            )
        )

    # Emit warnings for DOI collisions (distinct titles sharing a DOI)
    if collisions:
        print("[warn] DOI collisions detected (same DOI used by multiple distinct titles):")
        for base_id, entries in collisions.items():
            doi_value = base_id[len("doi:") :]
            print(f"  DOI {doi_value} -> {len(entries)} titles:")
            for pid, t in entries:
                print(f"    - {t} (id={pid})")
        print("[warn] Consider cleaning the source EndNote library to assign unique DOIs per distinct publication.")

    return [asdict(p) for p in pubs]


def main():
    ap = argparse.ArgumentParser(description="Parse an EndNote XML export into JSON")
    ap.add_argument("xml_path", help="Path to EndNote XML (.xml)")
    ap.add_argument("--json", dest="json_out", help="Write JSON to this path")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print a compact list")
    ap.add_argument(
        "--filter-title",
        dest="filter_title",
        help="Case-insensitive substring to filter titles for debug (prints matching full JSON records)",
    )
    args = ap.parse_args()

    pubs = parse_endnote_xml(args.xml_path)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(pubs, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(pubs)} items to {args.json_out}")
    if args.do_print:
        for p in pubs[:10]:
            doi = p.get("doi")
            print(f"{p.get('year')}: {p.get('title')} [{('DOI ' + doi) if doi else 'no doi'}]")
        if len(pubs) > 10:
            print(f"... and {len(pubs) - 10} more")
    if args.filter_title:
        ft = args.filter_title.lower()
        hits = [p for p in pubs if ft in (p.get("title") or "").lower()]
        print(f"[debug] {len(hits)} publication(s) with '{args.filter_title}' in title:")
        for p in hits:
            print(json.dumps(p, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
