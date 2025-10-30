#!/usr/bin/env python3
"""Generate an HTML publications page directly from an
EndNote XML export. Groups entries by year (descending) and formats according to
basic rules inferred from the existing hand‑maintained page. Optionally performs
online lookups (arXiv, OpenAIRE) while parsing to attach open access links.

Formatting rules (best effort, based solely on XML contents):
  - Authors: "Surname, I." style, separated by comma+space; before last author add ", and ".
  - Title: italic (<i>title</i>)
  - Conference papers: "..., <i>Title</i>, in VENUE, LOCATION, YEAR[, pp. X-Y]."
  - Journal articles: "..., <i>Title</i>, JOURNAL, vol. V[, no. N][, pp. X-Y], YEAR." (order matches existing examples)
  - Book sections / chapters: "..., <i>Chapter Title</i>, BOOK TITLE, EDITORS, eds.: PUBLISHER[, pp. X-Y], YEAR."
    (If only one editor use "ed." instead of "eds.")
  - Other / fallback types: treat like generic: authors, italic title, venue (if any), year.
  - Pages: we show them as "pp. PAGES" unless already prefixed or clearly non-numeric.
  - We intentionally strip detailed date ranges (day/month) – only year is used.

Limitations:
  - Relies on EndNote's ref-type name values (e.g. "Conference Paper", "Journal Article", "Book Section").
  - If the XML contains inconsistent metadata, output mirrors what is available.
  - DOI / links are not shown to keep parity with many lines in the original (could be added later).

Usage (basic):
    python tools/generate_publications_xml_page.py exported.xml --out xml_publications.html

With online OA lookup (adds [preprint] and / or [fulltext] links):
    python tools/generate_publications_xml_page.py exported.xml --out xml_publications.html \
            --lookup --crossref --openaire --cache oa_cache.json --limit 50

License filtering:
                If --openaire is enabled a discovered link is added as [fulltext] only if OpenAIRE reports a license
                indicating an open Creative Commons style license (e.g. CC-BY, CC-BY-SA, CC0, Public Domain).
            ArXiv entries are always labeled [preprint]. If a DOI itself is of the form
        10.48550/arXiv.<id> an arXiv link is derived directly without network search.
"""

from __future__ import annotations

import argparse
import html
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Tuple, Set
import re
import sys
from pathlib import Path
import json
import urllib.parse
import urllib.request
from urllib.error import URLError, HTTPError
import time

# Reuse author normalization and DOI extraction from existing parser when available
try:
    from parse_endnote_xml import _author_to_surname_initials, _extract_doi  # type: ignore
except Exception:  # pragma: no cover
    def _author_to_surname_initials(a: str) -> str:  # fallback
        parts = a.replace("\n", " ").strip().split()
        if not parts:
            return a.strip()
        last = parts[-1]
        initials = [p[0].upper() + "." for p in parts[:-1] if p]
        return f"{last}, {' '.join(initials)}".strip()

    # Minimal DOI extractor fallback (similar to parse_endnote_xml._extract_doi)
    import re as _re
    _DOI_RE = _re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", _re.I)

    def _extract_doi(texts: list) -> Optional[str]:
        for t in texts:
            if not t:
                continue
            m = _DOI_RE.search(t)
            if m:
                return m.group(1)
        return None


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def get_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return norm_space("".join(node.itertext()))


def first_text(parent: ET.Element, paths: List[str]) -> str:
    for p in paths:
        f = parent.find(p)
        t = get_text(f)
        if t:
            return t
    return ""


def all_text(parent: ET.Element, path: str) -> List[str]:
    vals = []
    for e in parent.findall(path):
        t = get_text(e)
        if t:
            vals.append(t)
    return vals


def join_authors(authors: List[str]) -> str:
    authors = [a for a in (authors or []) if a and a.strip()]
    if not authors:
        return ""
    normed = [_author_to_surname_initials(a) for a in authors]
    if len(normed) == 1:
        return normed[0]
    return ", ".join(normed[:-1]) + ", and " + normed[-1]


def classify(rec: Dict) -> str:
    rtype = (rec.get("type") or "").lower()
    if "conference" in rtype:
        return "conference"
    if "journal" in rtype:
        return "journal"
    if "book section" in rtype or "book chapter" in rtype:
        return "book_section"
    if rtype == "book" or rtype.startswith("book"):
        return "book"
    return "other"


def is_under_submission(rec: Dict) -> bool:
    """Return True if the record appears to be a non‑published work.

    Criteria (case‑insensitive substring match):
      * 'under submission'
      * 'submitted' (covers 'submitted to')
      * 'under review'
      * 'in review'
    We check canonical fields (title, journal, secondary, publisher, year) AND fall back to
    scanning all string values in the record (EndNote exports sometimes place status notes
    in less common fields).
    """
    phrases = ["under submission", "submitted", "under review", "in review"]
    # Primary fields first (cheap)
    for fld in [rec.get("journal"), rec.get("secondary"), rec.get("publisher"), rec.get("title"), rec.get("year")]:
        if not fld:
            continue
        low = fld.lower()
        if any(p in low for p in phrases):
            return True
    # Fallback: scan all string fields
    for v in rec.values():
        if isinstance(v, str):
            low = v.lower()
            if any(p in low for p in phrases):
                return True
    return False


def format_pages(p: str) -> str:
    if not p:
        return ""
    p = p.strip()
    if p.lower().startswith("pp"):
        return p
    return f"pp. {p}"


def normalize_doi(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    raw = re.sub(r"^https?://(dx\.)?doi\.org/", "", raw, flags=re.I)
    raw = re.sub(r"^doi:\s*", "", raw, flags=re.I)
    return raw.strip()


def parse_records(xml_path: str) -> List[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    recs: List[Dict] = []
    for rnode in root.findall('.//record'):
        rec: Dict[str, Optional[str]] = {}
        # type
        rt = rnode.find('ref-type')
        if rt is not None:
            rec['type'] = (rt.get('name') or '').strip()
        # titles
        rec['title'] = first_text(rnode, [
            './titles/title',
            './titles/primary-title',
            './titles/secondary-title',
        ])
        rec['journal'] = first_text(rnode, [
            './titles/secondary-title',
            './periodical/full-title',
            './periodical/abbr-1',
        ])
        rec['secondary'] = first_text(rnode, [
            './titles/tertiary-title',
            './titles/short-title',
        ])
        rec['year'] = first_text(rnode, ['./dates/year'])
        rec['volume'] = first_text(rnode, ['./volume'])
        rec['number'] = first_text(rnode, ['./number'])
        rec['pages'] = first_text(rnode, ['./pages'])
        rec['publisher'] = first_text(rnode, ['./publisher'])
        rec['location'] = first_text(rnode, ['./pub-location'])

        authors = all_text(rnode, './contributors/authors/author')
        editors = all_text(rnode, './contributors/editors/editor')
        rec['authors'] = authors
        rec['editors'] = editors

        # URLs: distinguish related (project) urls from pdf urls
        related_urls = all_text(rnode, './urls/related-urls/url')
        pdf_urls = all_text(rnode, './urls/pdf-urls/url')
        # fallback: some exports put urls directly under ./urls/url
        if not related_urls:
            related_urls = all_text(rnode, './urls/url')

        # Prepare initial extra links: prefer a project page from related_urls if present
        initial_links = []
        for u in related_urls:
            if not u:
                continue
            low = u.lower()
            # skip obvious DOIs and PDF links
            if low.endswith('.pdf'):
                continue
            if re.search(r"10\.\d{4,9}/", u):
                continue
            # first sensible related URL treat as a project page
            initial_links.append(('project page', u))
            break
        # include the first pdf url as an available fulltext link (kept for later merging)
        if pdf_urls:
            initial_links.append(('pdf', pdf_urls[0]))

        # DOI extraction (explicit fields or from URLs/texts)
        doi = ''
        for candidate in rnode.findall('.//electronic-resource-num'):
            t = get_text(candidate)
            if t:
                doi = t
                break
        if not doi:
            # generic scan for tag names containing doi
            for elem in rnode.iter():
                if 'doi' in elem.tag.lower():
                    t = get_text(elem)
                    if t:
                        doi = t
                        break
        # If no DOI found yet, check Notes fields for a DOI or arXiv id/url
        if not doi:
            notes_texts = []
            # Common EndNote note tag names
            notes_texts += all_text(rnode, './notes')
            notes_texts += all_text(rnode, './notes/note')
            notes_texts += all_text(rnode, './/note')
            notes_texts += all_text(rnode, './/notes')
            # Try to extract a DOI from notes
            found = _extract_doi(notes_texts)
            if found:
                doi = found
            else:
                # Look for explicit arXiv id/url patterns in notes and convert to a DOI-like form
                for t in notes_texts:
                    if not t:
                        continue
                    m = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s]?)(?P<aid>[0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", t, re.I)
                    if m:
                        aid = m.group('aid')
                        # Store as the canonical 10.48550/arXiv.<id> DOI so arXiv id extraction works later
                        doi = f"10.48550/arXiv.{aid}"
                        break
        rec['doi'] = normalize_doi(doi)
        # Seed any initial links discovered in the XML (project page, pdf)
        rec['extra_links'] = initial_links
        recs.append(rec)  # type: ignore
    return recs

# -------------------- Optional Online Lookup Support --------------------

PUNCT_SPLIT = re.compile(r"[\s\.,:;!\?\-\u2013\u2014\(\)\[\]\"'`+]+")
STOPWORDS = { 'the','a','an','of','and','or','to','in','on','for','with','from','by','at','into','as','is','are','be','using'}

def norm_title_for_lookup(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"<.*?>", " ", t)
    t = t.lower()
    t = PUNCT_SPLIT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def canonical_title_for_exact_match(t: Optional[str]) -> str:
    """Return a canonical form of title suitable for exact equality checks
    that ignore capitalization and punctuation (including most unicode
    punctuation). Accents/diacritics are removed for robustness.
    """
    if not t:
        return ""
    import unicodedata
    s = html.unescape(t)
    s = re.sub(r"<.*?>", " ", s)
    # Normalize and strip combining marks (accents)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Remove characters that are categorized as punctuation in Unicode
    s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'P')
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# similarity threshold to accept near-exact matches when canonical forms differ
ARXIV_EXACT_SIMILARITY = 0.95


def is_open_license(lic: Optional[str]) -> bool:
    if not lic:
        return False
    l = lic.lower()
    # Accept common open licenses (conservative; can expand later)
    allowed_tokens = [
        "cc-by", "cc by", "cc0", "cc-zero", "public domain", "cc-by-sa", "creative commons attribution"
    ]
    return any(tok in l for tok in allowed_tokens)


def safe_fetch(url: str, timeout: float = 12.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode('utf-8', errors='ignore')
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def arxiv_id_from_doi(doi: str) -> Optional[str]:
    # Pattern: 10.48550/arXiv.XXXX.XXXX
    m = re.match(r"10\.48550/\s*arxiv\.(?P<aid>[0-9]{4}\.[0-9]{4,5}(v\d+)?)", doi, re.IGNORECASE)
    if m:
        return m.group('aid')
    return None


def arxiv_lookup(doi: Optional[str], title_norm: str, *, exact: bool=False, verbose: bool=False, authors: Optional[list]=None, title: Optional[str]=None) -> Optional[str]:
    """Look up arXiv preprint via DOI or (fallback) title tokens.
    If exact is True, require the normalized title returned by arXiv for the first hit
    to match title_norm exactly; otherwise discard the result.
    """
    def extract_id_and_title(text: str) -> Tuple[Optional[str], Optional[str]]:
        if not (text and '<entry>' in text):
            return None, None
        m_id = re.search(r"<id>(https?://arxiv.org/abs/[^<]+)</id>", text)
        m_title = re.search(r"<entry>.*?<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        entry_title = html.unescape(m_title.group(1)).strip() if m_title else None
        return (m_id.group(1) if m_id else None), entry_title

    # DOI-based search (rare but fast)
    if doi:
        q = urllib.parse.urlencode({"search_query": f"doi:\"{doi}\"", "start": 0, "max_results": 1})
        text = safe_fetch(f"http://export.arxiv.org/api/query?{q}")
        if text:
            aid, atitle = extract_id_and_title(text)
            if aid:
                if exact and atitle is not None:
                    # Use canonical comparison that ignores punctuation and case
                    left = canonical_title_for_exact_match(atitle)
                    right = canonical_title_for_exact_match(title or title_norm)
                    if left != right:
                        # allow a small fuzzy tolerance (typos / small word differences)
                        from difflib import SequenceMatcher
                        ratio = SequenceMatcher(None, left, right).ratio()
                        if ratio < ARXIV_EXACT_SIMILARITY:
                            if verbose:
                                print(f"[arxiv-exact-skip] DOI query title mismatch: '{atitle}' (sim={ratio:.3f})")
                            return None
                return aid
    # Title fallback (use up to first 10 tokens for query but still enforce full normalized equality if exact)
    if title_norm:
        tokens = title_norm.split()[:10]
        phrase = ' '.join(tokens)
        q = urllib.parse.urlencode({"search_query": f"ti:\"{phrase}\"", "start": 0, "max_results": 1})
        url = f"http://export.arxiv.org/api/query?{q}"
        if verbose:
            print(f"[arxiv-debug] title-phrase query: {url}")
        text = safe_fetch(url)
        if text:
            aid, atitle = extract_id_and_title(text)
            if aid:
                if exact:
                    left = canonical_title_for_exact_match(atitle)
                    right = canonical_title_for_exact_match(title or title_norm)
                    if atitle is None or left != right:
                        from difflib import SequenceMatcher
                        ratio = SequenceMatcher(None, left, right).ratio()
                        if ratio < ARXIV_EXACT_SIMILARITY:
                            if verbose:
                                shown = (atitle or '')[:80]
                                print(f"[arxiv-exact-skip] title mismatch: '{shown}' (sim={ratio:.3f})")
                            return None
                return aid

    # Author+title fallbacks: some arXiv entries don't expose DOI and title phrasing
    # can differ slightly. Try first-author surname + small title phrases.
    if authors and title_norm:
        # Extract simple surname (works for 'Surname, I.' or 'Surname I.')
        first_author = authors[0] if authors else ''
        surname = first_author.split(',')[0].strip() if ',' in first_author else first_author.split()[0] if first_author else ''
        if surname:
            # Prefer longer alphabetic phrases and avoid 1-token numeric phrases
            toks = title_norm.split()
            # skip leading numeric-only tokens (e.g. '6' in '6 dof ...')
            start_idx = 0
            while start_idx < len(toks) and toks[start_idx].isdigit():
                start_idx += 1
            # try lengths in descending preference; ensure phrase contains alphabetic tokens
            for take in (5, 3, 2):
                if start_idx >= len(toks):
                    break
                phrase_toks = toks[start_idx:start_idx + take]
                if not phrase_toks:
                    continue
                # skip if phrase is mostly numeric or too short
                alpha_count = sum(1 for tok in phrase_toks if any(c.isalpha() for c in tok))
                if alpha_count == 0:
                    continue
                phrase = ' '.join(phrase_toks)
                q = urllib.parse.urlencode({"search_query": f"au:\"{surname}\" AND ti:\"{phrase}\"", "start": 0, "max_results": 1})
                url = f"http://export.arxiv.org/api/query?{q}"
                if verbose:
                    print(f"[arxiv-debug] author+title query: {url}")
                text = safe_fetch(url)
                if text:
                    aid, atitle = extract_id_and_title(text)
                    if aid:
                        if exact:
                            left = canonical_title_for_exact_match(atitle)
                            right = canonical_title_for_exact_match(title or title_norm)
                            if atitle is None or left != right:
                                from difflib import SequenceMatcher
                                ratio = SequenceMatcher(None, left, right).ratio()
                                if ratio < ARXIV_EXACT_SIMILARITY:
                                    if verbose:
                                        shown = (atitle or '')[:80]
                                        print(f"[arxiv-exact-skip] author+title mismatch: '{shown}' (sim={ratio:.3f})")
                                    return None
                        return aid
    return None


def openaire_lookup(doi: Optional[str], title_norm: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (url, license) for OpenAIRE record if found."""
    base = "https://api.openaire.eu/search/publications"

    def extract(xml_text: str) -> Tuple[Optional[str], Optional[str]]:
        # url: prefer <url> tag; license from <license> or <bestlicense>
        url_match = re.search(r"<url>(https?://[^<]+)</url>", xml_text, re.IGNORECASE)
        url = url_match.group(1) if url_match else None
        lic = None
        lm = re.search(r"<license>([^<]+)</license>", xml_text, re.IGNORECASE)
        if lm:
            lic = lm.group(1)
        else:
            lm2 = re.search(r"<bestlicense>([^<]+)</bestlicense>", xml_text, re.IGNORECASE)
            if lm2:
                lic = lm2.group(1)
        return url, lic

    if doi:
        q = urllib.parse.urlencode({"doi": doi})
        text = safe_fetch(f"{base}?{q}")
        if text:
            u, lic = extract(text)
            if u:
                return u, lic
    if title_norm:
        first_tokens = ' '.join(title_norm.split()[:6])
        q = urllib.parse.urlencode({"title": first_tokens})
        text = safe_fetch(f"{base}?{q}")
        if text:
            u, lic = extract(text)
            if u:
                return u, lic
    return None, None


def _title_tokens_core(t: str) -> List[str]:
    return [tok for tok in norm_title_for_lookup(t).split() if tok and tok not in STOPWORDS]

def _title_similarity(a: str, b: str) -> float:
    ta = _title_tokens_core(a)
    tb = _title_tokens_core(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    overlap = len(sa & sb)
    # directional coverages
    cov_a = overlap / max(1, len(sa))
    cov_b = overlap / max(1, len(sb))
    # penalize if one side has many extra tokens (spurious long title match)
    penalty = 0.0
    if len(sb) - overlap >= 5 and cov_a < 0.9:  # found title much longer
        penalty = 0.1
    score = min(cov_a, (cov_a + cov_b)/2) - penalty
    if score < 0:
        score = 0.0
    return score

def crossref_lookup(title: str, authors: List[str], year: str, mailto: Optional[str], *, similarity_threshold: float, exact: bool = False) -> Optional[str]:
    """Attempt to find a DOI via Crossref for a record missing one.
    Returns DOI string or None.
    We perform a bibliographic query with title + first author + year and validate by fuzzy title match.
    """
    title_clean = norm_title_for_lookup(title)
    if not title_clean:
        return None
    params_parts = [title_clean]
    if authors:
        params_parts.append(authors[0])
    if year and year.isdigit():
        params_parts.append(year)
    q_bib = " ".join(params_parts)[:300]
    qs = urllib.parse.urlencode({"query.bibliographic": q_bib, "rows": 1})
    url = f"https://api.crossref.org/works?{qs}"
    try:
        req = urllib.request.Request(url)
        ua = "pub-gen-script/1.0"
        if mailto:
            ua += f" (mailto:{mailto})"
        req.add_header("User-Agent", ua)
        with urllib.request.urlopen(req, timeout=12) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None
    try:
        items = data.get('message', {}).get('items', [])
        if not items:
            return None
        item = items[0]
        doi = item.get('DOI')
        found_title_list = item.get('title') or []
        found_title = found_title_list[0] if found_title_list else ''
        found_norm = norm_title_for_lookup(found_title)
        if not found_norm:
            return None
        if exact:
            if found_norm == title_clean and doi:
                return normalize_doi(doi)
            return None
        sim = _title_similarity(title, found_title)
        if sim >= similarity_threshold and doi:
            return normalize_doi(doi)
    except Exception:
        return None
    return None


def _remove_anchor_tags(html_text: str) -> str:
    """Remove <a ...>...</a> tags and normalize whitespace to single spaces."""
    # remove anchors
    s = re.sub(r"<a\b[^>]*>.*?</a>", "", html_text, flags=re.I | re.S)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # remove space before closing li
    s = re.sub(r"\s*</li>\s*$", "</li>", s, flags=re.I)
    # remove space after opening li
    s = re.sub(r"^\s*<li>\s*", "<li>", s, flags=re.I)
    return s


def _li_keys_for_record(rec: Dict) -> List[str]:
    """Return normalized keys representing the record as it appears in HTML (without links).

    We use the current HTML list item produced by format_entry(rec), remove any anchor tags,
    and return two variants:
      1) with the surrounding <li>...</li>
      2) inner content only (without the <li> wrappers)
    """
    li = format_entry(rec) or ""
    li_nolinks = _remove_anchor_tags(li)
    inner = li_nolinks
    if inner.lower().startswith("<li>") and inner.lower().endswith("</li>"):
        inner = inner[4:-5].strip()
    return [li_nolinks, inner]


def augment_records(records: List[Dict], *, cache: Dict[str, Dict], limit: Optional[int], delay: float, verbose: bool, use_openaire: bool, use_crossref: bool, mailto: Optional[str], crossref_threshold: float, crossref_exact: bool, arxiv_exact: bool, skip_set: Optional[Set[str]] = None) -> None:
    """Augment records with arXiv / OpenAIRE (conditional) links.

    Performance optimization:
      * Consult cache BEFORE attempting Crossref lookups. This prevents repeated
        Crossref calls for items already cached by title (previous runs).
      * When a new DOI is discovered, store it in the cache entry as 'found_doi'.
    """
    processed = 0
    for rec in records:
        if limit is not None and processed >= limit:
            break
        processed += 1
        title = rec.get('title') or ''
        title_norm = norm_title_for_lookup(title)
        doi = rec.get('doi') or ''

        # Respect skip list: if the record's rendered HTML (without links) matches an entry,
        # we skip all lookups for this record.
        if skip_set:
            keys = _li_keys_for_record(rec)
            if any(k in skip_set for k in keys):
                if verbose:
                    print(f"[skip-lookups] {title}")
                # Still ensure cache has a minimal entry for consistency
                key_tmp = doi or f"title::{title_norm}"
                cache.setdefault(key_tmp, {
                    'arxiv': None,
                    'full': None,
                    'full_license': None,
                    'found_doi': doi or None,
                    'source_title_norm': title_norm,
                    'source_doi': doi or None,
                    'source_title': title,
                })
                # Do not add any preprint/fulltext links; continue to next
                continue

        # Primary cache key (prefer existing DOI, else normalized title)
        key = doi or f"title::{title_norm}"
        cached = cache.get(key)

        # If not cached and missing DOI, optionally try Crossref (after seeing cache miss)
        if not cached and use_crossref and not doi:
            doi_found = crossref_lookup(title, rec.get('authors', []), rec.get('year') or '', mailto, similarity_threshold=crossref_threshold, exact=crossref_exact)
            if doi_found:
                rec['doi'] = doi = doi_found
                key = doi  # prefer DOI key from now on
                cached = cache.get(key)  # re-check cache in case earlier run stored via DOI

        if cached:
            # Invalidate cache entry if source DOI or normalized title changed since it was stored
            c_title = cached.get('source_title_norm')
            c_doi = cached.get('source_doi')
            invalidate = False
            if c_title is not None and c_title != title_norm:
                invalidate = True
            if c_doi is not None and c_doi != doi:
                invalidate = True
            if invalidate:
                if verbose:
                    print(f"[cache-refresh] invalidated {key} (title/doi changed)")
                cache.pop(key, None)
                cached = None
        if cached:
            if verbose:
                print(f"[cache] {key} title=\"{title}\" doi={doi or 'None'}")
            arxiv_url = cached.get('arxiv')
            full_url = cached.get('full')
            full_lic = cached.get('full_license')
        else:
            # Need to perform lookups
            if verbose:
                print(f"[lookup-start] {key} title=\"{title}\" doi={doi or 'None'}")
            aid = arxiv_id_from_doi(doi) if doi else None
            if aid:
                arxiv_url = f"https://arxiv.org/abs/{aid}"
            else:
                arxiv_url = arxiv_lookup(doi if doi else None, title_norm, exact=arxiv_exact, verbose=verbose, authors=rec.get('authors', []))
            full_url = None
            full_lic = None
            if use_openaire and doi and arxiv_id_from_doi(doi):
                o_url, o_lic = openaire_lookup(doi, title_norm)
                full_url, full_lic = o_url, o_lic
            cache[key] = {
                'arxiv': arxiv_url,
                'full': full_url,
                'full_license': full_lic,
                'found_doi': doi or None,
                'source_title_norm': title_norm,
                'source_doi': doi or None,
                'source_title': title,
            }
            if verbose:
                print(f"[lookup] {key} arxiv={arxiv_url} full={full_url} lic={full_lic}")
            if delay:
                time.sleep(delay)

        links: List[Tuple[str, str]] = []
        if arxiv_url:
            links.append(("preprint", arxiv_url))
        if full_url and full_url != arxiv_url and is_open_license(full_lic):
            links.append(("fulltext", full_url))
        # Merge with any existing extra_links (avoid duplicates)
        existing = rec.get('extra_links') or []
        seen = { (lbl.lower(), url) for lbl, url in existing }
        for lbl, url in links:
            key2 = (lbl.lower(), url)
            if key2 not in seen:
                existing.append((lbl, url))
                seen.add(key2)
        rec['extra_links'] = existing


def ensure_doi_links(records: List[Dict]) -> None:
    """Ensure every record with a DOI gets a [doi] link appended (if not already present)."""
    for rec in records:
        doi = (rec.get('doi') or '').strip()
        if not doi:
            continue
        url = f"https://doi.org/{doi}"
        links = rec.get('extra_links') or []
        if not any(lbl.lower() == 'doi' for lbl, _ in links):
            links.append(('doi', url))
        rec['extra_links'] = links


# -------------------- Local PDF Linking Support --------------------

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
ARTICLES = {"the", "a", "an"}

def _strip_accents_local(s: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _surname_key(first_author: str) -> str:
    raw = first_author.split(',')[0] if first_author else 'anon'
    raw = _strip_accents_local(raw).lower()
    tokens = [t for t in NON_ALNUM_RE.sub(' ', raw).split() if t]
    return ''.join(tokens) if tokens else 'anon'

def _title_key(title: str) -> str:
    t = _strip_accents_local(title).lower()
    tokens = [tok for tok in NON_ALNUM_RE.sub(' ', t).split() if tok]
    if not tokens:
        return 'x'
    if tokens[0] in ARTICLES and len(tokens) > 1:
        return tokens[0] + tokens[1]
    return tokens[0]

def expected_pdf_name(rec: Dict) -> Optional[str]:
    year = (rec.get('year') or '').strip()
    if not year or not re.match(r"^(19|20)\d{2}$", year):
        return None
    authors = rec.get('authors') or []
    if not authors:
        return None
    title = (rec.get('title') or '').strip()
    if not title:
        return None
    surname = _surname_key(authors[0])
    tkey = _title_key(title)
    return f"{year}-{surname}-{tkey}.pdf"

def build_pdf_index(pdf_dir: str) -> Dict[str, str]:
    """Return mapping of filename -> relative/served path for quick membership tests.
    If pdf_dir is a relative path, we keep it as a prefix when constructing link URLs.
    If absolute, we still expose a relative link using only the directory basename
    (so /abs/path/renamed_pdfs/file.pdf becomes renamed_pdfs/file.pdf). This ensures
    no file:// or absolute filesystem paths leak into the HTML output.
    """
    from pathlib import Path as _P
    p = _P(pdf_dir)
    if not p.is_dir():
        return {}
    # Determine link prefix (always relative); strip trailing slashes and '.' components.
    if p.is_absolute():
        rel_prefix = p.name
    else:
        rel_prefix = pdf_dir.rstrip('/') or '.'
        if rel_prefix == '.':
            rel_prefix = ''
    out: Dict[str, str] = {}
    for child in p.iterdir():
        if child.is_file() and child.suffix.lower() == '.pdf':
            name = child.name
            if rel_prefix:
                out[name] = f"{rel_prefix}/{name}" if rel_prefix else name
            else:
                out[name] = name
    return out


def _escape_bibtex(s: str) -> str:
    # Minimal escaping for common BibTeX special characters
    if s is None:
        return ''
    return s.replace('{', '\\{').replace('}', '\\}').replace('%', '\\%')


def _bibtex_key_for(rec: Dict) -> str:
    # year-surnameTitleKey style
    year = (rec.get('year') or '').strip()
    authors = rec.get('authors') or []
    surname = _surname_key(authors[0]) if authors else 'anon'
    tkey = _title_key(rec.get('title') or '')
    if year:
        return f"{year}-{surname}-{tkey}"
    return f"{surname}-{tkey}"


def _format_bibtex_entry(rec: Dict) -> str:
    # Create a conservative BibTeX @article/@inproceedings entry depending on type
    ctype = classify(rec)
    bibtype = 'article' if ctype == 'journal' else 'inproceedings' if ctype == 'conference' else 'misc'
    key = _bibtex_key_for(rec)
    authors = rec.get('authors') or []
    # Convert authors from 'Surname, I.' back to 'Surname, I.' which is acceptable in BibTeX
    author_field = ' and '.join(authors)
    title = rec.get('title') or ''
    year = (rec.get('year') or '').strip()
    journal = rec.get('journal') or rec.get('secondary') or ''
    pages = rec.get('pages') or ''
    volume = rec.get('volume') or ''
    number = rec.get('number') or ''
    doi = (rec.get('doi') or '').strip()
    bib_lines = [f"@{bibtype}{{{key},"]
    if author_field:
        bib_lines.append(f"  author = {{{_escape_bibtex(author_field)}}},")
    if title:
        bib_lines.append(f"  title = {{{_escape_bibtex(title)}}},")
    if journal:
        bib_lines.append(f"  journal = {{{_escape_bibtex(journal)}}},")
    if year:
        bib_lines.append(f"  year = {{{_escape_bibtex(year)}}},")
    if volume:
        bib_lines.append(f"  volume = {{{_escape_bibtex(volume)}}},")
    if number:
        bib_lines.append(f"  number = {{{_escape_bibtex(number)}}},")
    if pages:
        bib_lines.append(f"  pages = {{{_escape_bibtex(pages)}}},")
    if doi:
        bib_lines.append(f"  doi = {{{_escape_bibtex(doi)}}},")
    # remove trailing comma on last field
    if len(bib_lines) > 1:
        bib_lines[-1] = bib_lines[-1].rstrip(',')
    bib_lines.append('}')
    return '\n'.join(bib_lines) + '\n'


def export_bib_files(records: List[Dict], bib_dir: str, *, verbose: bool=False) -> None:
    """Export one .bib file per record using the same naming convention as PDFs.

    Filenames: <year>-<surname>-<titlekey>.bib (same as expected_pdf_name but with .bib)
    """
    from pathlib import Path as _P
    p = _P(bib_dir)
    p.mkdir(parents=True, exist_ok=True)
    # determine relative prefix for links (keep local/relative paths in HTML)
    if p.is_absolute():
        rel_prefix = p.name
    else:
        rel_prefix = bib_dir.rstrip('/') or '.'
        if rel_prefix == '.':
            rel_prefix = ''

    for rec in records:
        fname = expected_pdf_name(rec)
        if not fname:
            # fallback to bibtex key
            fname = f"{_bibtex_key_for(rec)}.bib"
        else:
            fname = re.sub(r"\.pdf$", ".bib", fname, flags=re.I)
        out_path = p / fname
        content = _format_bibtex_entry(rec)
        try:
            rel = f"{rel_prefix}/{fname}" if rel_prefix else fname
            # If file exists and content is identical, skip writing to preserve mtime
            if out_path.is_file():
                try:
                    existing = out_path.read_text(encoding='utf-8')
                except Exception:
                    existing = None
                if existing == content:
                    if verbose:
                        print(f"[bib-skip] {out_path}")
                    # ensure link is present
                    links = rec.get('extra_links') or []
                    if not any(lbl.lower() == 'bibtex' and url == rel for lbl, url in links):
                        links.append(('bibtex', rel))
                    rec['extra_links'] = links
                    continue
                else:
                    out_path.write_text(content, encoding='utf-8')
                    if verbose:
                        print(f"[bib-update] {out_path}")
            else:
                out_path.write_text(content, encoding='utf-8')
                if verbose:
                    print(f"[bib-write] {out_path}")
            # Add a relative bib link to the record's extra_links for HTML rendering
            links = rec.get('extra_links') or []
            if not any(lbl.lower() == 'bibtex' and url == rel for lbl, url in links):
                links.append(('bibtex', rel))
            rec['extra_links'] = links
        except Exception:
            if verbose:
                print(f"[bib-fail] {out_path}")


def add_local_pdf_links(records: List[Dict], pdf_dir: str, *, verbose: bool=False) -> None:
    index = build_pdf_index(pdf_dir)
    if not index:
        if verbose:
            print(f"[pdf-scan] directory empty or not found: {pdf_dir}")
        return
    for rec in records:
        exp = expected_pdf_name(rec)
        if not exp:
            continue
        if exp not in index:
            if verbose:
                print(f"[pdf-missing] {exp}")
            continue
        links = rec.get('extra_links') or []
        if any(lbl.lower() == 'pdf' for lbl, _ in links):
            continue  # already has a pdf link
        links.append(('pdf', index[exp]))
        rec['extra_links'] = links
        if verbose:
            print(f"[pdf-link] {exp} -> {index[exp]}")


def load_cache(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_cache(path: Optional[str], cache: Dict[str, Dict]) -> None:
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def format_entry(rec: Dict) -> str:
    ctype = classify(rec)
    # (Records tagged 'under submission' are filtered out earlier; guard left for safety.)
    if is_under_submission(rec):
        return ""

    authors = join_authors(rec.get("authors", []))
    title_html = f"<i>{html.escape(rec.get('title',''))}</i>"
    year = rec.get("year") or ""
    pages = rec.get("pages")
    pages_fmt = format_pages(pages) if pages else ""
    volume = rec.get("volume")
    number = rec.get("number")
    journal = rec.get("journal") or rec.get("secondary")
    location = rec.get("location")
    publisher = rec.get("publisher")
    editors = rec.get("editors", [])
    editors_joined = join_authors(editors)

    def cleanup_punctuation(line: str) -> str:
        line = re.sub(r",\s*\.", ".", line)
        line = re.sub(r",\s*:", ":", line)
        line = re.sub(r",\s*,+", ",", line)
        line = re.sub(r"\s+,", ",", line)
        return line

    # Link tags
    link_tags = ""
    extra = rec.get("extra_links") or []
    # Guarantee DOI link exists in link_tags if DOI present (in case ensure_doi_links missed or formatting suppressed)
    doi_val = (rec.get('doi') or '').strip()
    if doi_val and not any(lbl.lower() == 'doi' for lbl, _ in extra):
        extra.append(('doi', f'https://doi.org/{doi_val}'))
    if extra:
        parts_a = []
        for label, url in extra:
            parts_a.append(
                f"<a href=\"{html.escape(url, quote=True)}\" target=\"_blank\" rel=\"noopener noreferrer\">[{html.escape(label)}]</a>"
            )
        if parts_a:
            link_tags = " " + " ".join(parts_a)

    if ctype == "conference":
        venue = journal or rec.get("secondary") or ""
        parts = [authors, title_html]
        if venue:
            parts.append(f"in {html.escape(venue)}")
        if location:
            parts.append(html.escape(location))
        if year:
            parts.append(year)
        if pages_fmt:
            parts.append(pages_fmt)
        line = ", ".join([p for p in parts if p]) + "."
        return f"<li> {cleanup_punctuation(line)}{link_tags} </li>"

    if ctype == "journal":
        parts = [authors, title_html]
        if journal:
            parts.append(html.escape(journal))
        volbits = []
        if volume:
            volbits.append(f"vol. {html.escape(volume)}")
        if number:
            volbits.append(f"no. {html.escape(number)}")
        if volbits:
            parts.append(", ".join(volbits))
        if pages_fmt:
            parts.append(pages_fmt)
        if year:
            parts.append(year)
        line = ", ".join([p for p in parts if p]) + "."
        return f"<li> {cleanup_punctuation(line)}{link_tags} </li>"

    if ctype == "book_section":
        book_title = rec.get("secondary") or rec.get("journal") or ""
        parts = [authors, title_html]
        if book_title:
            parts.append(html.escape(book_title))
        if editors_joined:
            ed_suffix = "ed." if len(editors) == 1 else "eds."
            parts.append(f"{editors_joined}, {ed_suffix}")
        colon_seg = []
        if publisher:
            colon_seg.append(html.escape(publisher))
        if pages_fmt:
            colon_seg.append(pages_fmt)
        if year:
            colon_seg.append(year)
        if colon_seg:
            if parts:
                parts[-1] = parts[-1] + ": " + ", ".join(colon_seg)
            else:
                parts.append(": " + ", ".join(colon_seg))
        line = ", ".join([p for p in parts if p])
        if not line.endswith("."):
            line += "."
        return f"<li> {cleanup_punctuation(line)}{link_tags} </li>"

    if ctype == "book":
        parts = [authors, title_html]
        if publisher:
            parts.append(html.escape(publisher))
        if year:
            parts.append(year)
        line = ", ".join([p for p in parts if p]) + "."
        return f"<li> {cleanup_punctuation(line)}{link_tags} </li>"

    # Fallback
    parts = [authors, title_html]
    if journal:
        parts.append(html.escape(journal))
    if year:
        parts.append(year)
    line = ", ".join([p for p in parts if p]) + "."
    return f"<li> {cleanup_punctuation(line)}{link_tags} </li>"


def build_html(records: List[Dict], *, title: str) -> str:
    by_year: Dict[str, List[Dict]] = {}
    for r in records:
        year_raw = (r.get("year") or "").strip()
        # Skip entries lacking a clean 4‑digit year (prevents an 'Unknown' section)
        if not year_raw or not re.match(r"^(19|20)\d{2}$", year_raw):
            continue
        by_year.setdefault(year_raw, []).append(r)
    years = sorted(by_year.keys(), reverse=True)

    # Emit site layout identical to existing pages (container + top nav)
    lines = [
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>',
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />',
        '<link rel="stylesheet" type="text/css" href="style.css" />',
        f"<title>{html.escape(title)}</title>",
        '<script type="text/javascript" src="js/jquery.js"></script>',
        '',
        '</head>',
        '',
        '<body>',
        '<div id="container">',
        '<div id="nav">',
        '<div id="logo"><a href="default.htm"><br />',
        '</a></div>',
        '<ul>',
        '<li><a href="index.html">Home</a><span class="nav_text">Welcome to my site</span></li>',
        '<li><a href="bio.html">Brief Bio</a><span class="nav_text">Brief biography</span></li>',
        '<li><a href="publications.html">Publications</a><span class="nav_text">Full publication list</span></li>',
        '<li><a href="research.html">Research</a><span class="nav_text">My research and projects</span></li>',
        '</ul>',
        '</div>',
        '',
        '<!-- close nav -->',
        '<div id="left-publications">',
        f"<h2>{html.escape(title)}</h2><br/>",
    ]
    for y in years:
        lines.append(f"<h3>{html.escape(y)}</h3>")
        lines.append("<ul>")
        for rec in by_year[y]:
            rendered = format_entry(rec)
            if rendered:
                lines.append(rendered)
        lines.append("</ul>")
    # close left-publications and container, then body/html
    lines.append("</div>")
    lines.append("</div>")
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


## Unpaywall / external OA augmentation removed per user request.


def main():
    ap = argparse.ArgumentParser(description="Generate an HTML publications page from EndNote XML (optional OA lookups)")
    ap.add_argument("xml_path", help="Path to EndNote XML export")
    ap.add_argument("--out", required=True, help="Output HTML file path")
    ap.add_argument("--title", default="Publications", help="Page title / H2 heading")
    ap.add_argument("--lookup", action="store_true", help="Enable online lookups (arXiv + optional Crossref/OpenAIRE)")
    ap.add_argument("--crossref", action="store_true", help="Attempt Crossref DOI search when DOI missing in XML")
    ap.add_argument("--openaire", action="store_true", help="Enable OpenAIRE fulltext lookups (OFF by default)")
    ap.add_argument("--crossref-threshold", type=float, default=0.70, help="Similarity threshold (0-1) for accepting Crossref title match (default 0.70, raise to reduce false positives)")
    ap.add_argument("--crossref-exact", action="store_true", help="Require exact normalized title match for Crossref (overrides threshold)")
    # Exact arXiv title matching is the default behavior; provide an opt-out flag.
    ap.add_argument("--arxiv-exact", dest="arxiv_exact", action="store_true", help="Require canonical exact title match for arXiv results (default)")
    ap.add_argument("--no-arxiv-exact", dest="arxiv_exact", action="store_false", help="Disable canonical exact title matching for arXiv results")
    ap.set_defaults(arxiv_exact=True)
    ap.add_argument("--cache", help="JSON cache file for lookup results")
    ap.add_argument("--clean-cache", action="store_true", help="Remove cache file before performing lookups")
    ap.add_argument("--limit", type=int, help="Limit processing (and lookups) to first N records for quick tests")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between new network lookups (seconds)")
    ap.add_argument("--mailto", help="Contact email for polite Crossref User-Agent (recommended)")
    ap.add_argument("--verbose", action="store_true", help="Verbose lookup logging")
    ap.add_argument("--pdf-dir", help="Directory containing locally named PDFs (<year>-<surname>-<titlekey>.pdf) to auto-link as [pdf]")
    ap.add_argument("--bib-dir", help="Directory to write per-record .bib files (one .bib per entry)")
    ap.add_argument("--skip-list", help="Path to a file OR directory containing references (one per line) to SKIP lookups for. Use the same HTML line as in the generated page; link anchors are ignored.")
    args = ap.parse_args()

    if not Path(args.xml_path).is_file():
        print(f"Input XML not found: {args.xml_path}", file=sys.stderr)
        sys.exit(1)

    recs = parse_records(args.xml_path)
    total = len(recs)
    if args.limit is not None:
        recs = recs[:args.limit]
    if args.lookup:
        # Optionally remove existing cache file to force fresh lookups
        if args.clean_cache and args.cache:
            try:
                p = Path(args.cache)
                if p.is_file():
                    if args.verbose:
                        print(f"[cache] removing cache file: {args.cache}")
                    p.unlink()
            except Exception:
                # best-effort: continue if we cannot delete
                if args.verbose:
                    print(f"[cache] failed to remove cache file: {args.cache}")
        cache = load_cache(args.cache)

        # Pre-fill missing DOIs from cached entries to avoid repeated Crossref lookups
        # (We store 'found_doi' when we previously discovered a DOI via Crossref.)
        for rec in recs:
            if rec.get('doi'):
                continue
            title_norm = norm_title_for_lookup(rec.get('title') or '')
            tkey = f"title::{title_norm}"
            c = cache.get(tkey)
            if c:
                fd = c.get('found_doi')
                if fd:
                    rec['doi'] = fd

        # Load skip list if provided
        skip_set: Optional[Set[str]] = None
        if args.skip_list:
            def _read_skip_file(fp: Path) -> List[str]:
                try:
                    lines = [ln.strip() for ln in fp.read_text(encoding='utf-8').splitlines()]
                except Exception:
                    return []
                out: List[str] = []
                for ln in lines:
                    if not ln or ln.startswith('#'):
                        continue
                    # Normalize similar to runtime
                    ln_norm = _remove_anchor_tags(ln)
                    out.append(ln_norm)
                    # Also include inner-only variant
                    inner = ln_norm
                    if inner.lower().startswith('<li>') and inner.lower().endswith('</li>'):
                        inner = inner[4:-5].strip()
                    out.append(inner)
                return out

            sp = Path(args.skip_list)
            items: List[str] = []
            if sp.is_dir():
                # Read all .txt files in directory
                for child in sorted(sp.iterdir()):
                    if child.is_file() and child.suffix.lower() == '.txt':
                        items.extend(_read_skip_file(child))
            elif sp.is_file():
                items.extend(_read_skip_file(sp))
            if items:
                skip_set = set(items)
                if args.verbose:
                    print(f"[skip-list] loaded {len(skip_set)} entries from {args.skip_list}")

        augment_records(
            recs,
            cache=cache,
            limit=args.limit,
            delay=args.delay,
            verbose=args.verbose,
            use_openaire=args.openaire,
            use_crossref=args.crossref,
            mailto=args.mailto,
            crossref_threshold=1.0 if args.crossref_exact else max(0.0, min(1.0, args.crossref_threshold)),
            crossref_exact=args.crossref_exact,
            arxiv_exact=args.arxiv_exact,
            skip_set=skip_set,
        )
        save_cache(args.cache, cache)
    # Always attach DOI links (after possible augmentation) so they appear before rendering
    ensure_doi_links(recs)
    # Add local PDF links if requested
    if args.pdf_dir:
        add_local_pdf_links(recs, args.pdf_dir, verbose=args.verbose)
    # Export per-record .bib files if requested
    if args.bib_dir:
        export_bib_files(recs, args.bib_dir, verbose=args.verbose)
    html_text = build_html(recs, title=args.title)
    Path(args.out).write_text(html_text, encoding="utf-8")
    print(f"Wrote {len(recs)} records (from {total}) to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
