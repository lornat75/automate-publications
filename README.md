# Publications processing tools

Small Python utilities to parse EndNote XML exports and generate enriched HTML publication pages.

All scripts use only the Python standard library. Python 3.8+ recommended.

## Tools

- `parse_endnote_xml.py` — parses an EndNote XML export (`.xml`)
- `generate_publications_xml_page.py` — build a fresh publications HTML page directly from an EndNote XML export with optional online enrichment (arXiv preprints, Crossref DOI completion, optional OpenAIRE fulltext)

## Data schema (shared)

Each parsed publication is a dictionary with:

- `id`: stable identifier; `doi:<lowercase-doi>` when available, otherwise `title:<slugified-title>:<year>`
- `year`: integer year if detectable
- `authors`: list of strings, each like `Surname, I.` (multiple initials preserved)
- `title`: string
- `venue`: free-form description (journal/conference/volume info parsed from the text following the title)
- `doi`: DOI string if found (without the `https://doi.org/` prefix)
- `preprint_url`: URL to preprint if present
- `fulltext_url`: best-guess full text URL (`fulltext`/`online text`/`pdf` preferred)
- `bibtex_url`: URL to a .bib entry if present
- `links`: mapping of all discovered anchor texts (lowercased) to URLs
- `source`: `html` or `endnote-xml`

## Quick start

From the repository root:

```bash
# Parse an EndNote XML export (File -> Export -> XML in EndNote)
python3 tools/parse_endnote_xml.py /path/to/export.xml --print
python3 tools/parse_endnote_xml.py /path/to/export.xml --json tools/endnote_publications.json
```

### Generate a new enriched HTML page from EndNote XML

```bash
# Basic (no network lookups)
python3 tools/generate_publications_xml_page.py /path/to/export.xml --out publications-from-xml.html

# With arXiv + Crossref DOI completion (recommended) using a polite User-Agent including your email
python3 tools/generate_publications_xml_page.py /path/to/export.xml --out publications-from-xml.html \
	--lookup --crossref --mailto you@example.com --cache tools/oa_cache.json

# Add OpenAIRE fulltext discovery (slower, optional)
python3 tools/generate_publications_xml_page.py /path/to/export.xml --out publications-from-xml.html \
	--lookup --crossref --openaire --mailto you@example.com --cache tools/oa_cache.json --delay 0.5

# Limit to first N records for quick tests
python3 tools/generate_publications_xml_page.py /path/to/export.xml --out test.html \
	--lookup --crossref --limit 10 --verbose
```

Key flags:

- `--lookup` enable any network augmentation.
- `--crossref` fill in missing DOIs via Crossref (authoritative DOI source). Provide `--mailto` for etiquette.
- `--openaire` (off by default) attempt OpenAIRE lookups BUT only for records whose DOI is an arXiv DOI (10.48550/arXiv.*); if found open-licensed, adds `[fulltext]`.
- `--cache <file>` JSON cache of previous lookups (speeds repeated runs; safe to commit if desired).
- `--limit N` process only first N records (applies to lookups) for experimentation.
- `--delay S` sleep S seconds between new (non-cached) requests to be gentle to APIs.
- `--verbose` print lookup / cache diagnostics.

Link labels in generated HTML:

- `[preprint]` an arXiv abstract page (always included if arXiv found or implied by 10.48550/arXiv DOI).
- `[doi]` canonical DOI resolver link (always included when a DOI is known or discovered).
- `[fulltext]` open licensed version discovered via OpenAIRE (only when `--openaire` used AND DOI is an arXiv DOI AND license is permissive: CC-BY/SA, CC0, Public Domain).

Filtering rules:
- Entries whose titles or notes indicate "under submission", "under review", "in review", "submitted" are skipped.
- Records with unparseable years or year outside reasonable range (19xx/20xx) are ignored.

Caching details:
- Cache key is DOI when present, else a normalized title fingerprint (lower-case alphanumerics collapsed).
- Stored fields: `arxiv`, `full`, `full_license`, `found_doi`, `source_title_norm`, `source_doi`, `source_title` (human-readable title for easier manual inspection).
- Automatic refresh: if the normalized title or DOI in the source XML differs from `source_title_norm` / `source_doi` in cache, the entry is invalidated and re-fetched.
- Delete the cache file to force fresh lookups.

Verbose logging (`--verbose`):
- `[cache] <key> title="…" doi=…` — cache hit, no network calls.
- `[cache-refresh] invalidated <key> (title/doi changed)` — cached metadata differs from current XML; entry discarded and re-fetched.
- `[lookup-start] <key> title="…" doi=…` — beginning network augmentation for a cache miss.
- `[lookup] <key> arxiv=<url-or-None> full=<url-or-None> lic=<license-or-None>` — results after performing arXiv (and conditional OpenAIRE) lookups; new cache entry written.

Notes:
- A record that later acquires a DOI (via Crossref) will generate a new DOI-based cache key; the older `title::` key can be left or manually pruned—functionality is unaffected.
- `found_doi` records a DOI discovered via Crossref so subsequent runs can skip another query even if the original XML still lacks it.

## Notes and assumptions

- EndNote XML parser expects an XML export (not `.enl`), and reads typical fields like titles, periodical full title, year, authors, and related/pdf URLs.
- Author strings are processed heuristically; initials like `G.M.` are preserved.
- DOIs are extracted from inline text or DOI URLs.

## Notes on EndNote XML

Export from EndNote using File -> Export -> XML. Native `.enl` libraries are proprietary; use XML for comparison.
