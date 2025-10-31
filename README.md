# automate-publications

Generate a publications HTML page directly from an EndNote XML export. Pure Python 3 standard library.

The main script is `generate_publications_xml_page.py`. It parses EndNote XML, groups entries by year, formats each citation, optionally augments with arXiv/OpenAIRE links, auto-links local PDFs, and can export one `.bib` per record. The generated HTML includes the site’s layout (container + top navigation) so it can be dropped into the existing website.

## Quick start

1) Export from EndNote as XML (not RIS or text).

2) Generate HTML:

```bash
python3 generate_publications_xml_page.py \
  /path/to/publications-endnote.xml \
  --out /path/to/publications-from-xml.html \
  --title "Publications"
```

Defaults:
- Page title defaults to “Publications” if `--title` is omitted.
- Entries with “under submission/review” hints are skipped.

## Optional lookups and cache

```bash
python3 generate_publications_xml_page.py input.xml \
  --out out.html \
  --lookup \
  --crossref \
  --openaire \
  --cache oa_cache.json \
  --clean-cache \
  --mailto you@example.com \
  --verbose
```

Notes:
- arXiv: Title queries require canonical exact match by default; disable with `--no-arxiv-exact`.
- Crossref: controlled by `--crossref`, threshold via `--crossref-threshold` or `--crossref-exact`.
- The EndNote “Notes” field is scanned for DOI/arXiv identifiers; arXiv DOIs like `10.48550/arXiv.<id>` short-circuit the lookup.
- `--cache` stores lookup results; `--clean-cache` removes it before a run.

## Local PDFs

Pass a directory containing PDFs named like `<year>-<surname>-<titlekey>.pdf` to auto-add `[pdf]` links:

```bash
python3 generate_publications_xml_page.py input.xml \
  --out out.html \
  --pdf-dir /path/to/site/pdfs \
  --verbose
```

Naming:
- `surname` is the accent-stripped surname of the first author.
- `titlekey` is derived from the title (articles handled, punctuation/accents stripped).

## Per-record BibTeX export

Write one `.bib` file per entry (filenames mirror the PDF naming convention):

```bash
python3 generate_publications_xml_page.py input.xml \
  --out out.html \
  --bib-dir /path/to/site/pdfs \
  --verbose
```

Behavior:
- If an existing `.bib` file is identical, it is left untouched (`[bib-skip]`); otherwise it is overwritten (`[bib-update]`).
- The HTML includes a `[bibtex]` link for each exported entry.

## Skip DOI/arXiv lookups for specific entries

Use `--skip-list` to point to a text file (or a directory of `.txt` files) listing references for which network lookups should be skipped. This does NOT remove entries from the page and does NOT hide DOIs present in the XML—it only prevents DOI/arXiv/fulltext lookups for the matching entries.

Accepted line formats (one per line):
- The exact `<li> ... </li>` line copied from the generated HTML, or
- The same line without the surrounding `<li>` wrappers, or
- Plain text without HTML tags. Link anchors are ignored automatically, and whitespace is normalized.

Example:

```bash
python3 generate_publications_xml_page.py input.xml \
  --out out.html \
  --lookup \
  --skip-list /path/to/skip.txt \
  --verbose
```

Logs:
- You will see `[skip-list] loaded N entries …` when read.
- For matched records: `[skip-lookups] <title>`.

## Site layout

The generated HTML includes the site’s container and top navigation markup (doctype, head with `style.css` and `js/jquery.js`, `#container` + `#nav`). The H2 heading uses `--title`.

## CLI reference (high level)

- `--out PATH`: Output HTML file (required)
- `--title TEXT`: Page and H2 title (default: “Publications”)
- `--lookup`: Enable network lookups
- `--crossref`, `--crossref-threshold FLOAT`, `--crossref-exact`
- `--openaire`
- `--arxiv-exact` / `--no-arxiv-exact`
- `--cache PATH`, `--clean-cache`
- `--limit N`, `--delay SECONDS`, `--mailto EMAIL`, `--verbose`
- `--pdf-dir DIR`: Auto-link local PDFs
- `--bib-dir DIR`: Export per-record `.bib` files and add `[bibtex]` links
- `--skip-list PATH|DIR`: Skip lookups for listed references (keeps entries; does not hide XML DOIs)

## Troubleshooting

- No `[pdf]` link: filename doesn’t match the expected convention or PDF not present in the directory.
- No `[preprint]`: arXiv match requires canonical title equality by default; try `--no-arxiv-exact` if needed.
- Wrong DOI found: increase `--crossref-threshold`, or add the item to `--skip-list` and provide the DOI in the XML.
- Skips not applied: copy the full `<li> ... </li>` line from the generated HTML into the skip file (or paste the inner text); anchors are ignored.

---
This tool uses only the Python standard library and is designed to be conservative (exact/fuzzy matching with cache). Adjust options as needed for your workflow.
