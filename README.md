# lornat75.github.io

Personal homepage repository.

## Generating `publications.html` from an EndNote XML export

The repo contains tooling to convert an EndNote XML export directly into a year‑grouped HTML page that matches the style of the manually maintained `publications.html`.

### 1. Export references from EndNote
1. In EndNote select the references you want to export (or the whole library).
2. File → Export…
3. Choose a filename like `exported.xml`.
4. Set output format to **XML** (not RIS, not plain text) and save.

### 2. Generate the base HTML page
Use the script `tools/generate_publications_xml_page.py` (pure Python 3 standard library).

```bash
python3 tools/generate_publications_xml_page.py exported.xml \
		--out xml_publications.html \
		--title "Publications"
```

What it does:
* Parses the EndNote XML records.
* Normalizes authors to “Surname, I.” format.
* Groups entries by year (descending) and renders `<ul>` lists under `<h3>YEAR</h3>` sections.
* Attempts a best‑effort formatting based on record type (conference, journal, book section, etc.).
* Skips entries whose fields contain “under submission”.

### 3. (Optional) Add Open Access / preprint links
You can have the generator call Unpaywall (and arXiv as fallback) to attach labeled links such as `[arxiv]`, `[publisher]`, or repository labels.

```bash
python3 tools/generate_publications_xml_page.py exported.xml \
		--out xml_publications_oa.html \
		--title "Publications" \
		--augment-open-access \
		--unpaywall-email you@example.com \
		--oa-cache oa_lookup_cache.json
```

Notes:
* `--unpaywall-email` is required by the Unpaywall API usage policy (a real contact email).
* `--oa-cache` stores a JSON cache of DOI lookups so subsequent runs are fast and offline‑friendly.
* If Unpaywall does not yield an OA link but the DOI exists on arXiv, an `[arxiv]` link is added.

Additional arXiv/Notes behaviour
* The generator also inspects the EndNote `Notes` field for DOIs or arXiv identifiers/URLs. If a DOI of the form `10.48550/arXiv.<id>` or an arXiv URL/ID is present in `Notes`, that arXiv preprint is used directly (no title‑based search).
* By default the script requires a near‑exact canonical title match for arXiv results (this avoids false positives). You can opt out with `--no-arxiv-exact` if you prefer a looser matching strategy.
* Use `--clean-cache` to remove the lookup cache file before running (forces fresh network lookups).

### 4. (Optional) Merge discovered links into the hand‑maintained page
If you keep editing `publications.html` manually but want to bring in new links (arXiv/publisher/repository/pdf) found in the generated file, use the merge tool:

```bash
python3 tools/merge_fulltext_links.py \
		--from xml_publications_oa.html \
		--into publications.html \
		--out publications_merged.html \
		--match-year
```

Explanation:
* Uses the italicized `<i>Title</i>` text (normalized) — and, with `--match-year`, the year — as the key.
* Transfers bracketed links (e.g. `[arxiv]`, `[publisher]`, `[repository]`, `[fulltext]`, `[pdf]`).
* Skips entries that are ambiguous (duplicate normalized titles) in either source or target.
* Produces a merged output without altering other formatting.

You can restrict which labels are transferred:
```bash
python3 tools/merge_fulltext_links.py --from xml_publications_oa.html \
		--into publications.html --out publications_merged.html \
		--labels arxiv,publisher,repository --match-year
```

### 5. Review and replace
Open `publications_merged.html` in a browser, validate formatting, then (optionally) replace the live page:
```bash
mv publications.html publications_backup.html
mv publications_merged.html publications.html
```

### 6. Typical end‑to‑end workflow (quick reference)
```bash
# 1. Export EndNote XML (produces exported.xml)
# 2. Generate with OA augmentation
python3 tools/generate_publications_xml_page.py exported.xml \
	--out xml_publications_oa.html --title "Publications" \
	--augment-open-access --unpaywall-email you@example.com --oa-cache oa_cache.json

# 3. Merge links into manual page
python3 tools/merge_fulltext_links.py \
	--from xml_publications_oa.html --into publications.html \
	--out publications_merged.html --match-year

# 4. Review then publish
mv publications.html publications_old.html
mv publications_merged.html publications.html
```

### Troubleshooting
| Issue | Cause / Fix |
|-------|-------------|
| Script says “Input XML not found” | Verify path to the EndNote export. |
| Few / missing OA links | DOI absent in XML or no OA version; check the record’s DOI field in EndNote. |
| Links not merged | Titles normalized differently; try without `--match-year` or inspect duplicates. |
| ArXiv link missing | DOI might not be associated with an arXiv e-print; arXiv query returns none. |

### Local PDFs and per‑record BibTeX export
You can provide a directory containing locally named PDFs to automatically link them into the generated HTML. The script expects PDFs to be named using a convention similar to:

	<year>-<surname>-<titlekey>.pdf

where `surname` is the (accent‑stripped) surname of the first author and `titlekey` is a short title key (the script uses the same heuristic as the existing site: articles like "the/a/an" are handled, accents stripped, non‑alphanumeric characters removed).

To auto‑link local PDFs, pass the `--pdf-dir` option pointing to the directory containing your `.pdf` files:

```bash
python3 tools/generate_publications_xml_page.py exported.xml \
	--out xml_publications.html --pdf-dir path/to/pdfs
```

If a matching PDF is found for a record, a `[pdf]` link is appended to that entry in the generated HTML.

Per‑record BibTeX files
* The generator can also create one `.bib` file per record. Use `--bib-dir` to instruct the script where to write these files. For convenience you can write them into the site's `pdfs/` directory so they upload together with the PDFs:

```bash
python3 tools/generate_publications_xml_page.py exported.xml \
	--out xml_publications.html --bib-dir lornat75.github.io-sandbox/pdfs
```

* Filenames follow the PDF naming convention and end with `.bib` (e.g. `2025-natale-theicub.bib`).
* The HTML will include a `[bibtex]` link for each record pointing to the relative `.bib` file so links continue to work after uploading the site.
* The exporter is careful: if a `.bib` file already exists and the newly generated content is identical, the file is left untouched (the script logs `[bib-skip]`). If the content differs, the file is overwritten and the script logs `[bib-update]`.

Project / demo pages
* The script looks for project/demo URLs in the EndNote `Related URLs` field (XML path `./urls/related-urls/url`) and, as a fallback, `./urls/url`. The first sensible URL (not a DOI and not a PDF) is treated as a project page and rendered in the HTML as a `[project page]` link for that entry.
* If you prefer to store project links in a different EndNote field, tell the script maintainer the XML tag and it can be mapped easily.

### Skip lookups for specific entries
You can tell the generator to skip DOI/arXiv/fulltext lookups for specific references by listing them in a text file. Each line should be the citation as it appears in the generated HTML (the `<li> ... </li>` line), or just the inner citation text without the `<li>` wrappers. Link anchors are ignored automatically.

1) Create a file like `lornat75.github.io-sandbox/tools/doi-skip.txt` and paste one reference per line. Example lines (both are accepted):

```
<li> Doe, J., and Roe, R., <i>Example Paper Title</i>, Journal of Examples, vol. 12, 2021. </li>
Doe, J., and Roe, R., <i>Example Paper Title</i>, Journal of Examples, vol. 12, 2021.
```

2) Run the generator with the `--skip-list` option. You can pass a file path or a directory. If a directory is provided, all `*.txt` files inside will be read and combined.

```bash
# Adjust path to the generator as needed
python3 /path/to/generate_publications_xml_page.py exported.xml \
	--out publications-from-xml.html \
	--lookup --skip-list tools/doi-skip.txt

# Or scan all .txt files in a folder
python3 /path/to/generate_publications_xml_page.py exported.xml \
	--out publications-from-xml.html \
	--lookup --skip-list tools/
```

Notes:
* Only lookups are skipped for matching entries; existing DOIs present in the XML will still be rendered as `[doi]` links.
* Matching is robust to the presence or absence of link anchors in the line; whitespace differences are normalized.

### Implementation Notes
* All scripts use only the Python standard library (no extra dependencies).
* Normalization removes HTML tags inside titles and collapses whitespace; this ensures stable matching.
* The formatting intentionally keeps close to the existing manual style and may need light manual touch‑ups for edge cases.

---
Feel free to extend these tools (e.g., adding BibTeX export, local PDF integration, or stricter duplicate detection). Contributions or tweaks for your workflow are straightforward: each helper script is in `tools/` and self‑contained.
