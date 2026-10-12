"""Check lowercase documentation filenames and generated MkDocs page URLs."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


def check_filenames(paths: list[str]) -> list[str]:
    """Check tracked Markdown guides, not conventional repository files/assets."""
    errors: list[str] = []
    seen: dict[str, str] = {}
    for name in sorted(paths):
        path = PurePosixPath(name)
        parts = path.parts
        in_docs = parts[0].lower() == "docs" if parts else False
        in_package_docs = len(parts) >= 4 and parts[0].lower() == "packages" and parts[2].lower() == "docs"
        if not (in_docs or in_package_docs) or path.suffix.lower() != ".md":
            continue
        key = name.casefold()
        previous = seen.get(key)
        if previous is not None and previous != name:
            errors.append(f"Case-folding collision: {previous} and {name}")
        seen[key] = name
        if any(part != part.lower() for part in parts[:-1]):
            errors.append(f"Documentation directory must be lowercase: {name}")
        if path.name != "README.md" and path.name != path.name.lower():
            errors.append(f"Documentation filename must be lowercase: {name}")
    return errors


def tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True,
        text=True, encoding="utf-8",
    )
    return [name for name in result.stdout.split("\0") if name]


class CanonicalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "link" and "canonical" in (attributes.get("rel") or "").lower().split():
            self.urls.append(attributes.get("href") or "")


def check_url(url: str, source: str, *, allow_empty: bool = False) -> list[str]:
    """Hosts, query strings, fragments, and percent-escape casing are not paths."""
    if allow_empty and url == "":
        return []  # MkDocs search uses an empty location for the homepage.
    if not isinstance(url, str) or not url:
        return [f"Missing page URL in {source}"]
    path = unquote(urlsplit(url).path)
    if path != path.lower():
        return [f"Page URL path must be lowercase in {source}: {url}"]
    return []


def check_site(site_dir: Path) -> list[str]:
    """Check pages advertised in the sitemap, leaving HTML inspection assets alone."""
    errors: list[str] = []
    sitemap = site_dir / "sitemap.xml"
    try:
        document = ET.parse(sitemap)
    except (OSError, ET.ParseError) as exc:
        return [f"Cannot read sitemap {sitemap}: {exc}"]
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    urls = [element.findtext(namespace + "loc") or "" for element in document.findall(namespace + "url")]
    if not urls:
        errors.append(f"No page URLs in {sitemap}")
    for url in urls:
        errors.extend(check_url(url, "sitemap"))
        if not url:
            continue
        # These sites are hosted at domain roots, not under a site_url subpath.
        relative = unquote(urlsplit(url).path).lstrip("/")
        page = site_dir / relative
        if not relative or relative.endswith("/"):
            page = page / "index.html"
        if not page.resolve().is_relative_to(site_dir.resolve()):
            errors.append(f"Page URL escapes the site directory: {url}")
            continue
        try:
            parser = CanonicalParser()
            parser.feed(page.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append(f"Cannot read sitemap page {page}: {exc}")
            continue
        if len(parser.urls) != 1:
            errors.append(f"Expected one canonical URL in {page}, found {len(parser.urls)}")
        for canonical in parser.urls:
            errors.extend(check_url(canonical, f"canonical in {page}"))
    search_index = site_dir / "search" / "search_index.json"
    try:
        entries = json.loads(search_index.read_text(encoding="utf-8"))["docs"]
        if not isinstance(entries, list):
            raise ValueError("docs must be a list")
        if not entries:
            errors.append(f"No page locations in {search_index}")
        for entry in entries:
            errors.extend(check_url(entry["location"], "search index", allow_empty=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Cannot read search index {search_index}: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--site-dir", type=Path, help="Also validate a freshly built site")
    arguments = parser.parse_args(argv)
    try:
        errors = check_filenames(tracked_paths(arguments.repo_root))
    except subprocess.CalledProcessError as exc:
        print(f"Cannot list tracked files: {exc.stderr.strip()}", file=sys.stderr)
        return 1
    if arguments.site_dir is not None:
        errors.extend(check_site(arguments.site_dir))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Documentation paths are lowercase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
