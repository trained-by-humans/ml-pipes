from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def _load_checker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_docs_paths.py"
    spec = importlib.util.spec_from_file_location("check_docs_paths", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker():
    return _load_checker()


def test_accepts_lowercase_guides_and_conventional_exceptions(checker):
    assert checker.check_filenames([
        "docs/performance.md", "docs/verified_community_packages.md",
        "docs/README.md", "docs/index.md", "packages/core/docs/reference.md",
        "packages/core/docs/index.md", "packages/torch/docs/README.md",
        "README.md", "AGENTS.md", "CONTRIBUTING.md", "RELEASE.md", "SECURITY.md",
        ".agents/skills/maintainer-core/SKILL.md", "docs/assets/RESULT.PNG",
    ]) == []


@pytest.mark.parametrize("path", [
    "docs/PERFORMANCE.md", "docs/guide.MD", "docs/README.MD", "docs/Guide/index.md",
    "packages/core/docs/INDEX.md", "packages/Core/docs/index.md", "docs/Guide/README.md",
    "Docs/guide.md", "packages/core/Docs/guide.md",
])
def test_rejects_uppercase_guide_paths(checker, path):
    assert checker.check_filenames([path])


def test_detects_case_folding_collisions_without_using_the_filesystem(checker):
    errors = checker.check_filenames(["docs/guide.md", "docs/Guide.md"])
    assert any("collision" in error for error in errors)


def test_keeps_package_references_distinct(checker):
    assert checker.check_filenames([
        "packages/core/docs/reference.md", "packages/tensor/docs/reference.md",
    ]) == []


@pytest.mark.parametrize("url", [
    "https://EXAMPLE.com/guide/#OperatorAPI", "guide/?mode=GPU#ClassName",
    "https://example.com/%67uide/", "",
])
def test_url_checks_ignore_non_path_case_but_reject_empty_urls(checker, url):
    assert bool(checker.check_url(url, "test")) == (not url)


@pytest.mark.parametrize("url", [
    "https://example.com/Guide/", "Guide/#anchor", "https://example.com/%47uide/",
])
def test_url_checks_reject_uppercase_and_encoded_uppercase(checker, url):
    assert checker.check_url(url, "test")


def _site(root, sitemap_url="https://example.com/guide/", canonical_url=None, search_url="guide/#OperatorAPI"):
    canonical_url = sitemap_url if canonical_url is None else canonical_url
    root.mkdir(parents=True, exist_ok=True)
    (root / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{sitemap_url}</loc></url></urlset>", encoding="utf-8",
    )
    guide = root / sitemap_url.split("example.com/")[-1]
    guide.mkdir(parents=True, exist_ok=True)
    (guide / "index.html").write_text(
        f'<html><head><link rel="canonical" href="{canonical_url}"></head></html>', encoding="utf-8",
    )
    (root / "search").mkdir(exist_ok=True)
    (root / "search" / "search_index.json").write_text(
        json.dumps({"docs": [{"location": search_url}]}), encoding="utf-8",
    )


def test_site_accepts_lowercase_pages_and_leaves_inspection_assets_alone(checker, tmp_path):
    _site(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "Inspection.HTML").write_text("<html>No canonical required</html>")
    assert checker.check_site(tmp_path) == []


def test_site_accepts_the_empty_homepage_search_location(checker, tmp_path):
    _site(tmp_path, search_url="")
    assert checker.check_site(tmp_path) == []


@pytest.mark.parametrize("options,source", [
    ({"sitemap_url": "https://example.com/Guide/"}, "sitemap"),
    ({"canonical_url": "https://example.com/Guide/"}, "canonical"),
    ({"search_url": "Guide/#OperatorAPI"}, "search index"),
])
def test_site_rejects_uppercase_in_each_generated_surface(checker, tmp_path, options, source):
    _site(tmp_path, **options)
    assert any(source in error for error in checker.check_site(tmp_path))


def test_published_readme_is_not_exempt_from_lowercase_urls(checker, tmp_path):
    _site(tmp_path, sitemap_url="https://example.com/README/")
    assert checker.check_site(tmp_path)


def test_site_rejects_missing_canonical(checker, tmp_path):
    _site(tmp_path)
    (tmp_path / "guide" / "index.html").write_text("<html></html>")
    assert any("canonical" in error for error in checker.check_site(tmp_path))


def test_site_requires_sitemap_and_search_index(checker, tmp_path):
    assert any("sitemap" in error for error in checker.check_site(tmp_path))
    _site(tmp_path)
    (tmp_path / "search" / "search_index.json").unlink()
    assert any("search index" in error for error in checker.check_site(tmp_path))


def test_site_rejects_empty_sitemap_and_malformed_search_index(checker, tmp_path):
    _site(tmp_path)
    (tmp_path / "sitemap.xml").write_text('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>')
    (tmp_path / "search" / "search_index.json").write_text("not JSON")
    errors = checker.check_site(tmp_path)
    assert any("No page URLs" in error for error in errors)
    assert any("search index" in error for error in errors)


def test_main_checks_tracked_files_not_untracked_files(checker, tmp_path, capsys):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide")
    (docs / "UNTRACKED.md").write_text("# Scratch")
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/guide.md"], check=True)
    assert checker.main(["--repo-root", str(tmp_path)]) == 0
    assert "lowercase" in capsys.readouterr().out
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/UNTRACKED.md"], check=True)
    assert checker.main(["--repo-root", str(tmp_path)]) == 1
    assert "UNTRACKED.md" in capsys.readouterr().err


def test_main_checks_generated_site_when_requested(checker, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(checker, "tracked_paths", lambda root: ["docs/guide.md"])
    _site(tmp_path, canonical_url="https://example.com/Guide/")
    assert checker.main(["--site-dir", str(tmp_path)]) == 1
    assert "canonical" in capsys.readouterr().err
