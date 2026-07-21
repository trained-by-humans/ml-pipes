from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


def _load_capture_html_screenshot_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "docs_assets" / "capture_html_screenshot.py"
    spec = importlib.util.spec_from_file_location("capture_html_screenshot", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeLocator:
    def __init__(self, page: "_FakePage", selector: str) -> None:
        self.page = page
        self.selector = selector

    @property
    def first(self) -> "_FakeLocator":
        return self

    def wait_for(self, *, state: str, timeout: int) -> None:
        self.page.events.append(("wait_for", self.selector, state, timeout))

    def evaluate(self, _expression: str) -> dict[str, int]:
        self.page.events.append(("locator_evaluate", self.selector))
        return self.page.selector_dimensions[self.selector]

    def hover(self, *, timeout: int) -> None:
        self.page.events.append(("hover", self.selector, timeout))

    def click(self, *, timeout: int) -> None:
        self.page.events.append(("click", self.selector, timeout))

    def screenshot(self, *, path: str, animations: str) -> None:
        self.page.events.append(("locator_screenshot", self.selector, path, animations))


class _FakePage:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.selector_dimensions = {
            ".insp-container": {"width": 2_100, "height": 480},
        }
        self.page_dimensions = {"width": 1_200, "height": 900}
        self.page_kwargs: dict[str, object] = {}

    def goto(self, source_url: str, *, wait_until: str, timeout: int) -> None:
        self.events.append(("goto", source_url, wait_until, timeout))

    def locator(self, selector: str) -> _FakeLocator:
        self.events.append(("locator", selector))
        return _FakeLocator(self, selector)

    def wait_for_timeout(self, delay_ms: int) -> None:
        self.events.append(("wait", delay_ms))

    def set_viewport_size(self, size: dict[str, int]) -> None:
        self.events.append(("viewport", size["width"], size["height"]))

    def evaluate(self, _expression: str) -> dict[str, int]:
        self.events.append(("page_evaluate",))
        return self.page_dimensions

    def screenshot(self, *, path: str, full_page: bool, animations: str) -> None:
        self.events.append(("page_screenshot", path, full_page, animations))


class _FakeBrowser:
    def __init__(self) -> None:
        self.page = _FakePage()
        self.closed = False
        self.new_page_kwargs: dict[str, object] | None = None

    def new_page(self, **kwargs: object) -> _FakePage:
        self.new_page_kwargs = kwargs
        return self.page

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self) -> None:
        self.browser = _FakeBrowser()
        self.launch_kwargs: dict[str, object] | None = None

    def launch(self, **kwargs: object) -> _FakeBrowser:
        self.launch_kwargs = kwargs
        return self.browser


class _FakePlaywrightApi:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()


class _FakePlaywrightContext:
    def __init__(self, api: _FakePlaywrightApi) -> None:
        self.api = api

    def __enter__(self) -> _FakePlaywrightApi:
        return self.api

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePlaywrightModule:
    class Error(Exception):
        pass

    def __init__(self) -> None:
        self.api = _FakePlaywrightApi()

    def sync_playwright(self) -> _FakePlaywrightContext:
        return _FakePlaywrightContext(self.api)


def test_resolve_source_url_converts_local_file_to_file_uri(tmp_path: Path) -> None:
    module = _load_capture_html_screenshot_module()
    report = tmp_path / "report.html"
    report.write_text("<html></html>", encoding="utf-8")

    assert module._resolve_source_url(report) == report.resolve().as_uri()
    assert module._resolve_source_url(str(report)) == report.resolve().as_uri()


def test_require_playwright_raises_clear_error_when_dependency_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_capture_html_screenshot_module()

    def fake_import_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires Playwright"):
        module._require_playwright()


def test_capture_html_screenshot_drives_browser_with_interactions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_capture_html_screenshot_module()
    fake_playwright = _FakePlaywrightModule()
    report = tmp_path / "report.html"
    output = tmp_path / "captures" / "report.png"
    report.write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(module, "_require_playwright", lambda: fake_playwright)

    saved = module.capture_html_screenshot(
        report,
        output=output,
        selector=".insp-container",
        wait_for=".insp-container",
        hover=[".insp-cfg-icon"],
        click=["img[data-overlay]"],
        delay_ms=10,
        timeout_ms=5_000,
        width=1_280,
        height=720,
        device_scale_factor=2.0,
        fit_selector=".insp-container",
    )

    browser = fake_playwright.api.chromium.browser
    page = browser.page

    assert saved == output
    assert output.parent.is_dir()
    assert fake_playwright.api.chromium.launch_kwargs == {"headless": True}
    assert browser.new_page_kwargs == {
        "viewport": {"width": 1_280, "height": 720},
        "device_scale_factor": 2.0,
    }
    assert ("goto", report.resolve().as_uri(), "load", 5_000) in page.events
    assert ("wait_for", ".insp-container", "visible", 5_000) in page.events
    assert ("viewport", 2_124, 504) in page.events
    assert ("hover", ".insp-cfg-icon", 5_000) in page.events
    assert ("click", "img[data-overlay]", 5_000) in page.events
    assert ("locator_screenshot", ".insp-container", str(output), "disabled") in page.events
    assert browser.closed is True


def test_main_prints_saved_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    module = _load_capture_html_screenshot_module()
    output = tmp_path / "report.png"

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(
            source="report.html",
            output=output,
            selector=None,
            wait_for=None,
            hover=[],
            click=[],
            delay_ms=250,
            timeout_ms=30_000,
            width=1_600,
            height=900,
            device_scale_factor=1.0,
            fit_selector=None,
            fit_page=False,
            fit_padding=24,
            max_width=module._DEFAULT_MAX_DIMENSION,
            max_height=module._DEFAULT_MAX_DIMENSION,
            full_page=False,
            browser_path=None,
        ),
    )
    monkeypatch.setattr(module, "capture_html_screenshot", lambda *_args, **_kwargs: output)

    assert module.main() == 0
    assert capsys.readouterr().out.strip() == f"Saved screenshot: {output}"
