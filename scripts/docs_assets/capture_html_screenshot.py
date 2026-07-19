from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from urllib.parse import urlparse

_DEFAULT_MAX_DIMENSION = 16_384
_URL_SCHEMES = {"about", "data", "file", "http", "https"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a PNG screenshot from a local HTML file or URL using headless Chromium "
            "through Playwright."
        )
    )
    parser.add_argument(
        "source",
        help="Local HTML file path or URL to render.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--selector",
        default=None,
        help="Optional CSS selector to capture instead of the full page.",
    )
    parser.add_argument(
        "--wait-for",
        default=None,
        metavar="SELECTOR",
        help="Optional CSS selector to wait for before capture.",
    )
    parser.add_argument(
        "--hover",
        action="append",
        default=[],
        metavar="SELECTOR",
        help="CSS selector to hover before capture. Repeatable.",
    )
    parser.add_argument(
        "--click",
        action="append",
        default=[],
        metavar="SELECTOR",
        help="CSS selector to click before capture. Repeatable.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=250,
        metavar="N",
        help="Delay after each interaction and before capture in milliseconds (default: 250).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30_000,
        metavar="N",
        help="Playwright timeout in milliseconds (default: 30000).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1_600,
        metavar="PX",
        help="Initial viewport width in CSS pixels (default: 1600).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        metavar="PX",
        help="Initial viewport height in CSS pixels (default: 900).",
    )
    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=1.0,
        metavar="N",
        help="Device scale factor for the browser page (default: 1.0).",
    )
    parser.add_argument(
        "--fit-selector",
        default=None,
        metavar="SELECTOR",
        help=(
            "Resize the viewport to this selector's scroll size before capture. "
            "Useful for horizontally scrolling inspection reports."
        ),
    )
    parser.add_argument(
        "--fit-page",
        action="store_true",
        help="Resize the viewport to the full document scroll size before capture.",
    )
    parser.add_argument(
        "--fit-padding",
        type=int,
        default=24,
        metavar="PX",
        help="Extra padding added when fitting the viewport (default: 24).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=_DEFAULT_MAX_DIMENSION,
        metavar="PX",
        help=f"Maximum fitted viewport width (default: {_DEFAULT_MAX_DIMENSION}).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=_DEFAULT_MAX_DIMENSION,
        metavar="PX",
        help=f"Maximum fitted viewport height (default: {_DEFAULT_MAX_DIMENSION}).",
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="When not using --selector, capture the full page height.",
    )
    parser.add_argument(
        "--browser-path",
        type=Path,
        default=None,
        help="Optional Chromium or Chrome executable path.",
    )
    return parser.parse_args()


def _resolve_source_url(source: str | Path) -> str:
    if isinstance(source, Path):
        path = source.expanduser()
        if not path.is_file():
            raise ValueError(f"HTML source file does not exist: {path}")
        return path.resolve().as_uri()

    parsed = urlparse(source)
    if parsed.scheme in _URL_SCHEMES:
        return source

    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"HTML source file does not exist: {path}")
    return path.resolve().as_uri()


def _require_positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _require_non_negative_int(value: int, *, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _require_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _wait(page: object, delay_ms: int) -> None:
    if delay_ms > 0:
        page.wait_for_timeout(delay_ms)


def _require_playwright():
    try:
        return importlib.import_module("playwright.sync_api")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HTML screenshot capture requires Playwright. "
            "Install it with: python -m pip install playwright "
            "and then run: python -m playwright install chromium"
        ) from exc


def _first_locator(page: object, selector: str):
    return page.locator(selector).first


def _fit_viewport_to_selector(
    page: object,
    selector: str,
    *,
    timeout_ms: int,
    padding: int,
    max_width: int,
    max_height: int,
) -> None:
    locator = _first_locator(page, selector)
    locator.wait_for(state="visible", timeout=timeout_ms)
    # Expand to the element's scroll box so horizontally scrolling reports can be captured
    # as one image instead of only the current viewport.
    dimensions = locator.evaluate(
        """(element) => ({
            width: Math.ceil(Math.max(
                element.scrollWidth,
                element.clientWidth,
                element.getBoundingClientRect().width
            )),
            height: Math.ceil(Math.max(
                element.scrollHeight,
                element.clientHeight,
                element.getBoundingClientRect().height
            ))
        })"""
    )
    width = min(max_width, max(1, int(dimensions["width"]) + padding))
    height = min(max_height, max(1, int(dimensions["height"]) + padding))
    page.set_viewport_size({"width": width, "height": height})


def _fit_viewport_to_page(
    page: object,
    *,
    padding: int,
    max_width: int,
    max_height: int,
) -> None:
    dimensions = page.evaluate(
        """() => ({
            width: Math.ceil(Math.max(
                document.documentElement.scrollWidth,
                document.body ? document.body.scrollWidth : 0
            )),
            height: Math.ceil(Math.max(
                document.documentElement.scrollHeight,
                document.body ? document.body.scrollHeight : 0
            ))
        })"""
    )
    width = min(max_width, max(1, int(dimensions["width"]) + padding))
    height = min(max_height, max(1, int(dimensions["height"]) + padding))
    page.set_viewport_size({"width": width, "height": height})


def _fit_viewport(
    page: object,
    *,
    fit_selector: str | None,
    fit_page: bool,
    timeout_ms: int,
    fit_padding: int,
    max_width: int,
    max_height: int,
) -> None:
    if fit_selector is not None:
        _fit_viewport_to_selector(
            page,
            fit_selector,
            timeout_ms=timeout_ms,
            padding=fit_padding,
            max_width=max_width,
            max_height=max_height,
        )
        return

    if fit_page:
        _fit_viewport_to_page(
            page,
            padding=fit_padding,
            max_width=max_width,
            max_height=max_height,
        )


def capture_html_screenshot(
    source: str | Path,
    *,
    output: Path,
    selector: str | None = None,
    wait_for: str | None = None,
    hover: list[str] | None = None,
    click: list[str] | None = None,
    delay_ms: int = 250,
    timeout_ms: int = 30_000,
    width: int = 1_600,
    height: int = 900,
    device_scale_factor: float = 1.0,
    fit_selector: str | None = None,
    fit_page: bool = False,
    fit_padding: int = 24,
    max_width: int = _DEFAULT_MAX_DIMENSION,
    max_height: int = _DEFAULT_MAX_DIMENSION,
    full_page: bool = False,
    browser_path: Path | None = None,
) -> Path:
    if fit_selector is not None and fit_page:
        raise ValueError("--fit-selector and --fit-page are mutually exclusive")

    hover_selectors = hover or []
    click_selectors = click or []
    source_url = _resolve_source_url(source)

    _require_non_negative_int(delay_ms, name="delay_ms")
    _require_positive_int(timeout_ms, name="timeout_ms")
    _require_positive_int(width, name="width")
    _require_positive_int(height, name="height")
    _require_positive_float(device_scale_factor, name="device_scale_factor")
    _require_non_negative_int(fit_padding, name="fit_padding")
    _require_positive_int(max_width, name="max_width")
    _require_positive_int(max_height, name="max_height")

    playwright = _require_playwright()
    output.parent.mkdir(parents=True, exist_ok=True)

    error_type = getattr(playwright, "Error", None)

    try:
        with playwright.sync_playwright() as api:
            launch_kwargs: dict[str, object] = {"headless": True}
            if browser_path is not None:
                launch_kwargs["executable_path"] = str(browser_path)
            browser = api.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=device_scale_factor,
                )
                page.goto(source_url, wait_until="load", timeout=timeout_ms)

                if wait_for is not None:
                    _first_locator(page, wait_for).wait_for(state="visible", timeout=timeout_ms)
                _wait(page, delay_ms)

                _fit_viewport(
                    page,
                    fit_selector=fit_selector,
                    fit_page=fit_page,
                    timeout_ms=timeout_ms,
                    fit_padding=fit_padding,
                    max_width=max_width,
                    max_height=max_height,
                )
                if fit_selector is not None or fit_page:
                    _wait(page, delay_ms)

                for hover_selector in hover_selectors:
                    _first_locator(page, hover_selector).hover(timeout=timeout_ms)
                    _wait(page, delay_ms)

                for click_selector in click_selectors:
                    _first_locator(page, click_selector).click(timeout=timeout_ms)
                    _wait(page, delay_ms)

                if hover_selectors or click_selectors:
                    _fit_viewport(
                        page,
                        fit_selector=fit_selector,
                        fit_page=fit_page,
                        timeout_ms=timeout_ms,
                        fit_padding=fit_padding,
                        max_width=max_width,
                        max_height=max_height,
                    )
                    if fit_selector is not None or fit_page:
                        _wait(page, delay_ms)

                if selector is not None:
                    _first_locator(page, selector).wait_for(state="visible", timeout=timeout_ms)
                    _first_locator(page, selector).screenshot(
                        path=str(output),
                        animations="disabled",
                    )
                else:
                    page.screenshot(
                        path=str(output),
                        full_page=full_page,
                        animations="disabled",
                    )
            finally:
                browser.close()
    except Exception as exc:
        if error_type is not None and isinstance(exc, error_type):
            raise RuntimeError(f"Playwright failed to capture {source_url}: {exc}") from exc
        raise

    return output


def main() -> int:
    args = _parse_args()
    output = capture_html_screenshot(
        args.source,
        output=args.output,
        selector=args.selector,
        wait_for=args.wait_for,
        hover=args.hover,
        click=args.click,
        delay_ms=args.delay_ms,
        timeout_ms=args.timeout_ms,
        width=args.width,
        height=args.height,
        device_scale_factor=args.device_scale_factor,
        fit_selector=args.fit_selector,
        fit_page=args.fit_page,
        fit_padding=args.fit_padding,
        max_width=args.max_width,
        max_height=args.max_height,
        full_page=args.full_page,
        browser_path=args.browser_path,
    )
    print(f"Saved screenshot: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
