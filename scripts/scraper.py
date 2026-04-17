"""
scraper.py – Scrapes forpsd.com to collect every download URL.

Key insight from page analysis:
  • Each listing card on /?page=N already contains BOTH:
      - /product_detail/eyJ…   (product page)
      - /download/eyJ…         (direct download trigger)
  • We collect /download/eyJ… links directly — no need to visit
    product detail pages one by one.
  • /download/eyJ… redirects → Google Drive share URL.
"""

import re
import time
import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import BASE_URL

log = logging.getLogger(__name__)


class ForPSDScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": BASE_URL,
    }

    def __init__(self, cookie_string: str):
        self._session = self._build_session(cookie_string)

    # ── Session setup ──────────────────────────────────────────────────────
    def _build_session(self, cookie_string: str) -> requests.Session:
        s = requests.Session()
        s.headers.update(self.HEADERS)
        # Parse "name=value; name2=value2; ..." cookie string
        for part in cookie_string.split(";"):
            part = part.strip()
            if "=" in part:
                name, _, value = part.partition("=")
                s.cookies.set(name.strip(), value.strip(), domain="forpsd.com")
        return s

    # ── Page fetching ──────────────────────────────────────────────────────
    def _get(self, url: str) -> BeautifulSoup | None:
        try:
            r = self._session.get(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            return BeautifulSoup(r.content, "html.parser")
        except Exception as exc:
            log.error(f"GET failed {url}: {exc}")
            return None

    # ── Collect all download links ─────────────────────────────────────────
    def get_all_download_urls(self, page_limit: int = 0) -> list[str]:
        """
        Return list of /download/eyJ… absolute URLs for every item
        across all listing pages.

        Args:
            page_limit: Maximum number of pages to scrape.
                        0 (default) = no limit — scrape until the site
                        runs out of pages.
                        Any positive integer stops after that many pages,
                        e.g. page_limit=10  → pages 1–10 only.
        """
        download_urls: list[str] = []
        page = 1

        limit_msg = f"(limit: {page_limit} pages)" if page_limit > 0 else "(no page limit)"
        log.info(f"Starting scrape {limit_msg}")

        while True:

            # ── Page-limit check ───────────────────────────────────────────
            if page_limit > 0 and page > page_limit:
                log.info(
                    f"Reached PAGE_LIMIT={page_limit}. "
                    f"Stopping after {page - 1} pages "
                    f"({len(download_urls)} links collected)."
                )
                break

            url = f"{BASE_URL}/?page={page}"
            log.info(f"Scraping listing page {page}"
                     + (f"/{page_limit}" if page_limit > 0 else "")
                     + f": {url}")
            soup = self._get(url)
            if soup is None:
                break

            # ── Collect /download/ links ───────────────────────────────────
            found = soup.find_all("a", href=re.compile(r"/download/"))
            if not found:
                log.info(f"No download links on page {page} – stopping.")
                break

            for tag in found:
                href = tag.get("href", "")
                abs_url = urljoin(BASE_URL, href)
                if abs_url not in download_urls:
                    download_urls.append(abs_url)

            log.info(f"  Page {page}: +{len(found)} links  (total {len(download_urls)})")

            # ── Detect next page ───────────────────────────────────────────
            next_link = soup.find("a", attrs={"rel": "next"})
            if not next_link:
                # Bootstrap pagination: look for a page-(N+1) link
                next_link = soup.find(
                    "a", class_="page-link",
                    href=re.compile(rf"[?&]page={page + 1}")
                )
            if not next_link:
                log.info("No next page found – done scraping.")
                break

            page += 1
            time.sleep(0.3)   # polite pause (no rate limit on forpsd.com)

        log.info(f"Total download URLs collected: {len(download_urls)}")
        return download_urls

    # ── Resolve a /download/eyJ… → Google Drive URL ────────────────────────
    def resolve_drive_url(self, download_url: str) -> str | None:
        """
        Follow the /download/eyJ… link.
        Returns the final Google Drive URL, or None on failure.
        """
        try:
            r = self._session.get(
                download_url,
                timeout=30,
                allow_redirects=True,
            )
            # Case 1: direct redirect to Drive
            if "drive.google.com" in r.url:
                return r.url

            # Case 2: page contains a Drive link
            soup = BeautifulSoup(r.content, "html.parser")
            drive_tag = soup.find("a", href=re.compile(r"drive\.google\.com"))
            if drive_tag:
                return drive_tag["href"]

            # Case 3: meta refresh to Drive
            meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
            if meta:
                content = meta.get("content", "")
                m = re.search(r"url=(https://drive\.google\.com[^\s\"']+)", content, re.I)
                if m:
                    return m.group(1)

            log.warning(f"Could not resolve Drive URL from {download_url} (final: {r.url})")
            return None

        except Exception as exc:
            log.error(f"resolve_drive_url error for {download_url}: {exc}")
            return None
