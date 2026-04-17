"""
scraper.py – Scrapes forpsd.com to collect every download URL + category.

Key insight from page analysis:
  • Each listing card on /?page=N already contains BOTH:
      - /product_detail/eyJ…   (product page)
      - /download/eyJ…         (direct download trigger)
  • We collect /download/eyJ… links directly — no need to visit
    product detail pages one by one.
  • /download/eyJ… redirects → Google Drive share URL.

Category scraping:
  • For each item, we visit the product detail page to get the category.
  • Category is extracted from breadcrumb or category tag on the page.
  • Category is cleaned and used as a subfolder name (e.g. "wedding").
  • If category is not found, falls back to "uncategorized".
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

    # ── Category extraction from product detail page ───────────────────────
    def get_category(self, product_detail_url: str) -> str:
        """
        Visit the product detail page and extract the category.

        Tries multiple strategies in order:
          1. Breadcrumb navigation links (most reliable)
          2. Category badge / tag elements
          3. Meta keywords
          4. URL path segment

        Returns a clean lowercase category string, e.g. "wedding".
        Falls back to "uncategorized" if nothing is found.
        """
        if not product_detail_url:
            return "uncategorized"

        soup = self._get(product_detail_url)
        if soup is None:
            return "uncategorized"

        # Strategy 1: Breadcrumb — look for <nav> or <ol class="breadcrumb">
        breadcrumb = (
            soup.find("ol", class_=re.compile("breadcrumb", re.I))
            or soup.find("nav", attrs={"aria-label": re.compile("breadcrumb", re.I)})
            or soup.find("ul", class_=re.compile("breadcrumb", re.I))
        )
        if breadcrumb:
            crumbs = breadcrumb.find_all("li")
            # Skip first (Home) and last (current page title), take middle ones
            middle_crumbs = crumbs[1:-1] if len(crumbs) > 2 else crumbs[1:]
            for crumb in middle_crumbs:
                text = crumb.get_text(strip=True)
                if text and text.lower() not in ("home", ""):
                    return self._clean_category(text)

        # Strategy 2: Category tag / badge on the page
        for selector in [
            {"class": re.compile(r"categor", re.I)},
            {"class": re.compile(r"tag", re.I)},
            {"class": re.compile(r"label", re.I)},
            {"class": re.compile(r"badge", re.I)},
        ]:
            tag = soup.find("a", attrs=selector)
            if tag:
                text = tag.get_text(strip=True)
                if text and len(text) > 1:
                    return self._clean_category(text)

        # Strategy 3: Meta keywords
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw:
            kw = meta_kw.get("content", "").split(",")
            if kw and kw[0].strip():
                return self._clean_category(kw[0].strip())

        # Strategy 4: Extract from URL path e.g. /category/wedding/product...
        url_match = re.search(r"/category/([^/?#]+)", product_detail_url, re.I)
        if url_match:
            return self._clean_category(url_match.group(1))

        log.warning(f"Could not detect category for: {product_detail_url}")
        return "uncategorized"

    def _clean_category(self, raw: str) -> str:
        """
        Normalise a raw category string into a safe folder name.
        e.g. "Wedding Cards" → "wedding-cards"
             "Business & Corporate" → "business-corporate"
        """
        cleaned = raw.strip().lower()
        cleaned = re.sub(r"[&+/\\|]", "-", cleaned)   # special chars → dash
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)    # remove other specials
        cleaned = re.sub(r"[\s_]+", "-", cleaned)     # spaces/underscores → dash
        cleaned = re.sub(r"-{2,}", "-", cleaned)      # collapse multiple dashes
        cleaned = cleaned.strip("-")
        return cleaned if cleaned else "uncategorized"

    # ── Collect all download links + product detail URLs ──────────────────
    def get_all_items(self, page_limit: int = 0) -> list[dict]:
        """
        Return list of dicts for every item across all listing pages.
        Each dict: {"download_url": str, "detail_url": str}

        Args:
            page_limit: Maximum number of pages to scrape.
                        0 (default) = no limit.
        """
        items: list[dict] = []
        seen_downloads: set[str] = set()
        page = 1

        limit_msg = f"(limit: {page_limit} pages)" if page_limit > 0 else "(no page limit)"
        log.info(f"Starting scrape {limit_msg}")

        while True:
            # ── Page-limit check ───────────────────────────────────────────
            if page_limit > 0 and page > page_limit:
                log.info(
                    f"Reached PAGE_LIMIT={page_limit}. "
                    f"Stopping after {page - 1} pages "
                    f"({len(items)} items collected)."
                )
                break

            url = f"{BASE_URL}/?page={page}"
            log.info(
                f"Scraping listing page {page}"
                + (f"/{page_limit}" if page_limit > 0 else "")
                + f": {url}"
            )
            soup = self._get(url)
            if soup is None:
                break

            # ── Find download + detail links per card ──────────────────────
            download_tags = soup.find_all("a", href=re.compile(r"/download/"))
            if not download_tags:
                log.info(f"No download links on page {page} – stopping.")
                break

            found_this_page = 0
            for tag in download_tags:
                dl_href = tag.get("href", "")
                dl_url  = urljoin(BASE_URL, dl_href)
                if dl_url in seen_downloads:
                    continue
                seen_downloads.add(dl_url)

                # Find the matching product_detail link in the same card
                card = tag.find_parent(
                    lambda t: t.name in ("div", "li", "article")
                    and t.find("a", href=re.compile(r"/product_detail/"))
                )
                detail_url = ""
                if card:
                    detail_tag = card.find("a", href=re.compile(r"/product_detail/"))
                    if detail_tag:
                        detail_url = urljoin(BASE_URL, detail_tag["href"])

                items.append({"download_url": dl_url, "detail_url": detail_url})
                found_this_page += 1

            log.info(f"  Page {page}: +{found_this_page} items  (total {len(items)})")

            # ── Detect next page ───────────────────────────────────────────
            next_link = soup.find("a", attrs={"rel": "next"})
            if not next_link:
                next_link = soup.find(
                    "a", class_="page-link",
                    href=re.compile(rf"[?&]page={page + 1}")
                )
            if not next_link:
                log.info("No next page found – done scraping.")
                break

            page += 1
            time.sleep(0.3)

        log.info(f"Total items collected: {len(items)}")
        return items

    # ── Legacy helper (kept for compatibility) ─────────────────────────────
    def get_all_download_urls(self, page_limit: int = 0) -> list[str]:
        """Return plain list of download URLs (no category info)."""
        return [item["download_url"] for item in self.get_all_items(page_limit)]

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
