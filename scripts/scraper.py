"""
scraper.py – Scrapes forpsd.com to collect every download URL + category.

Category detection strategy:
  • The product detail page contains a <p> tag like:
      "File :6362 - Dr Ambedkar - flex - banner"
  • We extract that title/description and run keyword matching to determine
    the exact subfolder from the full category list.
  • Falls back to listing-card title text if detail page fetch fails.
  • Final fallback: "uncategorized".

Folder names produced (examples):
  ambedkar, wedding-flex, wedding-invitation, birthday-flex,
  birthday-invitation, tvk, dmk-calendar, earpiercing-flex, ...
"""

import re
import time
import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import BASE_URL

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY RULES — ordered MOST SPECIFIC → LEAST SPECIFIC.
#
#  Each entry: ( [keywords_that_must_all_appear], "folder-name" )
#
#  The post title is lowercased and each keyword is checked as a substring.
#  First matching rule wins.
# ═══════════════════════════════════════════════════════════════════════════

CATEGORY_RULES: list[tuple[list[str], str]] = [

    # ── CALENDAR subtypes (before party names so "dmk calendar" ≠ "dmk") ──
    (["god", "calendar"],                   "god-calendar"),
    (["admk", "calendar"],                  "admk-calendar"),
    (["aiadmk", "calendar"],                "admk-calendar"),
    (["dmk", "calendar"],                   "dmk-calendar"),
    (["tvk", "calendar"],                   "tvk-calendar"),
    (["pmk", "calendar"],                   "pmk-calendar"),
    (["vck", "calendar"],                   "vck-calendar"),
    (["calendar"],                          "calendar"),

    # ── INVITATION (before flex/banner to avoid "wedding invitation" → flex) ─
    (["wedding", "invitation"],             "wedding-invitation"),
    (["ear", "piercing", "invitation"],     "earpiercing-invitation"),
    (["earpiercing", "invitation"],         "earpiercing-invitation"),
    (["karnavedham", "invitation"],         "earpiercing-invitation"),
    (["puberty", "invitation"],             "puberty-invitation"),
    (["manjal", "neerattu", "invitation"],  "puberty-invitation"),
    (["birthday", "invitation"],            "birthday-invitation"),
    (["house", "warming", "invitation"],    "houswarming-invitation"),
    (["housewarming", "invitation"],        "houswarming-invitation"),
    (["gruha", "pravesam", "invitation"],   "houswarming-invitation"),
    (["baby", "shower"],                    "baby-shower-invitation"),
    (["general", "invitation"],             "general-invitation"),
    (["invitation"],                        "general-invitation"),

    # ── FRAME subtypes (before generic "frame") ───────────────────────────
    (["wedding", "frame"],                  "wedding-frame"),
    (["birthday", "frame"],                 "birthday-frame"),
    (["death", "frame"],                    "death-frame"),
    (["collage", "frame"],                  "collage-frame"),
    (["ai", "background"],                  "ai-background"),
    (["gift", "shield"],                    "gift-shield"),
    (["frame"],                             "frame"),

    # ── ALBUM subtypes ────────────────────────────────────────────────────
    (["wedding", "album"],                  "wedding-album"),
    (["birthday", "album"],                 "birthday-album"),
    (["floral", "album"],                   "floral-album"),
    (["vertical", "album"],                 "vertical-album"),
    (["album"],                             "wedding-album"),

    # ── SHOP ──────────────────────────────────────────────────────────────
    (["grand", "opening"],                  "grand-opening"),
    (["shop"],                              "shop"),

    # ── NOTICE / VISITING CARD / CERTIFICATE ──────────────────────────────
    (["temple", "notice"],                  "temple-notice"),
    (["visiting", "card"],                  "visiting-card"),
    (["certificate"],                       "certificate"),
    (["notice"],                            "notice"),

    # ── FLYERS subtypes (before generic christian/birthday) ───────────────
    (["bike", "flyer"],                     "bike-flyers"),
    (["bike", "flex"],                      "bike-flyers"),
    (["birthday", "flyer"],                 "birthday-flyers"),
    (["christian", "flyer"],                "christian-flyers"),
    (["food", "flyer"],                     "food-flyers"),
    (["flyer"],                             "flyers"),
    (["flyers"],                            "flyers"),

    # ── FESTIVALS ─────────────────────────────────────────────────────────
    (["pongal"],                            "pongal"),
    (["ramzan"],                            "ramzan"),
    (["ramadan"],                           "ramzan"),
    (["new year"],                          "new-year"),
    (["new", "year"],                       "new-year"),

    # ── GODS subtypes ─────────────────────────────────────────────────────
    (["amman"],                             "amman"),
    (["murugan"],                           "murugan"),
    (["ganesha"],                           "ganesha"),
    (["ganesh"],                            "ganesha"),
    (["vinayagar"],                         "ganesha"),
    (["ayyappan"],                          "ayyappan"),
    (["sasta"],                             "ayyappan"),
    (["muslim"],                            "muslim"),
    (["islamic"],                           "muslim"),
    (["christmas"],                         "christian"),
    (["christian"],                         "christian"),
    (["god"],                               "gods"),
    (["goddess"],                           "gods"),

    # ── ACTOR & ACTRESS (specific names before generic) ───────────────────
    (["vijay"],                             "vijay"),
    (["ajith"],                             "ajith"),
    (["thala"],                             "ajith"),
    (["rajini"],                            "rajini"),
    (["rajinikanth"],                       "rajini"),
    (["thalaivar"],                         "rajini"),
    (["actor"],                             "actor"),
    (["actress"],                           "actor"),

    # ── POLITICAL — specific leaders first (before party name) ───────────
    (["ambedkar"],                          "ambedkar"),
    (["periyar"],                           "ambedkar"),
    (["edappadi", "palanisamy"],            "edappadi-palanisamy"),
    (["edappadi"],                          "edappadi-palanisamy"),
    (["jayalalitha"],                       "jayalalitha"),
    (["jayalalithaa"],                      "jayalalitha"),
    (["amma"],                              "jayalalitha"),
    (["mgr"],                               "mgr"),
    (["kalaignar"],                         "kalaignar"),
    (["udhayanidhi"],                       "udhayanidhi-stalin"),
    (["mk", "stalin"],                      "mk-stalin"),
    (["m.k", "stalin"],                     "mk-stalin"),
    (["stalin"],                            "mk-stalin"),
    (["puratchi", "bharatha"],              "puratchi-bharatha-katchi"),
    (["pbk"],                               "puratchi-bharatha-katchi"),
    # Parties
    (["tvk"],                               "tvk"),
    (["admk"],                              "admk"),
    (["aiadmk"],                            "admk"),
    (["dmk"],                               "dmk"),
    (["pmk"],                               "pmk"),
    (["vck"],                               "vck"),
    (["ntk"],                               "ntk"),
    (["bjp"],                               "bjp"),
    (["congress"],                          "congress"),
    (["dmdk"],                              "dmdk"),
    (["ammk"],                              "ammk"),

    # ── OTHERS ────────────────────────────────────────────────────────────
    (["caricature"],                        "caricature"),
    (["3d", "text"],                        "3d-text"),
    (["font"],                              "fonts"),
    (["extras"],                            "extras"),
    (["name", "board"],                     "name-board"),
    (["nameboard"],                         "name-board"),
    (["cinematic"],                         "cinematic"),
    (["madurai"],                           "madurai-flex"),
    (["title"],                             "title-design"),

    # ── FLEX / BANNER subtypes (checked after invitation to avoid collision) ─
    (["wedding", "flex"],                   "wedding-flex"),
    (["wedding", "banner"],                 "wedding-flex"),
    (["wedding", "poster"],                 "wedding-flex"),
    (["ear", "piercing", "flex"],           "earpiercing-flex"),
    (["earpiercing", "flex"],               "earpiercing-flex"),
    (["ear", "piercing", "banner"],         "earpiercing-flex"),
    (["karnavedham", "flex"],               "earpiercing-flex"),
    (["karnavedham", "banner"],             "earpiercing-flex"),
    (["puberty", "flex"],                   "puberty-flex"),
    (["puberty", "banner"],                 "puberty-flex"),
    (["manjal", "neerattu"],                "puberty-flex"),
    (["birthday", "flex"],                  "birthday-flex"),
    (["birthday", "banner"],                "birthday-flex"),
    (["first", "birthday"],                 "birthday-flex"),
    (["1st", "birthday"],                   "birthday-flex"),
    (["death", "flex"],                     "death-flex"),
    (["death", "banner"],                   "death-flex"),
    (["memorial", "flex"],                  "memorial-flex"),
    (["memorial", "banner"],                "memorial-flex"),
    (["house", "warming", "flex"],          "houswarming-flex"),
    (["housewarming", "flex"],              "houswarming-flex"),
    (["house", "warming", "banner"],        "houswarming-flex"),
    (["gruha", "pravesam"],                 "houswarming-flex"),
    (["decoration", "flex"],                "decoration-flex"),
    (["decoration", "banner"],              "decoration-flex"),
    (["temple", "flex"],                    "temple-flex"),
    (["temple", "banner"],                  "temple-flex"),
    (["temple"],                            "temple-flex"),
    # Generic catch-all flex (must be LAST)
    (["flex"],                              "flex"),
    (["banner"],                            "flex"),
    (["poster"],                            "flex"),
]


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

    # ════════════════════════════════════════════════════════════════════════
    #  Category detection from post title
    # ════════════════════════════════════════════════════════════════════════

    def _detect_category_from_title(self, title: str) -> str:
        if not title:
            return "uncategorized"

        low = title.lower()

        for keywords, folder in CATEGORY_RULES:
            if all(kw in low for kw in keywords):
                log.debug(f"    Rule {keywords!r} → '{folder}'")
                return folder

        log.warning(f"  No category rule matched for title: {title!r}")
        return "uncategorized"

    def _extract_post_title(self, soup: BeautifulSoup) -> str:
        # Strategy 1: paragraph with "File :" prefix
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if re.search(r"file\s*:", text, re.I):
                cleaned = re.sub(r"(?i)^\s*file\s*:\s*\d*\s*", "", text).strip(" -–")
                cleaned = re.sub(
                    r"(?i)this\s+file\s+not\s+suitable[^-–]*[-–]?\s*", "", cleaned
                ).strip(" -–")
                if cleaned:
                    log.debug(f"  Title from <p>: {cleaned!r}")
                    return cleaned

        # Strategy 2: headings
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 3:
                log.debug(f"  Title from <{tag.name}>: {text!r}")
                return text

        # Strategy 3: <title> tag
        title_tag = soup.find("title")
        if title_tag:
            text = title_tag.get_text(strip=True)
            text = re.sub(r"\s*[|–-]+\s*(forpsd|ramarts).*$", "", text, flags=re.I).strip()
            if text:
                log.debug(f"  Title from <title>: {text!r}")
                return text

        return ""

    # ── Public: resolve category from product detail page ─────────────────
    def get_category(self, product_detail_url: str, hint_title: str = "") -> str:
        if hint_title:
            category = self._detect_category_from_title(hint_title)
            if category != "uncategorized":
                log.info(f"  Category (from card title): [{category}]  title={hint_title!r}")
                return category

        if not product_detail_url:
            return "uncategorized"

        soup = self._get(product_detail_url)
        if soup is None:
            return "uncategorized"

        title = self._extract_post_title(soup)
        if not title:
            log.warning(f"  Could not extract title from: {product_detail_url}")
            return "uncategorized"

        log.info(f"  Post title: {title!r}")
        category = self._detect_category_from_title(title)
        log.info(f"  → Category: [{category}]")
        return category

    # ── FIX: Re-scrape a detail page to get a fresh /download/ URL ─────────
    def get_fresh_download_url(self, detail_url: str) -> str | None:
        """
        Re-scrape a product detail page to obtain a fresh /download/ URL.

        Called when the stored download_url has expired (JWT token timeout)
        and resolve_drive_url returns None. Returns the new absolute download
        URL, or None if it cannot be found.
        """
        if not detail_url:
            return None

        log.info(f"  🔄 Re-scraping detail page for fresh download URL: {detail_url}")
        soup = self._get(detail_url)
        if soup is None:
            log.warning(f"  Could not fetch detail page: {detail_url}")
            return None

        tag = soup.find("a", href=re.compile(r"/download/"))
        if tag:
            fresh_url = urljoin(BASE_URL, tag["href"])
            log.info(f"  ✅ Fresh download URL: {fresh_url}")
            return fresh_url

        log.warning(f"  No /download/ link found on detail page: {detail_url}")
        return None

    # ── Collect all download links + product detail URLs ──────────────────
    def get_all_items(
        self,
        page_limit: int = 0,
        stop_at_known_urls: "set | None" = None,
        start_page: int = 1,
        max_new_items: int = 0,
    ) -> list[dict]:
        """
        Return list of dicts for items across listing pages.

        Each dict:
          {
            "download_url": str,
            "detail_url":   str,
            "card_title":   str,   <- title from listing card (best-effort)
          }

        start_page (int):
          Start scraping from this page number instead of page 1.

        max_new_items (int):
          Stop once this many new items are collected (0 = no limit).

        stop_at_known_urls (set):
          When provided, URLs in this set are skipped (not counted as new).
          In incremental mode this also triggers early-stop when an entire
          page yields zero new URLs.
        """
        items: list[dict] = []
        seen_downloads: set[str] = set()
        page = max(start_page, 1)

        incremental = stop_at_known_urls is not None
        limit_msg   = f"(limit: {page_limit} pages)" if page_limit > 0 else "(no page limit)"
        mode_msg    = " [incremental — stops at known territory]" if incremental else ""
        start_msg   = f" [start_page={page}]" if page > 1 else ""
        items_msg   = f" [max_new_items={max_new_items}]" if max_new_items > 0 else ""
        log.info(f"Starting scrape {limit_msg}{mode_msg}{start_msg}{items_msg}")

        while True:
            if page_limit > 0 and page > page_limit:
                log.info(
                    f"Reached PAGE_LIMIT={page_limit}. "
                    f"Stopping after {page - 1} pages "
                    f"({len(items)} new items collected)."
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

            download_tags = soup.find_all("a", href=re.compile(r"/download/"))
            if not download_tags:
                log.info(f"No download links on page {page} – stopping.")
                break

            new_this_page = 0
            for tag in download_tags:
                dl_href = tag.get("href", "")
                dl_url  = urljoin(BASE_URL, dl_href)
                if dl_url in seen_downloads:
                    continue
                seen_downloads.add(dl_url)

                if incremental and dl_url in stop_at_known_urls:
                    continue

                card = tag.find_parent(
                    lambda t: t.name in ("div", "li", "article")
                    and t.find("a", href=re.compile(r"/product_detail/"))
                )
                detail_url = ""
                card_title = ""
                if card:
                    detail_tag = card.find("a", href=re.compile(r"/product_detail/"))
                    if detail_tag:
                        detail_url = urljoin(BASE_URL, detail_tag["href"])
                    card_title = self._extract_card_title(card)

                items.append({
                    "download_url": dl_url,
                    "detail_url":   detail_url,
                    "card_title":   card_title,
                })
                new_this_page += 1

            log.info(f"  Page {page}: +{new_this_page} new items  (running total {len(items)})")

            if max_new_items > 0 and len(items) >= max_new_items:
                log.info(
                    f"  Reached max_new_items={max_new_items} — "
                    "stopping targeted scrape early."
                )
                break

            if incremental and new_this_page == 0:
                log.info(
                    f"  Page {page} had no new URLs — "
                    "stopping incremental scrape early."
                )
                break

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

        log.info(f"Total new items collected: {len(items)}")
        return items


    def _extract_card_title(self, card) -> str:
        for htag in card.find_all(["h1", "h2", "h3", "h4", "h5"]):
            text = htag.get_text(strip=True)
            if text and len(text) > 3:
                return text

        img = card.find("img", alt=True)
        alt = img.get("alt", "").strip() if img else ""
        # Skip generic/useless alt values that forpsd.com uses for thumbnails
        _GENERIC_ALTS = {"image preview", "preview", "image", "thumbnail", "photo", "img"}
        if alt and alt.lower() not in _GENERIC_ALTS:
            return alt

        for p in card.find_all("p"):
            text = p.get_text(strip=True)
            if text and 3 < len(text) < 120 and not re.search(r"file\s*:", text, re.I):
                return text

        return ""

    # ── Legacy helper (kept for compatibility) ─────────────────────────────
    def get_all_download_urls(self, page_limit: int = 0) -> list[str]:
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
            if "drive.google.com" in r.url:
                return r.url

            soup = BeautifulSoup(r.content, "html.parser")
            drive_tag = soup.find("a", href=re.compile(r"drive\.google\.com"))
            if drive_tag:
                return drive_tag["href"]

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
