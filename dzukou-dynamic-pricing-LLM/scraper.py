#!/usr/bin/env python3
"""Scrape competitor pricing data for each product category."""
import json
import logging
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from .utils import canonical_key

BASE_DIR = Path(__file__).resolve().parent
KEYWORDS_JSON = BASE_DIR / "category_keywords.json"
DATA_DIR = BASE_DIR / "product_data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_filename(text: str) -> str:
    base = re.sub(r"\W+", "_", text.lower()).strip("_")
    return base + ".csv"


DEFAULT_CATEGORIES = {
    "Wooden Sunglasses": {
        "search_terms": [
            "wooden sunglasses",
            "wood sunglasses",
            "sustainable sunglasses",
            "bamboo sunglasses",
            "eco-friendly sunglasses",
            "natural wood eyewear",
        ],
        "csv_filename": "wooden_sunglasses.csv",
    },
    "Thermos Bottles": {
        "search_terms": [
            "thermos bottle",
            "stainless steel bottle",
            "insulated bottle",
            "vacuum insulated bottle",
            "eco water bottle",
            "sustainable water bottle",
        ],
        "csv_filename": "thermos_bottles.csv",
    },
    "Coffee Mugs": {
        "search_terms": [
            "coffee mug",
            "ceramic mug",
            "bamboo mug",
            "eco-friendly mug",
            "sustainable coffee cup",
            "reusable coffee mug",
        ],
        "csv_filename": "coffee_mugs.csv",
    },
    "Lunch Box 1200ML": {
        "search_terms": [
            "1200ml lunch box",
            "lunch box 1200ml",
            "large lunch container",
            "stainless steel lunch box large",
            "eco lunch box large",
            "1.2L food container",
        ],
        "csv_filename": "lunch_box_1200ml.csv",
    },
    "Lunch Box 800ML": {
        "search_terms": [
            "800ml lunch box",
            "lunch box 800ml",
            "medium lunch container",
            "stainless steel lunch box medium",
            "eco lunch box medium",
            "800ml food container",
        ],
        "csv_filename": "lunch_box_800ml.csv",
    },
    "Silk Colored Stole": {
        "search_terms": [
            "silk colored stole",
            "colored silk stole",
            "silk stole",
            "colorful silk stole",
            "silk scarf colored",
            "vibrant silk stole",
            "multicolor silk wrap",
            "printed silk stole",
            "silk wrap colored",
            "handmade silk stole",
            "artisan silk scarf",
            "fair trade silk stole",
        ],
        "csv_filename": "silk_colored_stole.csv",
    },
    "White Silk Stole": {
        "search_terms": [
            "white silk stole",
            "white stole",
            "silk white stole",
            "pure white silk stole",
            "white silk scarf",
            "white silk wrap",
            "ivory silk stole",
            "cream silk stole",
            "white silk shawl",
            "natural white silk stole",
            "organic white silk",
            "handmade white silk",
        ],
        "csv_filename": "white_silk_stole.csv",
    },
    "Phone Stand": {
        "search_terms": [
            "phone stand",
            "wooden phone stand",
            "bamboo phone holder",
            "eco-friendly phone stand",
            "sustainable phone holder",
            "natural wood phone dock",
            "desk phone stand",
            "mobile phone holder",
        ],
        "csv_filename": "phone_stand.csv",
    },
    "Notebooks": {
        "search_terms": [
            "eco notebook",
            "sustainable notebook",
            "recycled paper journal",
            "bamboo notebook",
            "eco-friendly journal",
            "handmade paper notebook",
            "hemp paper notebook",
            "tree-free journal",
        ],
        "csv_filename": "notebooks.csv",
    },
}


class ProductScraper:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.user_agents = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
                "Gecko/20100101 Firefox/121.0"
            ),
        ]
        self.session.headers.update(
            {
                "User-Agent": random.choice(self.user_agents),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument(
            f"--user-agent={random.choice(self.user_agents)}")

        self.stores = {
            "Made Trade": {
                "url": "https://www.madetrade.com",
                "search_pattern": "/search?q={}"},
            "EarthHero": {
                "url": "https://earthhero.com",
                "search_pattern": "/search?q={}"},
            "Package Free Shop": {
                "url": "https://packagefreeshop.com",
                "search_pattern": "/search?q={}"},
            "Ten Thousand Villages": {
                "url": "https://www.tenthousandvillages.com",
                "search_pattern": "/search?q={}"},
            "Zero Waste Store": {
                "url": "https://zerowastestoreonline.com",
                "search_pattern": "/search?q={}"},
        }

        self.product_categories = self.load_categories()

        self.csv_dir = DATA_DIR
        self.csv_dir.mkdir(exist_ok=True)
        self.results_by_category = {
            cat: [] for cat in self.product_categories
        }

    def load_categories(self):
        """Return merged categories from defaults and keywords."""
        merged = {}

        # start with defaults using canonical keys
        for cat, info in DEFAULT_CATEGORIES.items():
            merged[canonical_key(cat)] = {
                "name": cat,
                "search_terms": list(info.get("search_terms", [])),
                "csv_filename": info.get(
                    "csv_filename", sanitize_filename(cat)
                ),
            }

        if KEYWORDS_JSON.exists():
            try:
                with open(KEYWORDS_JSON, "r", encoding="utf-8") as f:
                    kw_data = json.load(f)
                for cat, kws in kw_data.items():
                    key = canonical_key(cat)
                    if key in merged:
                        merged[key]["search_terms"] = list(
                            set(merged[key]["search_terms"] + list(kws))
                        )
                    else:
                        merged[key] = {
                            "name": cat.strip(),
                            "search_terms": list(kws),
                            "csv_filename": sanitize_filename(cat),
                        }
            except Exception as exc:
                logger.warning("Failed loading keywords: %s", exc)

        return {
            info["name"]: {
                "search_terms": info["search_terms"],
                "csv_filename": info["csv_filename"],
            }
            for info in merged.values()
        }

    def clean_price(self, price_text):
        if not price_text:
            return None
        m = re.search(r"(\$|€|£|Rs\.?|USD)?\s*(\d+[\.,]\d+|\d+)", price_text)
        if m:
            val = m.group(2).replace(",", ".")
            try:
                return float(val)
            except ValueError:
                return None
        return None

    def clean_product_name(self, name):
        if not name:
            return None
        name = name.strip()
        if len(name) < 3:
            return None
        if not re.search(r"[a-zA-Z]", name):
            return None
        return name

    def extract_products_from_html(self, html, search_term):
        products = []
        soup = BeautifulSoup(html, "html.parser")
        price_elems = soup.find_all(string=re.compile(r"\$\s*\d+"))
        for price_elem in price_elems:
            parent = price_elem.parent
            for _ in range(3):
                if parent and parent.parent:
                    parent = parent.parent
            if not parent:
                continue
            name_elems = parent.find_all(
                ["h1", "h2", "h3", "h4", "a", "span", "div"])
            product_name = None
            for elem in name_elems:
                text = elem.get_text(strip=True)
                if len(text) > 3 and "$" not in text:
                    product_name = self.clean_product_name(text)
                    if product_name:
                        break
            if not product_name:
                continue
            price = self.clean_price(price_elem)
            if product_name and price and price > 0:
                products.append(
                    {
                        "name": product_name,
                        "price": price,
                        "search_term": search_term,
                    }
                )
        return products

    def scrape_with_requests(self, store_name, store_cfg, term):
        products = []
        try:
            url = store_cfg["url"] + \
                store_cfg["search_pattern"].format(quote(term))
            logger.info("Requesting: %s", url)
            self.session.headers["User-Agent"] = random.choice(
                self.user_agents)
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                products = self.extract_products_from_html(resp.content, term)
                logger.info("Found %d products via requests", len(products))
        except Exception as exc:
            logger.error("Error with requests: %s", exc)
        return products

    def scrape_with_selenium(self, store_name, store_cfg, term):
        products = []
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            url = store_cfg["url"] + \
                store_cfg["search_pattern"].format(quote(term))
            logger.info("Selenium visiting: %s", url)
            driver.get(url)
            WebDriverWait(
                driver, 10).until(
                EC.presence_of_element_located(
                    (By.TAG_NAME, "body")))
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            elems = driver.find_elements(
                By.XPATH, "//*[contains(text(), '$')]")
            for elem in elems[:30]:
                parent = elem
                for _ in range(3):
                    if parent:
                        try:
                            parent = parent.find_element(By.XPATH, "..")
                        except Exception:
                            break
                if not parent:
                    continue
                name_elems = parent.find_elements(
                    By.CSS_SELECTOR,
                    "h1, h2, h3, h4, a, .title, .name, span",
                )
                product_name = None
                for nm in name_elems:
                    text = nm.text.strip()
                    if len(text) > 3 and "$" not in text:
                        product_name = self.clean_product_name(text)
                        if product_name:
                            break
                if not product_name:
                    continue
                price = self.clean_price(elem.text)
                if product_name and price and price > 0:
                    products.append(
                        {
                            "name": product_name,
                            "price": price,
                            "search_term": term,
                        }
                    )
        except Exception as exc:
            logger.error("Error with Selenium: %s", exc)
        finally:
            if driver:
                driver.quit()
        logger.info("Found %d products via Selenium", len(products))
        return products

    def scrape_store(self, store_name, store_cfg, category, terms):
        all_products = []
        for term in terms:
            logger.info("Searching for '%s' in %s", term, store_name)
            products = self.scrape_with_requests(store_name, store_cfg, term)
            if not products:
                products = self.scrape_with_selenium(
                    store_name, store_cfg, term)
            if products:
                all_products.extend(products)
            time.sleep(random.uniform(1, 2))
        unique = []
        seen = set()
        for p in all_products:
            key = (p["name"].lower()[:20], round(p["price"], 0))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def scrape_all_stores(self):
        logger.info("Starting product scraping...")
        for category, info in self.product_categories.items():
            terms = info["search_terms"]
            if not terms:
                logger.warning(
                    "No search terms for category '%s', skipping", category
                )
                continue
            for store_name, store_cfg in self.stores.items():
                logger.info("Checking %s...", store_name)
                try:
                    products = self.scrape_store(
                        store_name, store_cfg, category, terms)
                    for product in products:
                        self.results_by_category[category].append(
                            {
                                "category": category,
                                "store": store_name,
                                "product_name": product["name"],
                                "price": product["price"],
                                "search_term": product["search_term"],
                                "store_url": store_cfg["url"],
                            }
                        )
                except requests.exceptions.RequestException as exc:
                    logger.error("Network error with %s: %s", store_name, exc)
                except Exception as exc:
                    logger.error(
                        "Error scraping %s for %s: %s",
                        store_name,
                        category,
                        exc)
                time.sleep(random.uniform(1.5, 3.0))

    def save_category_csvs(self):
        saved = []
        for category, products in self.results_by_category.items():
            csv_name = self.product_categories[category]["csv_filename"]
            path = self.csv_dir / csv_name
            if products:
                df = pd.DataFrame(products)
                df.to_csv(path, index=False)
                logger.info("Saved %d products to %s", len(df), str(path))
            else:
                pd.DataFrame(
                    columns=[
                        "category",
                        "store",
                        "product_name",
                        "price",
                        "search_term",
                        "store_url"]).to_csv(
                    path,
                    index=False)
                logger.info("Created empty file: %s", str(path))
            saved.append(path)
        print("\nCategory files created:")
        for p in saved:
            count = len(pd.read_csv(p)) if p.exists() else 0
            print(f"  • {p.name}: {count} products")
        print(f"\nAll files saved to: {str(self.csv_dir)}/ directory")


def main():
    scraper = ProductScraper()
    print(
        "Starting Category-Specific Product Scraper\n",
        (
            f"Searching {len(scraper.product_categories)} product "
            f"categories across {len(scraper.stores)} stores\n"
        ),
    )
    for cat, info in scraper.product_categories.items():
        print(f"  • {cat}: {info['csv_filename']}")
    print()
    scraper.scrape_all_stores()
    scraper.save_category_csvs()


if __name__ == "__main__":
    main()
