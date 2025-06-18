#!/usr/bin/env python3
"""Price optimizer for Dzukou products."""

import csv
import os
import re
import statistics
import json
import random
from scipy import stats
from pathlib import Path
from typing import Dict, List

try:
    from skopt import gp_minimize
    from skopt.space import Real
except Exception:  # library may not be installed
    gp_minimize = None
    Real = None

BASE_DIR = Path(__file__).resolve().parent

PRICE_STEP = 0.25  # granularity for optimizer

# Cap ratio relative to the mean competitor price. This is calculated
# dynamically from competitor price dispersion so recommendations stay
# competitive in diverse markets.
def compute_mean_cap(avg: float, stdev: float, max_cap: float = 1.3) -> float:
    """Return a limit for how far above the mean a price may go."""
    if avg <= 0:
        return max_cap
    # scale with coefficient of variation so markets with high dispersion allow
    # slightly higher pricing while still capping extreme values
    cv = stdev / avg
    return min(1.0 + cv * 0.5, max_cap)


def clean_prices(prices: List[float]) -> List[float]:
    """Remove outliers and invalid data from scraped prices."""
    valid = [p for p in prices if 0 < p < 10000]
    if len(valid) >= 4:
        # compute quartiles separately for clarity
        q1 = statistics.quantiles(valid, n=4)[0]
        q3 = statistics.quantiles(valid, n=4)[2]
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        valid = [p for p in valid if low <= p <= high]
    return valid


def optimize_price(
    prices: List[float],
    current_price: float,
    unit_cost: float,
    margin: float,
    elasticity: float = 1.2,
    max_markup: float = 1.8,
    max_increase: float = 0.30,
    max_decrease: float = 0.25,
    price_step: float = PRICE_STEP,
    demand_base: float = 100.0,
    saturation: float = 1.0,
    mean_cap_ratio: float | None = None,
) -> float:
    """Search for a price that maximizes estimated profit.

    The optimizer explores prices above and below the current price while
    respecting limits on how far the new price may deviate from the
    competitor average and the current price. ``mean_cap_ratio`` controls the
    maximum allowed ratio relative to the competitor mean; if ``None`` it is
    computed from the price dispersion to adapt dynamically. The function
    returns the price yielding the highest simulated profit within these
    boundaries.
    """
    base = unit_cost * (1 + margin)

    if not prices:
        avg = current_price
        competitor_high = current_price
        competitor_low = current_price
    else:
        avg = statistics.mean(prices)
        competitor_high = max(prices)
        competitor_low = min(prices)
    stdev = statistics.stdev(prices) if len(prices) > 1 else 0.0
    if mean_cap_ratio is None:
        mean_cap_ratio = compute_mean_cap(avg, stdev)

    # explore from the minimal acceptable price up to a generous markup,
    # while respecting caps relative to the current price and competitor mean
    low = max(base, competitor_low * 0.9, current_price * (1 - max_decrease))
    high = max(current_price, competitor_high, avg, base) * max_markup
    high = min(high, current_price * (1 + max_increase), avg * mean_cap_ratio)
    if high < low:
        high = low

    best_price = base
    best_profit = -1e9

    # grid search with adaptive resolution so very wide ranges don't take too
    # long. the step is adjusted if more than 100 steps would be required.
    steps = int((high - low) / price_step) + 1
    if steps > 100:
        price_step = (high - low) / 100

    price = low
    while price <= high:
        profit = simulate_profit(
            price,
            unit_cost,
            avg,
            demand_base,
            elasticity,
            saturation,
        )
        if profit > best_profit:
            best_profit = profit
            best_price = price
        price += price_step

    best_price = max(best_price, base)
    return round_price(best_price)


def bayesian_optimize_price(
    prices: List[float],
    current_price: float,
    unit_cost: float,
    margin: float,
    elasticity: float = 1.2,
    max_markup: float = 1.8,
    max_increase: float = 0.30,
    max_decrease: float = 0.25,
    demand_base: float = 100.0,
    saturation: float = 1.0,
    mean_cap_ratio: float | None = None,
    n_calls: int = 25,
) -> float:
    """Use Bayesian optimization to maximize profit."""
    if gp_minimize is None:
        # fall back to grid search when skopt is unavailable
        return optimize_price(
            prices,
            current_price,
            unit_cost,
            margin,
            elasticity=elasticity,
            max_markup=max_markup,
            max_increase=max_increase,
            max_decrease=max_decrease,
            demand_base=demand_base,
            saturation=saturation,
            mean_cap_ratio=mean_cap_ratio,
        )

    base = unit_cost * (1 + margin)
    if not prices:
        avg = current_price
        competitor_high = current_price
        competitor_low = current_price
    else:
        avg = statistics.mean(prices)
        competitor_high = max(prices)
        competitor_low = min(prices)
    stdev = statistics.stdev(prices) if len(prices) > 1 else 0.0
    if mean_cap_ratio is None:
        mean_cap_ratio = compute_mean_cap(avg, stdev)

    low = max(base, competitor_low * 0.9, current_price * (1 - max_decrease))
    high = max(current_price, competitor_high, avg, base) * max_markup
    high = min(high, current_price * (1 + max_increase), avg * mean_cap_ratio)
    if high <= low:
        return round_price(max(low, base))

    space = [Real(low, high)]

    def objective(x: List[float]) -> float:
        p = x[0]
        profit = simulate_profit(
            p,
            unit_cost,
            avg,
            demand_base,
            elasticity,
            saturation,
        )
        return -profit

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=max(5, n_calls // 4),
        random_state=42,
    )

    best_price = res.x[0]
    best_price = max(best_price, base)
    best_price = min(best_price, current_price * (1 + max_increase))
    best_price = max(best_price, current_price * (1 - max_decrease))
    best_price = min(best_price, avg * mean_cap_ratio)
    return round_price(best_price)


try:
    import requests
except ImportError:  # requests might not be installed; LLM calls become optional
    requests = None


OVERVIEW_CSV = BASE_DIR / "Dzukou_Pricing_Overview_With_Names - Copy.csv"
MAPPING_CSV = BASE_DIR / "product_data_mapping.csv"

# Minimum profit margins by category
PROFIT_MARGINS = {
    "Sunglasses": 0.15,
    "Bottles": 0.10,
    "Coffee mugs": 0.10,
    "Phone accessories": 0.10,
    "Notebook": 0.10,
    "Lunchbox": 0.10,
    "Premium shawls": 0.30,
    "Eri silk shawls": 0.20,
    "Cotton scarf": 0.15,
    "Other scarves and shawls": 0.15,
    "Cushion covers": 0.20,
    "Coasters & placements": 0.15,
    "Towels": 0.15,
}

# Category-specific demand elasticity (higher values mean more price sensitive)
DEMAND_ELASTICITY = {
    "Sunglasses": 1.3,
    "Bottles": 1.1,
    "Coffee mugs": 1.1,
    "Phone accessories": 1.2,
    "Notebook": 1.0,
    "Lunchbox": 1.0,
    "Premium shawls": 0.8,
    "Eri silk shawls": 0.9,
    "Cotton scarf": 1.1,
    "Other scarves and shawls": 1.1,
    "Cushion covers": 1.0,
    "Coasters & placements": 1.0,
    "Towels": 1.0,
}

# Maximum market saturation factor per category. A value of 1.0 means
# demand cannot exceed ``demand_base`` units when the price approaches
# zero. These can be tuned from historical sales data.
DEMAND_SATURATION = {
    "Sunglasses": 1.0,
    "Bottles": 1.0,
    "Coffee mugs": 1.0,
    "Phone accessories": 1.0,
    "Notebook": 1.0,
    "Lunchbox": 1.0,
    "Premium shawls": 1.0,
    "Eri silk shawls": 1.0,
    "Cotton scarf": 1.0,
    "Other scarves and shawls": 1.0,
    "Cushion covers": 1.0,
    "Coasters & placements": 1.0,
    "Towels": 1.0,
}

# Maximum markup relative to the average competitor price
MAX_MARKUP = {
    "Sunglasses": 1.8,
    "Bottles": 1.6,
    "Coffee mugs": 1.6,
    "Phone accessories": 1.5,
    "Notebook": 1.5,
    "Lunchbox": 1.6,
    "Premium shawls": 2.0,
    "Eri silk shawls": 1.9,
    "Cotton scarf": 1.7,
    "Other scarves and shawls": 1.7,
    "Cushion covers": 1.6,
    "Coasters & placements": 1.6,
    "Towels": 1.5,
}

# Maximum allowed price increase relative to the current price
MAX_INCREASE = {
    "default": 0.30,
    "Sunglasses": 0.30,
    "Bottles": 0.30,
    "Coffee mugs": 0.30,
    "Phone accessories": 0.30,
    "Notebook": 0.30,
    "Lunchbox": 0.30,
    "Premium shawls": 0.30,
    "Eri silk shawls": 0.30,
    "Cotton scarf": 0.30,
    "Other scarves and shawls": 0.30,
    "Cushion covers": 0.30,
    "Coasters & placements": 0.30,
    "Towels": 0.30,
}

# Maximum allowed price decrease relative to the current price
MAX_DECREASE = {
    "default": 0.25,
    "Sunglasses": 0.25,
    "Bottles": 0.25,
    "Coffee mugs": 0.25,
    "Phone accessories": 0.25,
    "Notebook": 0.25,
    "Lunchbox": 0.25,
    "Premium shawls": 0.25,
    "Eri silk shawls": 0.25,
    "Cotton scarf": 0.25,
    "Other scarves and shawls": 0.25,
    "Cushion covers": 0.25,
    "Coasters & placements": 0.25,
    "Towels": 0.25,
}


def round_price(price: float) -> float:
    """Round price to a corporate-friendly format (e.g., .99 or .49)."""
    # round to nearest 0.50 then subtract 0.01 for psychological pricing
    rounded = round(price * 2) / 2
    if rounded > 50:
        # for high priced items prefer .99 endings
        return int(rounded) - 0.01
    # alternate between .49 and .99 for lower price points
    return rounded - 0.01

# Default keywords used if no external mapping file is present. The
# mapping associates categories with words that identify products in that
# category. ``manage_products.py`` can update ``category_keywords.json`` to
# customize this mapping.
DEFAULT_CATEGORY_KEYWORDS = {
    "Sunglasses": ["sunglasses"],
    "Bottles": ["bottle"],
    "Coffee mugs": ["mug"],
    "Phone accessories": ["phone"],
    "Notebook": ["notebook"],
    "Lunchbox": ["lunchbox"],
    "Premium shawls": ["premium shawl"],
    "Eri silk shawls": ["eri silk"],
    "Cotton scarf": ["cotton scarf"],
    "Other scarves and shawls": ["stole", "shawl", "scarf"],
}

KEYWORDS_JSON = BASE_DIR / "category_keywords.json"


def load_category_keywords() -> Dict[str, List[str]]:
    """Return category keywords loaded from ``KEYWORDS_JSON`` if present."""
    path = KEYWORDS_JSON
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: list(map(str, v)) for k, v in data.items()}
        except Exception:
            pass
    return DEFAULT_CATEGORY_KEYWORDS


CATEGORY_KEYWORDS = load_category_keywords()


def categorize_product(name: str) -> str:
    name_l = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_l:
                return category
    return "Other scarves and shawls"  # default to shawls if unknown


def read_overview() -> Dict[str, Dict[str, float]]:
    data = {}
    # The overview CSV may come from Excel and often uses Windows-1252 encoding
    with open(OVERVIEW_CSV, newline="", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Product Name"].strip()
            cur_price = float(row[" Current Price "].replace("€", "").strip())
            unit_cost = float(row[" Unit Cost "].replace("€", "").strip())
            data[name] = {"current_price": cur_price, "unit_cost": unit_cost}
    return data


def read_prices(csv_path: Path, category: str | None = None) -> List[float]:
    """Return cleaned competitor prices from a CSV file.

    The files scraped for competitor pricing often include unrelated
    products. When ``category`` is provided, rows whose product name does
    not contain any of the configured keywords for that category are
    discarded before the price cleaning step.
    """

    prices = []
    keywords: List[str] | None = None
    if category:
        words = CATEGORY_KEYWORDS.get(category)
        if words:
            keywords = [w.lower() for w in words]

    with open(csv_path, newline="", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keywords:
                name_field = (row.get("product_name") or "").lower()
                term_field = (row.get("search_term") or "").lower()
                if not any(k in name_field or k in term_field for k in keywords):
                    continue

            price = row.get("price") or row.get("Price")
            if not price:
                continue
            price = price.replace("€", "").replace(",", "").strip()
            try:
                value = float(price)
            except ValueError:
                continue
            if 0 < value < 10000:
                prices.append(value)
    return clean_prices(prices)


def fallback_price(avg: float, min_p: float, cur: float, unit: float, margin: float) -> float:
    base = unit * (1 + margin)
    return max(base, cur * 1.05, avg, min_p * 1.1)


def call_mistral(prompt: str) -> float:
    """Query the Mistral API and extract a numeric price from the response."""
    if requests is None:
        raise RuntimeError("requests package not installed")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable not set")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, headers=headers, json=data, timeout=10)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\d+(?:\.\d+)?", content)
    if not match:
        raise ValueError("No numeric price returned")
    return float(match.group())


def call_groq(prompt: str, model: str | None = None) -> float:
    """Use Groq API to get a numeric price recommendation."""
    if requests is None:
        raise RuntimeError("requests package not installed")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    if not model:
        model = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(url, headers=headers, json=data, timeout=10)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\d+(?:\.\d+)?", content)
    if not match:
        raise ValueError("No numeric price returned")
    return float(match.group())


def call_gemini(prompt: str) -> float:
    """Use Google's Gemini API to get a numeric price recommendation."""
    if requests is None:
        raise RuntimeError("requests package not installed")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-002:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = requests.post(url, headers=headers, json=data, timeout=10)
    resp.raise_for_status()
    content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    match = re.search(r"\d+(?:\.\d+)?", content)
    if not match:
        raise ValueError("No numeric price returned")
    return float(match.group())


def call_llm(prompt: str) -> float:
    """Try multiple LLM providers and return the first successful price."""
    for func in (call_mistral, call_groq, call_gemini):
        try:
            return func(prompt)
        except Exception:
            continue
    raise RuntimeError("No LLM API available")


def simulate_profit(
    price: float,
    unit_cost: float,
    avg_competitor: float,
    demand_base: float = 100.0,
    elasticity: float = 1.2,
    saturation: float = 1.0,
) -> float:
    """Estimate profit using a logistic demand model with market dynamics."""
    if price <= 0:
        return 0.0

    # Price competitiveness factor
    relative_price = price / max(avg_competitor, 0.01)

    # Demand calculation with saturation and elasticity
    # When ``relative_price`` is below 1 the ``(relative_price - 1)`` term
    # becomes negative. Raising a negative number to a fractional power would
    # result in a complex value which breaks later calculations. Using the
    # absolute difference keeps the demand model smooth while avoiding complex
    # results.
    demand = demand_base * saturation / (
        1 + abs(relative_price - 1) ** elasticity
    )

    # Additional market adjustments
    if relative_price > 1.5:  # heavy penalty for very high prices
        demand *= 0.7
    elif relative_price < 0.8:  # slight boost for bargains
        demand *= 1.2

    demand = max(0, demand)

    return demand * (price - unit_cost)


def run_ab_test(
    current_price: float,
    new_price: float,
    unit_cost: float,
    avg_competitor: float,
    demand_base: float = 100.0,
    elasticity: float = 1.2,
    saturation: float = 1.0,
    noise: float = 0.1,
    n: int = 1000,
) -> Dict[str, float]:
    """Simulate an A/B test comparing two prices."""
    profits_control = []
    profits_test = []
    for _ in range(n):
        base_a = demand_base * saturation / (1 + (current_price / max(avg_competitor, 0.01)) ** elasticity)
        base_b = demand_base * saturation / (1 + (new_price / max(avg_competitor, 0.01)) ** elasticity)
        demand_a = base_a * max(0.0, random.gauss(1.0, noise))
        demand_b = base_b * max(0.0, random.gauss(1.0, noise))
        profits_control.append(demand_a * (current_price - unit_cost))
        profits_test.append(demand_b * (new_price - unit_cost))

    mean_a = statistics.mean(profits_control)
    mean_b = statistics.mean(profits_test)
    stat, p_value = stats.ttest_ind(profits_test, profits_control, equal_var=False)
    return {
        "profit_control": mean_a,
        "profit_test": mean_b,
        "profit_delta": mean_b - mean_a,
        "p_value": float(p_value),
    }




def suggest_price(
    product_name: str,
    category: str,
    prices: List[float],
    cur: float,
    unit: float,
) -> float:
    margin = PROFIT_MARGINS.get(category, 0.30)
    elasticity = DEMAND_ELASTICITY.get(category, 1.2)
    max_markup = MAX_MARKUP.get(category, 1.8)
    max_increase = MAX_INCREASE.get(category, MAX_INCREASE["default"])
    max_decrease = MAX_DECREASE.get(category, MAX_DECREASE["default"])

    if not prices:
        return round_price(max(unit * (1 + margin), cur))

    avg = statistics.mean(prices)
    median = statistics.median(prices)
    stdev = statistics.stdev(prices) if len(prices) > 1 else 0.0
    min_p = min(prices)
    max_p = max(prices)
    prompt = (
        f"Product: {product_name}\n"
        f"Category: {category}\n"
        f"Competitor prices: {', '.join(f'{p:.2f}' for p in prices)}\n"
        f"Average price: {avg:.2f}\n"
        f"Median price: {median:.2f}\n"
        f"Std Dev: {stdev:.2f}\n"
        f"Min price: {min_p:.2f}\n"
        f"Max price: {max_p:.2f}\n"
        f"Current price: {cur:.2f}\n"
        f"Unit cost: {unit:.2f}\n"
        f"Required margin: {margin*100:.0f}%\n"
        "Recommend a competitive selling price that maximizes profit. "
        "Only respond with the number."
    )
    try:
        price = call_llm(prompt)
    except Exception:
        price = bayesian_optimize_price(
            prices,
            cur,
            unit,
            margin,
            elasticity=elasticity,
            max_markup=max_markup,
            max_increase=max_increase,
            max_decrease=max_decrease,
            saturation=DEMAND_SATURATION.get(category, 1.0),
            mean_cap_ratio=compute_mean_cap(avg, stdev),
        )
    else:
        base = unit * (1 + margin)
        price = max(price, base)
        price = min(price, cur * (1 + max_increase))
        price = max(price, cur * (1 - max_decrease))
        price = min(price, avg * compute_mean_cap(avg, stdev))
    return round_price(price)


def main():
    overview = read_overview()
    results = []
    total_current = 0.0
    total_recommended = 0.0
    with open(MAPPING_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Product Name"].strip()
            data_file = Path(row["Data File"].strip())
            info = overview.get(name)
            if not info:
                continue
            category = categorize_product(name)
            prices = read_prices(data_file, category=category)
            price = suggest_price(
                name,
                category,
                prices,
                info["current_price"],
                info["unit_cost"],
            )
            avg = statistics.mean(prices) if prices else info["current_price"]
            median = statistics.median(prices) if prices else info["current_price"]
            stdev = statistics.stdev(prices) if len(prices) > 1 else 0.0
            min_p = min(prices) if prices else info["current_price"]
            max_p = max(prices) if prices else info["current_price"]

            saturation = DEMAND_SATURATION.get(category, 1.0)
            profit_cur = simulate_profit(
                info["current_price"],
                info["unit_cost"],
                avg,
                saturation=saturation,
            )
            profit_new = simulate_profit(
                price,
                info["unit_cost"],
                avg,
                saturation=saturation,
            )

            ab = run_ab_test(
                info["current_price"],
                price,
                info["unit_cost"],
                avg,
                elasticity=DEMAND_ELASTICITY.get(category, 1.2),
                saturation=saturation,
            )

            total_current += profit_cur
            total_recommended += profit_new

            results.append({
                "Product Name": name,
                "Product ID": row["Product ID"],
                "Recommended Price": f"{price:.2f}",
                "Category": category,
                "Avg Competitor Price": f"{avg:.2f}",
                "Min Competitor Price": f"{min_p:.2f}",
                "Max Competitor Price": f"{max_p:.2f}",
                "Median Competitor Price": f"{median:.2f}",
                "Std Competitor Price": f"{stdev:.2f}",
                "Competitor Count": len(prices),
                "Profit Current": f"{profit_cur:.2f}",
                "Profit Recommended": f"{profit_new:.2f}",
                "Profit Delta": f"{(profit_new - profit_cur):.2f}",
                "AB Profit Control": f"{ab['profit_control']:.2f}",
                "AB Profit Test": f"{ab['profit_test']:.2f}",
                "AB Profit Delta": f"{ab['profit_delta']:.2f}",
                # Use scientific notation to avoid displaying extremely small
                # p-values as zero when rounded. Four decimal places caused
                # all values to appear as 0.0000, so show the raw magnitude
                # with exponent format.
                "AB P-Value": f"{ab['p_value']:.3e}",
            })
    out_path = BASE_DIR / "recommended_prices.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Product Name",
                "Product ID",
                "Recommended Price",
                "Category",
                "Avg Competitor Price",
                "Min Competitor Price",
                "Max Competitor Price",
                "Median Competitor Price",
                "Std Competitor Price",
                "Competitor Count",
                "Profit Current",
                "Profit Recommended",
                "Profit Delta",
                "AB Profit Control",
                "AB Profit Test",
                "AB Profit Delta",
                "AB P-Value",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} recommendations to {str(out_path)}")
    print(
        f"Total estimated profit now: {total_current:.2f} -> {total_recommended:.2f} (delta {(total_recommended-total_current):.2f})"
    )


if __name__ == "__main__":
    main()
