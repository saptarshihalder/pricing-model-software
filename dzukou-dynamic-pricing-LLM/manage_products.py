#!/usr/bin/env python3
"""GUI utility for adding products and keywords used by the pricing pipeline."""
import csv
import json
import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import ast

from scraper import DEFAULT_CATEGORIES
from utils import canonical_key

BASE_DIR = Path(__file__).resolve().parent
MAPPING_CSV = BASE_DIR / "product_data_mapping.csv"
KEYWORDS_JSON = BASE_DIR / "category_keywords.json"
OVERVIEW_CSV = BASE_DIR / "Dzukou_Pricing_Overview_With_Names - Copy.csv"
DATA_DIR = BASE_DIR / "product_data"
SCRAPER_PY = BASE_DIR / "scraper.py"
class ProductManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Data Manager")
        self.root.geometry("600x500")

        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))

        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="Add Product to Pricing Pipeline", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        ttk.Label(main_frame, text="Product Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.name_entry = ttk.Entry(main_frame, width=40)
        self.name_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Product ID:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.id_entry = ttk.Entry(main_frame, width=40)
        self.id_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Category:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.category_entry = ttk.Entry(main_frame, width=40)
        self.category_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Current Price:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.price_entry = ttk.Entry(main_frame, width=40)
        self.price_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Unit Cost:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.cost_entry = ttk.Entry(main_frame, width=40)
        self.cost_entry.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="Keywords:").grid(row=6, column=0, sticky=(tk.W, tk.N), pady=5)
        keywords_frame = ttk.Frame(main_frame)
        keywords_frame.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5)

        self.keywords_entry = ttk.Entry(keywords_frame, width=40)
        self.keywords_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(keywords_frame, text="(comma-separated)", font=('Arial', 9, 'italic')).grid(row=1, column=0, sticky=tk.W)
        keywords_frame.columnconfigure(0, weight=1)

        self.add_button = ttk.Button(main_frame, text="Add Product", command=self.add_product)
        self.add_button.grid(row=7, column=0, columnspan=2, pady=20)

        ttk.Label(main_frame, text="Output:").grid(row=8, column=0, sticky=(tk.W, tk.N), pady=5)
        self.output_text = scrolledtext.ScrolledText(main_frame, height=8, width=50)
        self.output_text.grid(row=8, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        main_frame.rowconfigure(8, weight=1)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready to add products")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.update_status()

    def canonical_category(self, category: str) -> str:
        """Return a canonical category name if it already exists."""
        key = canonical_key(category)
        existing = list(DEFAULT_CATEGORIES.keys())
        existing.extend(self.load_keywords().keys())
        for cat in existing:
            if canonical_key(cat) == key:
                return cat
        return category.strip()

    def sanitize_filename(self, name: str) -> str:
        base = re.sub(r"\W+", "_", name.lower()).strip("_")
        return base + ".csv"

    def load_keywords(self) -> dict:
        if KEYWORDS_JSON.exists():
            with open(KEYWORDS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_keywords(self, data: dict) -> None:
        with open(KEYWORDS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def update_scraper_categories(self, category: str, keywords: list, csv_filename: str):
        """Update the DEFAULT_CATEGORIES in scraper.py file."""
        try:
            # Read the current scraper.py file
            if not SCRAPER_PY.exists():
                messagebox.showwarning(
                    "Warning", "scraper.py file not found. Cannot update DEFAULT_CATEGORIES."
                )
                return False

            with open(SCRAPER_PY, "r", encoding="utf-8") as f:
                content = f.read()

            # Find the DEFAULT_CATEGORIES dictionary
            pattern = r"DEFAULT_CATEGORIES\s*=\s*\{[\s\S]*?\n\}"
            match = re.search(pattern, content)

            if not match:
                messagebox.showwarning(
                    "Warning", "DEFAULT_CATEGORIES not found in scraper.py"
                )
                return False

            # Load existing categories
            categories = DEFAULT_CATEGORIES.copy()

            # Update or add the new category
            categories[category] = {
                "search_terms": keywords,
                "csv_filename": csv_filename,
            }

            # Generate the new DEFAULT_CATEGORIES string
            new_dict_str = "DEFAULT_CATEGORIES = {\n"
            for cat_name, cat_data in categories.items():
                new_dict_str += f'    "{cat_name}": {{\n'
                new_dict_str += "        \"search_terms\": [\n"
                for term in cat_data["search_terms"]:
                    new_dict_str += f'            "{term}",\n'
                new_dict_str += "        ],\n"
                new_dict_str += f'        "csv_filename": "{cat_data["csv_filename"]}",\n'
                new_dict_str += "    },\n"
            new_dict_str += "}"

            # Replace the old dictionary with the new one
            new_content = content[: match.start()] + new_dict_str + content[match.end() :]

            # Write back to the file
            with open(SCRAPER_PY, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update scraper.py: {str(e)}")
            return False

    def add_product(self):
        name = self.name_entry.get().strip()
        prod_id = self.id_entry.get().strip()
        category_input = self.category_entry.get().strip()
        category = self.canonical_category(category_input)
        price_str = self.price_entry.get().strip()
        cost_str = self.cost_entry.get().strip()
        keywords_str = self.keywords_entry.get().strip()

        if not name:
            messagebox.showerror("Error", "Product name is required")
            return
        if not prod_id:
            messagebox.showerror("Error", "Product ID is required")
            return
        if not category:
            messagebox.showerror("Error", "Category is required")
            return
        if not keywords_str:
            messagebox.showerror("Error", "At least one keyword is required")
            return
        if not price_str or not cost_str:
            messagebox.showerror("Error", "Price and unit cost are required")
            return

        try:
            price_val = float(price_str)
            cost_val = float(cost_str)
        except ValueError:
            messagebox.showerror("Error", "Price and cost must be numbers")
            return

        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

        try:
            DATA_DIR.mkdir(exist_ok=True)

            mapping = []
            if MAPPING_CSV.exists():
                with open(MAPPING_CSV, newline="") as f:
                    mapping = list(csv.DictReader(f))

            # Use a single CSV per category so competitor data is grouped
            # regardless of product names
            file_name = self.sanitize_filename(category)
            data_file = DATA_DIR / file_name
            if not data_file.exists():
                data_file.write_text("category,store,product_name,price,search_term,store_url\n")

            mapping.append({"Product Name": name, "Product ID": prod_id, "Data File": str(data_file)})

            with open(MAPPING_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Product Name", "Product ID", "Data File"])
                writer.writeheader()
                writer.writerows(mapping)

            # update overview csv
            overview_rows = []
            if OVERVIEW_CSV.exists():
                with open(OVERVIEW_CSV, newline="", encoding="cp1252") as f:
                    overview_rows = list(csv.DictReader(f))

            overview_rows.append({
                "Product Name": name,
                "Product ID": prod_id,
                " Current Price ": f"{price_val}",
                " Unit Cost ": f"{cost_val}"
            })

            with open(OVERVIEW_CSV, "w", newline="", encoding="cp1252") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Product Name", "Product ID", " Current Price ", " Unit Cost "],
                )
                writer.writeheader()
                writer.writerows(overview_rows)

            kw_data = self.load_keywords()
            kws = kw_data.setdefault(category, [])
            for kw in keywords:
                if kw and kw not in kws:
                    kws.append(kw)
            self.save_keywords(kw_data)

            # Update scraper.py with the new category
            scraper_updated = self.update_scraper_categories(category, keywords, file_name)

            output_msg = f"Added product '{name}' with data file {str(data_file)}\n"
            if category != category_input:
                output_msg += f"Category mapped to existing '{category}'.\n"
            if keywords:
                output_msg += f"Keywords added to category '{category}': {', '.join(keywords)}\n"
            if scraper_updated:
                output_msg += f"Updated scraper.py with category '{category}'\n"
            output_msg += "-" * 50 + "\n"

            self.output_text.insert(tk.END, output_msg)
            self.output_text.see(tk.END)

            self.name_entry.delete(0, tk.END)
            self.id_entry.delete(0, tk.END)
            self.category_entry.delete(0, tk.END)
            self.price_entry.delete(0, tk.END)
            self.cost_entry.delete(0, tk.END)
            self.keywords_entry.delete(0, tk.END)

            self.status_var.set(f"Successfully added product: {name}")
            self.update_status()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add product: {str(e)}")
            self.status_var.set("Error occurred while adding product")

    def update_status(self):
        try:
            product_count = 0
            if MAPPING_CSV.exists():
                with open(MAPPING_CSV, newline="") as f:
                    product_count = len(list(csv.DictReader(f)))

            category_count = 0
            if KEYWORDS_JSON.exists():
                kw_data = self.load_keywords()
                category_count = len(kw_data)

            status = f"Products: {product_count} | Categories: {category_count}"
            self.root.after(100, lambda: self.status_var.set(status))
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = ProductManagerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
