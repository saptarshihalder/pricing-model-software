import tkinter as tk
from tkinter import ttk, messagebox
import threading


class PricingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dzukou Pricing App")
        self.root.geometry("300x250")

        frame = ttk.Frame(root, padding=10)
        frame.grid(sticky=(tk.N, tk.S, tk.E, tk.W))
        ttk.Label(
            frame,
            text="Dzukou Pricing Toolkit",
            font=("Arial", 14, "bold")
        ).grid(row=0, column=0, pady=10)

        ttk.Button(
            frame,
            text="Manage Products",
            command=self.open_product_manager
        ).grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Button(
            frame,
            text="Scrape Prices",
            command=self.run_scraper
        ).grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Button(
            frame,
            text="Optimize Prices",
            command=self.run_optimizer
        ).grid(row=3, column=0, sticky="ew", pady=5)
        ttk.Button(
            frame,
            text="View Dashboard",
            command=self.run_dashboard
        ).grid(row=4, column=0, sticky="ew", pady=5)
        ttk.Button(
            frame,
            text="Quit",
            command=root.quit
        ).grid(row=5, column=0, sticky="ew", pady=10)

    def open_product_manager(self):
        from .manage_products import ProductManagerGUI
        win = tk.Toplevel(self.root)
        ProductManagerGUI(win)

    def run_scraper(self):
        from .scraper import main as scrape_main
        threading.Thread(target=scrape_main, daemon=True).start()
        messagebox.showinfo(
            "Scraper",
            "Scraping started. Check console for progress."
        )

    def run_optimizer(self):
        from .price_optimizer import main as optimize_main
        threading.Thread(target=optimize_main, daemon=True).start()
        messagebox.showinfo(
            "Optimizer",
            "Optimization started. Check console for progress."
        )

    def run_dashboard(self):
        from .dashboard import main as dashboard_main
        dashboard_main()
        messagebox.showinfo(
            "Dashboard",
            "Dashboard generated as dashboard.html"
        )


def main():
    root = tk.Tk()
    PricingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
