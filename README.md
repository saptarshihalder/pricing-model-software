# Dzukou Pricing Toolkit

This repository bundles several scripts for managing product data, scraping competitor prices, optimizing recommendations and viewing the results in an interactive dashboard.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main steps using the CLI wrapper or open the GUI:

```bash
python -m dzukou_dynamic_pricing_LLM.cli manage     # add products
python -m dzukou_dynamic_pricing_LLM.cli scrape     # collect competitor data
python -m dzukou_dynamic_pricing_LLM.cli optimize   # compute new prices
python -m dzukou_dynamic_pricing_LLM.cli dashboard  # open dashboard.html
python -m dzukou_dynamic_pricing_LLM.app_gui        # launch GUI app
```

See `dzukou-dynamic-pricing-LLM/README.md` for detailed information about each module.

Licensed under the MIT License.
