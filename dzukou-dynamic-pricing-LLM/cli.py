import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for the Dzukou pricing toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("manage", help="Add products and categories")
    subparsers.add_parser("scrape", help="Scrape competitor prices")
    subparsers.add_parser("optimize", help="Generate price recommendations")
    subparsers.add_parser("dashboard", help="Display the dashboard")

    args = parser.parse_args()

    if args.command == "manage":
        from .manage_products import main as manage_main
        manage_main()
    elif args.command == "scrape":
        from .scraper import main as scrape_main
        scrape_main()
    elif args.command == "optimize":
        from .price_optimizer import main as optimize_main
        optimize_main()
    elif args.command == "dashboard":
        from .dashboard import main as dashboard_main
        dashboard_main()


if __name__ == "__main__":
    main()
