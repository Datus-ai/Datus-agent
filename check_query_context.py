#!/usr/bin/env python3
"""
Superset Chart Query Context Checker

Check which charts have empty query_context via /api/v1/explore?slice_id={id}
"""

import argparse
import json
import os
import re
import sys
import time

import requests

# Configuration
SUPERSET_URL = os.environ.get("SUPERSET_URL", "http://127.0.0.1:8088")
SUPERSET_USER = os.environ.get("SUPERSET_USER", "admin")
SUPERSET_PASS = os.environ.get("SUPERSET_PASS", "admin")


class SupersetClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
        )

    def login(self) -> bool:
        """Login to Superset"""
        print(f"Logging into {self.base_url}...")

        try:
            # Get CSRF token
            login_page = self.session.get(f"{self.base_url}/login/", timeout=10)
            csrf_match = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login_page.text)
            csrf_token = csrf_match.group(1) if csrf_match else ""

            # Login
            resp = self.session.post(
                f"{self.base_url}/login/",
                data={"username": self.username, "password": self.password, "csrf_token": csrf_token},
                timeout=10,
            )

            if "/login" in resp.url:
                print("[ERROR] Login failed")
                return False

            print("[OK] Login successful")
            return True

        except Exception as e:
            print(f"[ERROR] Login exception: {e}")
            return False

    def get_all_charts(self) -> list:
        """Get all charts via API"""
        print("Fetching all charts...")

        charts = []
        page = 0
        page_size = 100

        while True:
            try:
                resp = self.session.get(
                    f"{self.base_url}/api/v1/chart/",
                    params={"q": json.dumps({"page": page, "page_size": page_size})},
                    timeout=30,
                )

                if resp.status_code != 200:
                    print(f"[ERROR] Failed to fetch charts: {resp.status_code}")
                    break

                data = resp.json()
                results = data.get("result", [])

                if not results:
                    break

                for chart in results:
                    charts.append(
                        {
                            "id": chart.get("id"),
                            "name": chart.get("slice_name", "Unknown"),
                            "viz_type": chart.get("viz_type", "Unknown"),
                        }
                    )

                if len(charts) >= data.get("count", 0):
                    break

                page += 1

            except Exception as e:
                print(f"[ERROR] Exception fetching charts: {e}")
                break

        print(f"[OK] Found {len(charts)} charts")
        return charts

    def check_query_context(self, slice_id: int) -> dict:
        """
        Check query_context for a specific chart via /api/v1/explore?slice_id={id}

        Returns:
            dict with keys: has_query_context, query_context, error
        """
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/explore/", params={"slice_id": slice_id}, timeout=30)

            if resp.status_code != 200:
                return {"has_query_context": False, "query_context": None, "error": f"HTTP {resp.status_code}"}

            data = resp.json()
            result = data.get("result", {})
            slice_info = result.get("slice", {})

            # Check for query_context in the response
            query_context = slice_info.get("query_context")

            # Determine if query_context is empty
            is_empty = (
                query_context is None
                or query_context == ""
                or query_context == {}
                or (isinstance(query_context, str) and query_context.strip() == "")
            )

            return {"has_query_context": not is_empty, "query_context": query_context, "error": None}

        except Exception as e:
            return {"has_query_context": False, "query_context": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Check which Superset charts have empty query_context")
    parser.add_argument("--url", "-u", default=SUPERSET_URL, help=f"Superset URL (default: {SUPERSET_URL})")
    parser.add_argument("--username", default=SUPERSET_USER, help="Superset username (default: admin)")
    parser.add_argument("--password", default=SUPERSET_PASS, help="Superset password (default: admin)")
    parser.add_argument(
        "--output",
        "-o",
        choices=["summary", "json", "detail"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument("--only-empty", action="store_true", help="Only show charts with empty query_context")

    args = parser.parse_args()

    # Create client and login
    client = SupersetClient(args.url, args.username, args.password)

    if not client.login():
        sys.exit(1)

    # Get all charts
    charts = client.get_all_charts()

    if not charts:
        print("No charts found")
        sys.exit(0)

    # Check each chart
    print(f"\nChecking query_context for {len(charts)} charts...\n")

    results = []
    empty_charts = []
    error_charts = []

    for i, chart in enumerate(charts, 1):
        slice_id = chart["id"]
        slice_name = chart["name"]
        viz_type = chart["viz_type"]

        check = client.check_query_context(slice_id)

        result = {
            "id": slice_id,
            "name": slice_name,
            "viz_type": viz_type,
            "has_query_context": check["has_query_context"],
            "error": check["error"],
        }
        results.append(result)

        # Categorize
        if check["error"]:
            error_charts.append(result)
            status = f"[ERROR] {check['error']}"
        elif check["has_query_context"]:
            status = "[OK] has query_context"
        else:
            empty_charts.append(result)
            status = "[EMPTY] query_context is empty"

        if args.output == "detail" or (args.only_empty and not check["has_query_context"]):
            print(f"[{i}/{len(charts)}] slice_id={slice_id} {slice_name[:40]:<40} {status}")
        elif args.output == "summary" and not args.only_empty:
            # Progress indicator
            print(f"\rChecking... {i}/{len(charts)}", end="", flush=True)
        time.sleep(0.3)

    if args.output == "summary" and not args.only_empty:
        print()  # New line after progress

    # Output results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total charts:              {len(charts)}")
    print(f"With query_context:        {len(charts) - len(empty_charts) - len(error_charts)}")
    print(f"Empty query_context:       {len(empty_charts)}")
    print(f"Errors:                    {len(error_charts)}")
    print("=" * 60)

    if empty_charts:
        print("\nCharts with EMPTY query_context:")
        print("-" * 60)
        for chart in empty_charts:
            print(f"  slice_id={chart['id']:<6} [{chart['viz_type']:<20}] {chart['name']}")

    if error_charts:
        print("\nCharts with ERRORS:")
        print("-" * 60)
        for chart in error_charts:
            print(f"  slice_id={chart['id']:<6} {chart['name'][:30]:<30} Error: {chart['error']}")

    # JSON output
    if args.output == "json":
        print("\n" + "=" * 60)
        print("JSON OUTPUT")
        print("=" * 60)
        output = {
            "total": len(charts),
            "with_query_context": len(charts) - len(empty_charts) - len(error_charts),
            "empty_query_context": len(empty_charts),
            "errors": len(error_charts),
            "empty_charts": empty_charts,
            "error_charts": error_charts,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

    # Exit code: 1 if there are empty charts
    if empty_charts:
        print(f"\n[WARN] Found {len(empty_charts)} charts with empty query_context")
        sys.exit(1)
    else:
        print("\n[OK] All charts have query_context")
        sys.exit(0)


if __name__ == "__main__":
    main()
