#!/usr/bin/env python3
"""Fetch Gemini discovery routes and compare or print the pytest fixture."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


DISCOVERY_URL = "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta"
TESTS_PATH = Path("tests/test_antigravity_proxy.py")


def _walk_methods(resource: dict[str, Any]):
    for method in resource.get("methods", {}).values():
        http_method = method.get("httpMethod")
        flat_path = method.get("flatPath") or method.get("path")
        if http_method and flat_path:
            yield str(http_method).upper(), str(flat_path)
    for child in resource.get("resources", {}).values():
        yield from _walk_methods(child)


def fetch_discovery(url: str = DISCOVERY_URL) -> tuple[str, tuple[tuple[str, str], ...]]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    revision = str(payload.get("revision", ""))
    routes = tuple(sorted(set(_walk_methods(payload))))
    return revision, routes


def _literal_after_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find {name} in {TESTS_PATH}")


def load_fixture(path: Path = TESTS_PATH) -> tuple[str, tuple[tuple[str, str], ...]]:
    source = path.read_text(encoding="utf-8")
    revision = _literal_after_assignment(source, "GEMINI_V1BETA_DISCOVERY_REVISION")
    routes_name_match = re.search(r"GEMINI_V1BETA_DISCOVERY_ROUTES_\d+", source)
    if not routes_name_match:
        raise RuntimeError(f"Could not find Gemini route fixture in {path}")
    routes = _literal_after_assignment(source, routes_name_match.group(0))
    return str(revision), tuple(tuple(item) for item in routes)


def format_fixture(revision: str, routes: tuple[tuple[str, str], ...]) -> str:
    lines = [
        f'GEMINI_V1BETA_DISCOVERY_REVISION = "{revision}"',
        "",
        f"GEMINI_V1BETA_DISCOVERY_ROUTES_{revision} = (",
    ]
    lines.extend(f'    ("{method}", "{flat_path}"),' for method, flat_path in routes)
    lines.append(")")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed fixture is stale")
    parser.add_argument("--url", default=DISCOVERY_URL, help="Gemini discovery URL")
    args = parser.parse_args()

    revision, routes = fetch_discovery(args.url)
    if not args.check:
        print(format_fixture(revision, routes))
        return 0

    fixture_revision, fixture_routes = load_fixture()
    if (revision, routes) == (fixture_revision, fixture_routes):
        print(f"Gemini discovery fixture is current: revision {revision}, {len(routes)} routes.")
        return 0

    missing = sorted(set(routes) - set(fixture_routes))
    extra = sorted(set(fixture_routes) - set(routes))
    print(
        f"Gemini discovery fixture is stale: live revision {revision}, "
        f"fixture revision {fixture_revision}.",
        file=sys.stderr,
    )
    if missing:
        print("Missing routes:", file=sys.stderr)
        for method, flat_path in missing:
            print(f"  {method} {flat_path}", file=sys.stderr)
    if extra:
        print("Extra routes:", file=sys.stderr)
        for method, flat_path in extra:
            print(f"  {method} {flat_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
