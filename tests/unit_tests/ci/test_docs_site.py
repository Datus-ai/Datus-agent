from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("docs_site", REPO_ROOT / "ci/docs_site.py")
assert SPEC and SPEC.loader
docs_site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs_site)


@pytest.fixture
def published_site(tmp_path):
    versions = [
        {"version": "dev", "aliases": []},
        {"version": "0.4", "aliases": ["latest"]},
        {"version": "0.3", "aliases": []},
    ]
    for locale in ("", "zh/"):
        docs_site.write(tmp_path, locale + "versions.json", json.dumps(versions))
        for version in ("dev", "0.4", "0.3"):
            docs_site.write(tmp_path, f"{locale}{version}/assets/theme.css", "body{}")
            for slug in ("", "configuration/agent/"):
                depth = len(slug.strip("/").split("/")) if slug else 0
                css = "../" * depth + "assets/theme.css"
                docs_site.write(
                    tmp_path,
                    f"{locale}{version}/{slug}index.html",
                    '<html><head><link rel="canonical" '
                    f'href="https://docs.datus.ai/{locale}{version}/{version}/{slug}">'
                    f'<link rel="stylesheet" href="{css}">'
                    '<link rel="alternate" hreflang="en" href="/">'
                    "</head><body><pre>first\n\nlast</pre></body></html>",
                )
    for locale in ("", "zh/"):
        docs_site.write(
            tmp_path,
            f"{locale}0.3/metricflow/introduction/index.html",
            "<html><head></head><body>Archived MetricFlow documentation</body></html>",
        )
    docs_site.write(tmp_path, "metricflow/introduction/index.html", docs_site.redirect_html("/latest/missing/"))
    docs_site.write(tmp_path, "0.3.0/zh/configuration/agent/index.html", docs_site.redirect_html("/0.3/"))
    docs_site.write(tmp_path, "unknown/index.html", docs_site.redirect_html("/latest/missing/"))
    docs_site.write(tmp_path, "sitemap.xml", '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>')
    return tmp_path


def canonical(site, path):
    source = (site / path).read_text()
    links = [docs_site.attributes(match[0]) for match in docs_site.LINK_RE.finditer(source)]
    return next(link["href"] for link in links if link.get("rel") == "canonical")


def test_finalization_repairs_historical_metadata_and_redirects_without_chains(published_site):
    docs_site.finalize(published_site)

    assert canonical(published_site, "0.4/configuration/agent/index.html") == (
        "https://docs.datus.ai/0.4/configuration/agent/"
    )
    assert canonical(published_site, "zh/0.4/0.4/configuration/agent/index.html") == (
        "https://docs.datus.ai/zh/0.4/configuration/agent/"
    )
    assert canonical(published_site, "dev/zh/configuration/agent/index.html") == (
        "https://docs.datus.ai/zh/dev/configuration/agent/"
    )
    assert canonical(published_site, "0.3.0/zh/configuration/agent/index.html") == (
        "https://docs.datus.ai/zh/0.3/configuration/agent/"
    )
    for path in ("metricflow/introduction/index.html", "latest/metricflow/introduction/index.html"):
        assert canonical(published_site, path) == "https://docs.datus.ai/0.3/metricflow/introduction/"
    assert not (published_site / "unknown/index.html").exists()
    assert "<pre>first\n\nlast</pre>" in (published_site / "0.4/index.html").read_text()
    error = (published_site / "404.html").read_text()
    assert 'content="noindex"' in error
    assert "stylesheet" not in error


def test_new_bilingual_routes_automatically_gain_entrypoints_and_sitemap_alternates(published_site):
    docs_site.finalize(published_site)
    for locale in ("", "zh/"):
        docs_site.write(
            published_site, f"{locale}0.4/new/nested/page/index.html", "<html><head></head><body>New</body></html>"
        )
    docs_site.finalize(published_site)

    for locale in ("", "zh/"):
        target = f"https://docs.datus.ai/{locale}0.4/new/nested/page/"
        for prefix in (locale, locale + "latest/"):
            assert canonical(published_site, f"{prefix}new/nested/page/index.html") == target
        sitemap = ET.parse(published_site / f"{locale}0.4/sitemap.xml")
        entry = next(
            entry for entry in sitemap.getroot() if entry.find(f"{{{docs_site.SITEMAP_NS}}}loc").text == target
        )
        twins = {link.attrib["hreflang"]: link.attrib["href"] for link in entry if "hreflang" in link.attrib}
        assert twins == {
            "en": "https://docs.datus.ai/0.4/new/nested/page/",
            "zh-Hans": "https://docs.datus.ai/zh/0.4/new/nested/page/",
            "x-default": "https://docs.datus.ai/0.4/new/nested/page/",
        }


def test_finalization_is_idempotent_and_preserves_english_only_pages(published_site):
    docs_site.write(published_site, "0.4/english-only/index.html", "<html><head></head><body>English</body></html>")
    docs_site.finalize(published_site)
    before = {
        file.relative_to(published_site): file.read_bytes() for file in published_site.rglob("*") if file.is_file()
    }
    docs_site.finalize(published_site)
    after = {
        file.relative_to(published_site): file.read_bytes() for file in published_site.rglob("*") if file.is_file()
    }
    assert after == before
    assert 'hreflang="zh-Hans"' not in (published_site / "0.4/english-only/index.html").read_text()


@pytest.mark.parametrize("failure", ["canonical", "stylesheet", "redirect", "sitemap", "hreflang"])
def test_publishing_rejects_invalid_cross_page_contracts(published_site, failure):
    docs_site.finalize(published_site)
    if failure == "stylesheet":
        (published_site / "0.4/assets/theme.css").unlink()
    elif failure == "redirect":
        docs_site.write(
            published_site, "bad/index.html", docs_site.redirect_html("https://docs.datus.ai/latest/missing/")
        )
    elif failure == "sitemap":
        file = published_site / "0.4/sitemap.xml"
        file.write_text(
            file.read_text().replace("https://docs.datus.ai/0.4/</loc>", "https://docs.datus.ai/missing/</loc>")
        )
    else:
        file = published_site / "0.4/index.html"
        if failure == "canonical":
            file.write_text(
                file.read_text().replace(
                    'rel="canonical" href="https://docs.datus.ai/0.4/"',
                    'rel="canonical" href="https://docs.datus.ai/0.4/0.4/"',
                )
            )
        else:
            file.write_text(file.read_text().replace('hreflang="zh-Hans"', 'hreflang="invalid"'))
    with pytest.raises(ValueError):
        docs_site.validate(published_site)


@pytest.mark.parametrize("locale", ["en", "zh"])
def test_workflow_site_url_builds_correct_native_mike_canonicals(tmp_path, locale):
    pytest.importorskip("mike", reason="The dedicated docs workflow installs requirements-docs.txt")
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/deploy-docs.yml").read_text())
    step = next(step for step in workflow["jobs"]["deploy"]["steps"] if step.get("name") == "Configure language builds")
    env_file = tmp_path / "github-env"
    env = {**os.environ, "DOCS_VERSION": "0.4", "GITHUB_ENV": str(env_file)}
    subprocess.run(["bash", "-eu", "-c", step["run"]], env=env, check=True)
    resolved = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    site_url = resolved[f"DOCS_{locale.upper()}_SITE_URL"]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text("# Home\n")
    (docs / "nested.md").write_text("# Nested\n")
    config = tmp_path / "mkdocs.yml"
    config.write_text(yaml.safe_dump({"site_name": "Test", "site_url": site_url, "plugins": ["mike"]}))
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(config)],
        env={**os.environ, "MIKE_DOCS_VERSION": "0.4"},
        check=True,
        capture_output=True,
        text=True,
    )
    public_url = "https://docs.datus.ai/" + ("zh/" if locale == "zh" else "") + "0.4/"
    assert canonical(tmp_path / "site", "nested/index.html") == public_url + "nested/"
    docs_site.validate_build(tmp_path / "site", public_url)
