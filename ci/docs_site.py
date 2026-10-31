#!/usr/bin/env python3
"""Finalize and validate the complete bilingual mike site before publishing it."""

import argparse
import gzip
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import yaml

SITE = "https://docs.datus.ai"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
XHTML_NS = "http://www.w3.org/1999/xhtml"
VERSION_RE = re.compile(r"\d+(?:\.\d+)+|dev|latest")
LINK_RE = re.compile(r"<link\b[^>]*>", re.IGNORECASE)
REFRESH_RE = re.compile(r"<meta\b[^>]*http-equiv=[\"\']refresh[\"\']", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(r"([\w-]+)=[\"\']([^\"\']*)[\"\']")
SEO_START = "<!-- datus-docs-seo -->"
SEO_END = "<!-- /datus-docs-seo -->"
SEO_BLOCK_RE = re.compile(re.escape(SEO_START) + r".*?" + re.escape(SEO_END) + r"\n?", re.DOTALL)
# Renamed entry points with direct replacements. Other retired pages stay on
# their newest available archive; unknown URLs retain an actual HTTP 404.
RENAMED_PAGES = yaml.safe_load((Path(__file__).with_name("docs_redirects.yml")).read_text(encoding="utf-8"))
EXTERNAL_REDIRECTS = {
    "blog/posts/meet_datus.html": "https://datus.ai/blog/meet_datus/",
}


def attributes(tag: str) -> dict[str, str]:
    return {key.lower(): html.unescape(value) for key, value in ATTRIBUTE_RE.findall(tag)}


def route(path: str) -> tuple[str, str, str]:
    parts = path.strip("/").removesuffix("index.html").strip("/").split("/")
    locale, version = "en", ""
    while parts and (parts[0] == "zh" or VERSION_RE.fullmatch(parts[0])):
        part = parts.pop(0)
        if part == "zh":
            locale = "zh"
        elif not version:
            version = part
    return locale, version, "/".join(parts)


@dataclass(frozen=True)
class Page:
    path: str
    locale: str
    version: str
    slug: str

    @property
    def url(self) -> str:
        return SITE + "/" + self.path.removesuffix("index.html")


def inventory(site_dir: Path) -> tuple[dict[tuple[str, str, str], Page], set[str]]:
    pages, redirects = {}, set()
    for file in sorted(site_dir.rglob("*.html")):
        path = file.relative_to(site_dir).as_posix()
        if REFRESH_RE.search(file.read_text(encoding="utf-8")):
            redirects.add(path)
        elif file.name == "index.html":
            locale, version, slug = route(path)
            if version:
                pages[locale, version, slug] = Page(path, locale, version, slug)
    return pages, redirects


def version_metadata(site_dir: Path) -> dict[str, str]:
    aliases = {}
    for prefix in ("", "zh/"):
        file = site_dir / prefix / "versions.json"
        if not file.exists():
            continue
        for item in json.loads(file.read_text(encoding="utf-8")):
            for alias in item.get("aliases", []):
                aliases[alias] = item["version"]
    return aliases


def resolve_page(path: str, pages: dict[tuple[str, str, str], Page], aliases: dict[str, str]) -> Page | None:
    locale, requested, slug = route(path)
    requested = aliases.get(requested, requested)
    versions = {page.version for page in pages.values() if page.locale == locale}
    if requested not in versions and re.fullmatch(r"\d+\.\d+\.\d+", requested):
        requested = requested.rsplit(".", 1)[0]
    stable = sorted(
        (v for v in versions if re.fullmatch(r"\d+(?:\.\d+)+", v)),
        key=lambda value: tuple(map(int, value.split("."))),
        reverse=True,
    )
    candidates = list(dict.fromkeys([requested, aliases.get("latest", ""), *stable, "dev"]))
    for version in candidates:
        if page := pages.get((locale, version, slug)):
            return page
    if replacement := RENAMED_PAGES.get(slug):
        for version in candidates:
            if page := pages.get((locale, version, replacement)):
                return page
    return None


def redirect_html(target: str) -> str:
    escaped = html.escape(target, quote=True)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Redirecting…</title>"
        f'<link rel="canonical" href="{escaped}">'
        f'<meta http-equiv="refresh" content="0; url={escaped}">'
        f"<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>"
        f'</head><body>Redirecting to <a href="{escaped}">{escaped}</a>…</body></html>\n'
    )


def not_found_html() -> str:
    # GitHub Pages serves only the root 404.html for arbitrary missing paths.
    # Keep it independent of a version's assets and never guess a redirect here.
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>Page not found - Datus Docs</title>
<style>body{font:18px/1.6 system-ui,sans-serif;color:#202735;background:#f5f7fa;margin:0}
main{max-width:680px;margin:12vh auto;padding:32px}a{color:#315cb7}h1{line-height:1.2}</style>
</head><body><main><p>Datus Docs</p><h1>404 — Page not found</h1>
<p>This page does not exist. Choose a documentation edition below.</p>
<p lang="zh">此页面不存在，请选择下方文档入口。</p>
<p><a href="/latest/">English documentation</a> · <a href="/zh/latest/">中文文档</a></p>
</main></body></html>
"""


def write(site_dir: Path, path: str, content: str) -> None:
    file = site_dir / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")


def alternates(page: Page, pages: dict[tuple[str, str, str], Page]) -> dict[str, str]:
    result = {}
    for locale, hreflang in (("en", "en"), ("zh", "zh-Hans")):
        if peer := pages.get((locale, page.version, page.slug)):
            result[hreflang] = peer.url
    if "en" in result:
        result["x-default"] = result["en"]
    return result


def update_page(site_dir: Path, page: Page, pages: dict[tuple[str, str, str], Page]) -> None:
    file = site_dir / page.path
    source = file.read_text(encoding="utf-8")

    def remove_seo_link(match: re.Match) -> str:
        attrs = attributes(match[0])
        if attrs.get("rel") == "canonical" or "hreflang" in attrs:
            return ""
        return match[0]

    head, separator, body = source.partition("</head>")
    if not separator:
        raise ValueError(f"Page is missing an HTML head: {page.path}")
    head = LINK_RE.sub(remove_seo_link, SEO_BLOCK_RE.sub("", head))
    tags = [f'<link rel="canonical" href="{page.url}">']
    tags.extend(
        f'<link rel="alternate" hreflang="{lang}" href="{url}">' for lang, url in alternates(page, pages).items()
    )
    # Replace only our own block on later deploys, preserving scripts and body.
    source = head + SEO_START + "\n" + "\n".join(tags) + "\n" + SEO_END + "\n" + separator + body
    file.write_text(source, encoding="utf-8")


def write_xml(site_dir: Path, path: str, tree: ET.Element) -> None:
    ET.register_namespace("", SITEMAP_NS)
    ET.register_namespace("xhtml", XHTML_NS)
    content = ET.tostring(tree, encoding="utf-8", xml_declaration=True) + b"\n"
    file = site_dir / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(content)
    file.with_suffix(file.suffix + ".gz").write_bytes(gzip.compress(content, mtime=0))


def write_sitemaps(site_dir: Path, pages: dict[tuple[str, str, str], Page]) -> None:
    index = ET.Element(f"{{{SITEMAP_NS}}}sitemapindex")
    maintained = {"sitemap.xml"}
    for prefix in ("", "zh/"):
        metadata_file = site_dir / prefix / "versions.json"
        if not metadata_file.exists():
            continue
        for item in json.loads(metadata_file.read_text(encoding="utf-8")):
            version = item["version"]
            locale = "zh" if prefix else "en"
            tree = ET.Element(f"{{{SITEMAP_NS}}}urlset")
            for page in sorted(pages.values(), key=lambda page: page.path):
                if (page.locale, page.version) != (locale, version):
                    continue
                entry = ET.SubElement(tree, f"{{{SITEMAP_NS}}}url")
                ET.SubElement(entry, f"{{{SITEMAP_NS}}}loc").text = page.url
                for lang, url in alternates(page, pages).items():
                    ET.SubElement(entry, f"{{{XHTML_NS}}}link", rel="alternate", hreflang=lang, href=url)
            if not len(tree):
                continue
            sitemap_path = f"{prefix}{version}/sitemap.xml"
            maintained.add(sitemap_path)
            write_xml(site_dir, sitemap_path, tree)
            if not item.get("properties", {}).get("hidden") and version != "dev":
                entry = ET.SubElement(index, f"{{{SITEMAP_NS}}}sitemap")
                ET.SubElement(entry, f"{{{SITEMAP_NS}}}loc").text = SITE + "/" + sitemap_path
    if not len(index):
        raise ValueError("No published stable documentation sitemap found")
    write_xml(site_dir, "sitemap.xml", index)
    # Older root/language/alias sitemaps outlive mike deploys unless removed.
    for file in site_dir.rglob("sitemap.xml"):
        if file.relative_to(site_dir).as_posix() not in maintained:
            file.unlink()
            file.with_suffix(".xml.gz").unlink(missing_ok=True)
    write(site_dir, "robots.txt", f"User-agent: *\nAllow: /\n\nSitemap: {SITE}/sitemap.xml\n")


def route_candidates(locale: str, slug: str, versions: set[str]) -> set[str]:
    prefix = "zh/" if locale == "zh" else ""
    suffix = f"{slug}/index.html" if slug else "index.html"
    result = {f"{prefix}{suffix}", f"{prefix}latest/{suffix}"}
    for version in versions:
        result.update({f"{prefix}{version}/{suffix}", f"{prefix}{version}/{version}/{suffix}"})
        if locale == "zh":
            result.add(f"{version}/zh/{suffix}")
    return result


def finalize(site_dir: Path) -> None:
    pages, redirects = inventory(site_dir)
    aliases = version_metadata(site_dir)
    if not pages or "latest" not in aliases:
        raise ValueError("Expected a published mike site with a latest alias")
    versions = {page.version for page in pages.values()}
    versions.update(
        file.name
        for file in site_dir.iterdir()
        if file.is_dir() and file.name != "latest" and VERSION_RE.fullmatch(file.name)
    )
    for page in pages.values():
        update_page(site_dir, page, pages)
        redirects.update(route_candidates(page.locale, page.slug, versions))
    for old_slug in RENAMED_PAGES:
        for locale in ("en", "zh"):
            redirects.update(route_candidates(locale, old_slug, versions))
    content_paths = {page.path for page in pages.values()}
    resolved_redirects = {}
    for path in sorted(redirects - content_paths):
        if target := resolve_page(path, pages, aliases):
            resolved_redirects[path] = target.url
        elif (site_dir / path).exists():
            # Remove stale stubs whose target no longer has any equivalent page.
            (site_dir / path).unlink()
    for path, target in resolved_redirects.items():
        write(site_dir, path, redirect_html(target))
    for path, target in EXTERNAL_REDIRECTS.items():
        write(site_dir, path, redirect_html(target))
    for file in [site_dir / "404.html", *site_dir.glob("**/404.html")]:
        file.write_text(not_found_html(), encoding="utf-8")
    write_sitemaps(site_dir, pages)
    validate(site_dir)


def validate(site_dir: Path) -> None:
    pages, redirects = inventory(site_dir)
    valid_urls = {page.url for page in pages.values()}
    for page in pages.values():
        source = (site_dir / page.path).read_text(encoding="utf-8")
        links = [attributes(match[0]) for match in LINK_RE.finditer(source)]
        canonical = [link.get("href") for link in links if link.get("rel") == "canonical"]
        if canonical != [page.url]:
            raise ValueError(f"Invalid canonical for {page.path}: {canonical}")
        actual_alternates = {link["hreflang"]: link.get("href") for link in links if "hreflang" in link}
        if actual_alternates != alternates(page, pages):
            raise ValueError(f"Invalid language alternates for {page.path}")
        for link in links:
            if link.get("rel") == "stylesheet":
                url = urlsplit(urljoin(page.url, link["href"]))
                if url.netloc == urlsplit(SITE).netloc and not (site_dir / url.path.lstrip("/")).is_file():
                    raise ValueError(f"Missing stylesheet for {page.path}: {url.path}")
    for path in redirects:
        source = (site_dir / path).read_text(encoding="utf-8")
        links = [attributes(match[0]) for match in LINK_RE.finditer(source)]
        targets = [link.get("href") for link in links if link.get("rel") == "canonical"]
        if len(targets) != 1:
            raise ValueError(f"Missing redirect canonical: {path}")
        target = targets[0]
        if urlsplit(target).netloc == urlsplit(SITE).netloc and target not in valid_urls:
            raise ValueError(f"Redirect does not terminate at a content page: {path} -> {target}")
    for file in site_dir.rglob("sitemap.xml"):
        tree = ET.fromstring(file.read_bytes())
        if tree.tag != f"{{{SITEMAP_NS}}}urlset":
            continue
        for loc in tree.findall(f".//{{{SITEMAP_NS}}}loc"):
            if loc.text not in valid_urls:
                raise ValueError(f"Sitemap contains a noncanonical page: {file}: {loc.text}")


def validate_build(site_dir: Path, public_url: str) -> None:
    """Check the native mike build before finalization can repair its metadata."""
    for file in site_dir.rglob("*.html"):
        source = file.read_text(encoding="utf-8")
        if REFRESH_RE.search(source):
            continue
        path = file.relative_to(site_dir).as_posix()
        links = [attributes(match[0]) for match in LINK_RE.finditer(source)]
        url = urljoin(public_url, path.removesuffix("index.html"))
        if file.name == "index.html":
            canonical = [link.get("href") for link in links if link.get("rel") == "canonical"]
            if canonical != [url]:
                raise ValueError(f"Build canonical mismatch for {path}: {canonical}, expected {url}")
        for link in links:
            if link.get("rel") != "stylesheet":
                continue
            stylesheet = urljoin(url, link["href"])
            if stylesheet.startswith(public_url) and not (site_dir / stylesheet.removeprefix(public_url)).is_file():
                raise ValueError(f"Build stylesheet missing for {path}: {stylesheet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-build-url")
    args = parser.parse_args()
    if args.check_build_url:
        validate_build(args.site_dir, args.check_build_url)
    elif args.check:
        validate(args.site_dir)
    else:
        finalize(args.site_dir)


if __name__ == "__main__":
    main()
