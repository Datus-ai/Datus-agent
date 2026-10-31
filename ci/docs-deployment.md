# Documentation publishing

The documentation lives in Datus-agent and is published to GitHub Pages with
MkDocs, Material, static-i18n and mike. The marketing website's blog is built
separately in Datus-website. Dosi uses its own Cloudflare Pages deployment in
osi-engine.

## URL contract

- English content: `https://docs.datus.ai/<version>/<page>/`.
- Chinese content: `https://docs.datus.ai/zh/<version>/<page>/`.
- Releases update their minor line (`v0.4.4` publishes `0.4`). Main publishes
  `dev`. The newest stable release owns `latest`.
- Pass the **locale root**, without a version, as MkDocs `site_url`. mike adds
  `MIKE_DOCS_VERSION` itself. For Chinese, `--deploy-prefix zh` controls file
  placement and the locale-root `site_url` controls public URLs.

The October 2026 GSC incident came from passing an already versioned `site_url`
to mike: valid pages advertised nonexistent `/0.4/0.4/...` canonicals and
sitemap entries. An obsolete root `404.html` also referenced assets no longer
present at the root. Some unversioned redirects pointed to pages removed from
the latest release, and older Chinese URLs used version-first prefixes.

## One complete publication

`deploy-docs.yml` serializes deployments and performs these steps:

1. Export the requested source ref with the current build infrastructure.
2. Strictly build each locale with the same `MIKE_DOCS_VERSION` environment that
   mike uses. Validate native canonicals and stylesheets before publishing.
3. Stage both mike builds, aliases and version visibility locally, without
   pushing either locale separately.
4. Run `ci/docs_site.py` on the complete staged `gh-pages` tree. Its inventory
   includes actual current and archived content, and distinguishes redirect
   stubs from content. It writes self-canonicals, bidirectional per-page
   hreflang only for existing translations, compressed sitemaps, a root sitemap
   index, robots.txt, and an asset-independent root 404 page. Hidden archives and
   dev are not advertised in the stable sitemap index.
5. Validate the complete tree, then push once. Invalid canonicals, language
   alternates, stylesheet paths, redirect destinations or sitemap entries stop
   publication.

Finalization is idempotent and also corrects already published metadata, so a
main deployment repairs older stable/archived builds without rebuilding them
from a newer release's content. It preserves page bodies and does not push to
production when invoked locally.

## Adding, moving and removing pages

**New pages require no redirect/sitemap registration.** Add the Markdown and
navigation entries as usual. After the containing version is deployed, its
actual files automatically supply canonical URLs, translated peers, sitemap
entries, unversioned entry points, and latest/legacy-prefix compatibility URLs.
A page added only to dev is available there; publishing a stable release still
requires the normal release process.

**Renames require explicit intent.** Add a source-slug to destination-slug entry
in `ci/docs_redirects.yml` when moving or consolidating a page. The pipeline can
discover that a file exists, but cannot infer which new subject replaces an old
one. This small manifest is independent of any GSC export.

Compatibility URLs resolve directly to content in the requested locale/version
when available, then the latest stable version, other stable archives and dev.
For example, the retired MetricFlow introduction resolves to the existing 0.3
article, rather than the nonexistent latest page. A semantic rename is used
when the original slug has no surviving content. Query strings and fragments
are preserved. Unknown URLs without an equivalent remain HTTP 404; they are
never redirected wholesale to the homepage.

GitHub Pages cannot configure server-side 301s from these artifacts. Known old
URLs use immediate meta-refresh redirects with a canonical and a JavaScript
fallback. Redirect-source exclusions in GSC are expected. A future hosting
migration can reuse the route inventory to emit server-side redirects.

## Validation and rollout

PRs touching documentation run the native bilingual builds and publishing
contract tests. The tests also add a new nested bilingual page after an initial
publication and verify that routes, language alternates and sitemaps update
without manual registration. Native mike regression tests fail with the former
versioned `site_url` workflow.

After merging, wait for Deploy Docs and GitHub Pages to finish. Check both
locales, the root 404, an old double-version URL and the MetricFlow redirect.
Submit `https://docs.datus.ai/sitemap.xml` in GSC, inspect representative final
URLs, then validate the 404/redirect-error fixes. Indexing reports lag deployment
and legitimate canonical, redirect and noindex exclusions need not disappear.
