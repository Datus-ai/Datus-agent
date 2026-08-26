(function () {
  var GA_ID = "G-8HMXHZDV7F";
  var allowedHost = "docs.datus.ai";

  // Only load GA4 on the production docs host. This keeps localhost,
  // 127.0.0.1, and *.github.io preview/developer traffic out of GA4.
  if (window.location.hostname !== allowedHost) {
    return;
  }

  window.dataLayer = window.dataLayer || [];
  function gtag() {
    window.dataLayer.push(arguments);
  }
  window.gtag = gtag;

  var script = document.createElement("script");
  script.async = true;
  script.src = "https://www.googletagmanager.com/gtag/js?id=" + GA_ID;
  document.head.appendChild(script);

  gtag("js", new Date());
  gtag("config", GA_ID, {
    page_title: document.title,
    page_location: window.location.href,
    page_path: window.location.pathname + window.location.search
  });

  // Material for MkDocs uses instant navigation; re-send a page_view on each
  // client-side navigation so page_location / page_path stay accurate and the
  // "(not set)" landing-page rate stays low.
  if (window.document$ && window.location$) {
    window.location$.subscribe(function (url) {
      gtag("config", GA_ID, {
        page_title: document.title,
        page_location: window.location.href,
        page_path: url.pathname + url.search
      });
    });
  }
})();
