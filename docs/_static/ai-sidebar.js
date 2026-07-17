(function () {
  function cleanTitle(value) {
    return (value || "D-ACE documentation")
      .replace(/\s+#$/, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function pageTitle() {
    var heading = document.querySelector("h1");
    if (heading && heading.textContent) {
      return cleanTitle(heading.textContent);
    }
    return cleanTitle(document.title.replace(/[-\u2014]\s*D-ACE Documentation$/, ""));
  }

  function pageUrl() {
    var url = new URL(window.location.href);
    url.search = "";
    url.hash = "";
    return url.toString();
  }

  function promptText() {
    return [
      'Read this D-ACE documentation page about "' + pageTitle() + '".',
      "",
      "Read " + pageUrl(),
      "",
      "Summarize the key points, usage steps, examples, caveats, and best practices."
    ].join("\n");
  }

  function providerUrl(provider) {
    var prompt = promptText();
    var encoded = encodeURIComponent(prompt);

    if (provider === "openai") {
      return "https://chatgpt.com/?q=" + encoded + "&hints=search";
    }
    if (provider === "anthropic") {
      return "https://claude.ai/new?q=" + encoded;
    }
    if (provider === "google") {
      return "https://www.google.com/search?q=" + encoded + "&udm=50";
    }
    if (provider === "kimi") {
      return "https://www.kimi.com/_prefill_chat?force_search=true&prefill_prompt=" + encoded + "&send_immediately=true";
    }
    return pageUrl();
  }

  var currentUrl = new URL(window.location.href);
  var requestedProvider = currentUrl.searchParams.get("provider");
  if (requestedProvider) {
    window.location.replace(providerUrl(requestedProvider));
    return;
  }

  document.querySelectorAll("[data-ai-provider]").forEach(function (link) {
    var provider = link.getAttribute("data-ai-provider");
    var url = new URL(window.location.href);
    url.searchParams.set("provider", provider);
    link.setAttribute("href", url.toString());
  });
})();
