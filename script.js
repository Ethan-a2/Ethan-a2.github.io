const config = window.siteConfig ?? {};

const externalAttributes = (item) =>
  item.external || /^https?:\/\//.test(item.href)
    ? ' target="_blank" rel="noreferrer"'
    : "";

const escapeHtml = (value = "") =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const colorMap = {
  cyan: "#48e5ff",
  violet: "#9f7cff",
  green: "#60f7ad",
  orange: "#ffb454",
};

document.querySelectorAll("[data-year]").forEach((element) => {
  element.textContent = new Date().getFullYear();
});

document.querySelectorAll("[data-owner-intro]").forEach((element) => {
  if (config.owner?.intro) {
    element.textContent = config.owner.intro;
  }
});

document.querySelectorAll("[data-current-path]").forEach((element) => {
  element.textContent = window.location.pathname;
});

document.querySelectorAll("[data-nav]").forEach((nav) => {
  nav.innerHTML = (config.navigation ?? [])
    .map(
      (item) =>
        `<a href="${escapeHtml(item.href)}"${externalAttributes(item)}>${escapeHtml(
          item.label,
        )}</a>`,
    )
    .join("");
});

document.querySelectorAll("[data-quick-links]").forEach((container) => {
  container.innerHTML = (config.quickLinks ?? [])
    .map((item) => {
      const accent = colorMap[item.accent] ?? colorMap.cyan;

      return `<a class="card" style="--accent: ${accent}" href="${escapeHtml(
        item.href,
      )}"${externalAttributes(item)}>
        <span class="card-label">${escapeHtml(item.label)}</span>
        <h3>${escapeHtml(item.title)}</h3>
        <p>${escapeHtml(item.description)}</p>
      </a>`;
    })
    .join("");
});

document.querySelectorAll("[data-portfolio]").forEach((container) => {
  container.innerHTML = (config.portfolio ?? [])
    .map(
      (item) => `<a class="portfolio-item" href="${escapeHtml(item.href)}"${externalAttributes(
        item,
      )}>
        <span class="portfolio-type">${escapeHtml(item.type)}</span>
        <span>
          <h3>${escapeHtml(item.name)}</h3>
          <p>${escapeHtml(item.description)}</p>
        </span>
        <span class="portfolio-arrow" aria-hidden="true">↗</span>
      </a>`,
    )
    .join("");
});
