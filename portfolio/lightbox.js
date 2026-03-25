(function () {
  const root = document.getElementById("lightbox");
  if (!root) return;

  const img = root.querySelector(".lightbox-img");
  const swatch = root.querySelector(".lightbox-swatch");

  function close() {
    root.classList.remove("is-open");
    root.setAttribute("aria-hidden", "true");
    img.removeAttribute("src");
    img.alt = "";
    img.setAttribute("hidden", "");
    if (swatch) {
      swatch.setAttribute("hidden", "");
      swatch.className = "lightbox-swatch";
    }
    document.body.style.overflow = "";
  }

  function openSwatch(trigger) {
    const kind = trigger.getAttribute("data-swatch") || "thumb";
    img.setAttribute("hidden", "");
    if (!swatch) return;
    swatch.className = "lightbox-swatch lightbox-swatch--" + kind;
    swatch.removeAttribute("hidden");
  }

  function openImage(trigger) {
    const full =
      trigger.getAttribute("data-full") ||
      trigger.querySelector("img")?.currentSrc ||
      trigger.querySelector("img")?.src;
    if (!full) return false;
    const thumb = trigger.querySelector("img");
    if (swatch) swatch.setAttribute("hidden", "");
    img.removeAttribute("hidden");
    img.src = full;
    img.alt = thumb?.alt || "";
    return true;
  }

  function open(trigger) {
    if (!openImage(trigger)) {
      openSwatch(trigger);
    }
    root.classList.add("is-open");
    root.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  document.body.addEventListener("click", function (e) {
    const trigger = e.target.closest(".piece-hit");
    if (trigger) {
      e.preventDefault();
      open(trigger);
      return;
    }
    if (e.target.closest("[data-lightbox-close]")) {
      close();
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && root.classList.contains("is-open")) {
      close();
    }
  });
})();
