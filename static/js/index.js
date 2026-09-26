function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth",
  });
}

function copyBibTeX() {
  const citationElement = document.getElementById("bibtex-code");
  const citationText = citationElement ? citationElement.textContent : "";
  const button = document.querySelector(".copy-bibtex-btn");
  const buttonText = button ? button.querySelector(".copy-text") : null;
  const originalText = buttonText ? buttonText.textContent : "";

  const showResult = () => {
    if (!button || !buttonText) return;
    button.classList.add("copied");
    buttonText.textContent = "Copied";
    window.setTimeout(() => {
      button.classList.remove("copied");
      buttonText.textContent = originalText;
    }, 2000);
  };

  const copyWithTextarea = () => {
    const temporaryTextarea = document.createElement("textarea");
    temporaryTextarea.value = citationText;
    temporaryTextarea.setAttribute("readonly", "");
    temporaryTextarea.style.position = "absolute";
    temporaryTextarea.style.left = "-9999px";
    document.body.appendChild(temporaryTextarea);
    temporaryTextarea.select();
    document.execCommand("copy");
    document.body.removeChild(temporaryTextarea);
    showResult();
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(citationText).then(showResult).catch(copyWithTextarea);
    return;
  }
  copyWithTextarea();
}

window.addEventListener("scroll", () => {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (!scrollButton) return;
  scrollButton.classList.toggle("visible", window.scrollY > 300);
});
