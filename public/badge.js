// "Made by Kiro" badge — injected into the Chainlit UI
(function () {
  const badge = document.createElement("a");
  badge.className = "kiro-badge";
  badge.href = "https://kiro.dev";
  badge.target = "_blank";
  badge.rel = "noopener noreferrer";
  badge.title = "Built with Kiro — AI-powered development environment";
  badge.innerHTML =
    '<span class="kiro-icon">👻</span>' +
    '<span class="kiro-text">Made by Kiro</span>';
  document.body.appendChild(badge);
})();
