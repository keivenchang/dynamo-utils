"""
Shared HTML/CSS/JS snippets to keep UI styling consistent across scripts.

These scripts currently generate standalone HTML without a bundler, so the safest
way to ensure styling stays in-sync is to centralize shared snippets here and
reuse them at render-time.
"""

# Style for the optional pass count in the "REQ+OPT✓" compact CI summary.
# Keep this in sync across show_dynamo_branches.py and show_commit_history.j2.
PASS_PLUS_STYLE = "font-size: 10px; font-weight: 600; opacity: 0.9;"

# Tooltip styling for GitHub check summaries. This is embedded into <style>.
# Matches the behavior and look from show_commit_history.j2.
GH_STATUS_TOOLTIP_CSS = """
.gh-status-tooltip {
  position: relative;
  display: inline-block;
}
.gh-status-tooltip .tooltiptext {
  visibility: hidden;
  max-width: calc(100vw - 40px);
  background-color: #24292f;
  color: #ffffff;
  text-align: left;
  border-radius: 6px;
  padding: 8px 12px;
  position: fixed;
  z-index: 1000;
  left: 20px;
  right: 20px;
  width: auto;
  opacity: 0;
  transition: opacity 0.3s, visibility 0s 1s;
  font-size: 12px;
  line-height: 1.5;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
  white-space: normal;
  word-wrap: break-word;
  overflow-wrap: break-word;
}
.gh-status-tooltip .tooltiptext::after {
  content: "";
  position: absolute;
  top: 100%;
  left: 20px;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: #24292f transparent transparent transparent;
}
.gh-status-tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
  transition-delay: 1s;
}
""".strip()

# Tooltip positioning JS. Embed into an existing <script> tag (no <script> wrapper).
# It positions the tooltip above the hovered element.
GH_STATUS_TOOLTIP_JS = """
// Position tooltips dynamically (shared)
document.addEventListener('DOMContentLoaded', function() {
  var tooltips = document.querySelectorAll('.gh-status-tooltip');
  tooltips.forEach(function(tooltip) {
    tooltip.addEventListener('mouseenter', function() {
      var tooltipText = this.querySelector('.tooltiptext');
      if (!tooltipText) return;
      var rect = this.getBoundingClientRect();
      // Position tooltip above the element
      tooltipText.style.top = (rect.top - tooltipText.offsetHeight - 10) + 'px';
    });
  });
});
""".strip()


