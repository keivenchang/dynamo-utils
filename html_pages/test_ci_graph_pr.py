#!/usr/bin/env python3
"""
Test CI Graph View for a single PR - displays CI jobs as a clickable DAG
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import GitHubAPIClient
from html_pages.ci_graph_view import CIGraph, GraphNode, parse_workflow_yaml_to_graph, render_graph_html

def main():
    # Test with keivenchang's first PR
    github_api = GitHubAPIClient()
    repo_root = Path("/home/keivenc/dynamo/dynamo_latest")
    
    # Get PRs for user
    prs = github_api.get_open_pr_info_for_author(
        owner="ai-dynamo",
        repo="dynamo",
        author="keivenchang",
        max_prs=5,
    )
    
    if not prs:
        print("No PRs found for keivenchang")
        return
    
    # Use first PR
    pr = prs[0]
    print(f"\n[CI Graph] Processing PR #{pr.number}: {pr.title}")
    print(f"[CI Graph] Branch: {pr.head_ref}, Commit: {pr.head_sha[:8] if pr.head_sha else 'N/A'}")
    
    # Parse workflow YAML to get graph structure
    graph = parse_workflow_yaml_to_graph(repo_root)
    print(f"[CI Graph] Parsed {len(graph.nodes)} nodes from YAML workflows")
    
    # TODO: Overlay actual CI data from GitHub API
    # For now, just show the YAML structure
    print(f"\n[CI Graph] Showing YAML workflow structure (no live CI data yet)")
    
    layers, standalone_nodes = graph.get_layers()
    print(f"[CI Graph] Graph has {len(layers)} layers + {len(standalone_nodes)} standalone nodes")
    
    # Print layer info
    for i, layer in enumerate(layers):
        layer_desc = "TOP (roots - no parents)" if i == 0 else f"BOTTOM (leaves - no children)" if i == len(layers) - 1 else f"Middle"
        print(f"  Layer {i} ({layer_desc}): {len(layer)} nodes")
    
    # Generate HTML
    html = render_graph_html(graph, pr_number=pr.number)
    
    # Wrap in a full HTML page
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CI Graph - PR #{pr.number}</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; margin: 20px;">
    <h1>CI Dependency Graph - PR #{pr.number}</h1>
    <h2>{pr.title}</h2>
    <p>Branch: <code>{pr.head_ref or 'N/A'}</code> | Commit: <code>{pr.head_sha[:8] if pr.head_sha else 'N/A'}</code></p>
    
    {html}
    
    <script>
    // Add click handlers to nodes
    document.querySelectorAll('.ci-graph-node').forEach(node => {{
        node.addEventListener('click', function() {{
            const jobId = this.getAttribute('data-job-id');
            alert('Clicked: ' + jobId);
            // TODO: Show job details, logs, dependencies, etc.
        }});
    }});
    </script>
</body>
</html>"""
    
    output_file = Path("/home/keivenc/dynamo/speedoflight/users/keivenchang/ci-graph-pr.html")
    output_file.write_text(full_html, encoding="utf-8")
    print(f"\n[CI Graph] Generated HTML: {output_file}")

if __name__ == "__main__":
    main()
