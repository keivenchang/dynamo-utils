#!/usr/bin/env python3
"""
CI Graph Visualization - View CI jobs as a DAG (Directed Acyclic Graph)

This module treats CI workflows as graphs rather than trees, allowing nodes
to have multiple parents (e.g., changed-files is needed by many jobs).
"""

import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    """A node in the CI dependency graph."""
    job_id: str  # Unique identifier (e.g., "vllm (amd64)")
    display_name: str  # Human-readable name
    status: str = "unknown"  # success, failure, pending, running, etc.
    parents: Set[str] = field(default_factory=set)  # job_ids that depend on this node
    children: Set[str] = field(default_factory=set)  # job_ids this node depends on
    workflow_file: str = ""  # Which workflow defines this job
    is_real_ci: bool = False  # True if from actual GitHub CI, False if from YAML only
    log_url: str = ""
    duration: str = ""
    is_required: bool = False


class CIGraph:
    """Represents the CI workflow as a directed acyclic graph."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}  # job_id -> GraphNode
        self.yaml_parent_child: Dict[str, List[str]] = {}  # YAML needs: relationships
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes[node.job_id] = node
    
    def add_edge(self, parent_id: str, child_id: str):
        """Add a directed edge: parent depends on child (parent -> child)."""
        if parent_id in self.nodes:
            self.nodes[parent_id].children.add(child_id)
        if child_id in self.nodes:
            self.nodes[child_id].parents.add(parent_id)
    
    def get_root_nodes(self) -> List[GraphNode]:
        """Get nodes with no children (leaves that don't depend on anything)."""
        return [node for node in self.nodes.values() if not node.children]
    
    def get_leaf_nodes(self) -> List[GraphNode]:
        """Get nodes with no parents (roots that nothing depends on)."""
        return [node for node in self.nodes.values() if not node.parents]
    
    def topological_sort(self) -> List[str]:
        """Return nodes in topological order (children before parents).
        
        Uses Kahn's algorithm for topological sorting.
        Returns list of job_ids in order where all dependencies appear before dependents.
        """
        # Count in-degree (number of children) for each node
        in_degree = {job_id: len(node.children) for job_id, node in self.nodes.items()}
        
        # Start with nodes that have no children (leaves)
        queue = [job_id for job_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # For each parent of current node
            if current in self.nodes:
                for parent_id in self.nodes[current].parents:
                    in_degree[parent_id] -= 1
                    if in_degree[parent_id] == 0:
                        queue.append(parent_id)
        
        return result
    
    def get_layers(self) -> List[List[str]]:
        """Group nodes into layers for top-down visualization.
        
        Correct layering:
        - Standalone: Nodes with no parents AND no children (separate section)
        - Layer 0 (TOP): Nodes with no parents but have children (root/entry points)
        - Layer N (BOTTOM): Nodes with no children (leaves/dependencies)
        - Middle layers: Based on distance from bottom (leaves)
        
        Algorithm:
        1. Leaves (no children) are at layer 0 (bottom)
        2. Each parent is at max(child_layers) + 1
        3. Roots (no parents) end up at the top
        """
        # Separate standalone nodes (no connections)
        standalone_nodes = [
            job_id for job_id, node in self.nodes.items()
            if not node.parents and not node.children
        ]
        
        # Calculate layers for connected nodes
        node_layer: Dict[str, int] = {}
        
        def calculate_layer(job_id: str) -> int:
            """Calculate layer from bottom up (leaves = 0)."""
            if job_id in node_layer:
                return node_layer[job_id]
            
            node = self.nodes.get(job_id)
            if not node:
                node_layer[job_id] = 0
                return 0
            
            # Leaf node (no children) is at layer 0 (bottom)
            if not node.children:
                node_layer[job_id] = 0
                return 0
            
            # Non-leaf: layer is 1 + max layer of all children
            max_child_layer = max(calculate_layer(child_id) for child_id in node.children)
            node_layer[job_id] = max_child_layer + 1
            return node_layer[job_id]
        
        # Calculate layers for all connected nodes
        connected_nodes = [
            job_id for job_id in self.nodes
            if job_id not in standalone_nodes
        ]
        
        for job_id in connected_nodes:
            calculate_layer(job_id)
        
        # Group nodes by layer (bottom to top)
        max_layer = max(node_layer.values()) if node_layer else 0
        layers = [[] for _ in range(max_layer + 1)]
        for job_id, layer in node_layer.items():
            layers[layer].append(job_id)
        
        # Reverse so Layer 0 is at top (roots), and last layer is at bottom (leaves)
        layers.reverse()
        
        return layers, standalone_nodes


def parse_workflow_yaml_to_graph(repo_root: Path) -> CIGraph:
    """Parse workflow YAML files and build a graph of job dependencies."""
    graph = CIGraph()
    workflows_dir = repo_root / ".github" / "workflows"
    
    if not workflows_dir.exists():
        print(f"[CI Graph] No workflows directory found at {workflows_dir}")
        return graph
    
    for workflow_file in workflows_dir.glob("*.yml"):
        with open(workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        if not workflow_data or 'jobs' not in workflow_data:
            continue
        
        jobs = workflow_data.get('jobs', {})
        for job_id, job_data in jobs.items():
            if not isinstance(job_data, dict):
                continue
            
            job_name = job_data.get('name', job_id)
            needs = job_data.get('needs', [])
            if isinstance(needs, str):
                needs = [needs]
            
            # Create node for this job
            node = GraphNode(
                job_id=job_name,
                display_name=job_name,
                workflow_file=workflow_file.name,
            )
            graph.add_node(node)
            
            # Store YAML relationships
            graph.yaml_parent_child[job_name] = needs
            
            # Add edges for dependencies
            for child_name in needs:
                # Create child node if it doesn't exist
                if child_name not in graph.nodes:
                    child_node = GraphNode(
                        job_id=child_name,
                        display_name=child_name,
                        workflow_file=workflow_file.name,
                    )
                    graph.add_node(child_node)
                
                # Add edge: this job (parent) depends on child
                graph.add_edge(job_name, child_name)
    
    print(f"[CI Graph] Parsed {len(graph.nodes)} nodes from YAML")
    return graph


def render_graph_html(graph: CIGraph, pr_number: int) -> str:
    """Render the CI graph as interactive HTML with clickable nodes and dependency lines.
    
    Layout:
    - Standalone section: Nodes with no connections
    - Layer 0 (TOP): Root nodes (no parents, have children) - entry points
    - Layer N (BOTTOM): Leaf nodes (no children) - dependencies like changed-files
    - Lines connect parents to children (parent -> child)
    """
    layers, standalone_nodes = graph.get_layers()
    
    # Assign positions to each node for line drawing
    node_positions: Dict[str, Tuple[int, int]] = {}  # job_id -> (x, y)
    layer_height = 150  # px between layers
    node_width = 180
    node_spacing = 20
    
    html_parts = []
    html_parts.append(f'<div class="ci-graph-container" data-pr="{pr_number}">')
    html_parts.append('<style>')
    html_parts.append('''
        .ci-graph-container {
            padding: 20px;
            background: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            margin: 20px 0;
            position: relative;
        }
        .ci-graph-svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }
        .ci-graph-layer {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
            position: relative;
            z-index: 2;
        }
        .ci-graph-node {
            border: 2px solid #d0d7de;
            border-radius: 6px;
            padding: 12px 16px;
            background: #f6f8fa;
            cursor: pointer;
            transition: all 0.2s;
            min-width: 150px;
            text-align: center;
            position: relative;
        }
        .ci-graph-node:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
            z-index: 10;
        }
        .ci-graph-node.status-success {
            border-color: #2da44e;
            background: #dafbe1;
        }
        .ci-graph-node.status-failure {
            border-color: #c83a3a;
            background: #ffebe9;
        }
        .ci-graph-node.status-pending {
            border-color: #bf8700;
            background: #fff8c5;
        }
        .ci-graph-node.status-running {
            border-color: #0969da;
            background: #ddf4ff;
        }
        .ci-graph-node-name {
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 4px;
        }
        .ci-graph-node-status {
            font-size: 11px;
            color: #656d76;
        }
        .ci-graph-layer-label {
            text-align: center;
            font-size: 12px;
            color: #656d76;
            font-weight: 600;
            margin: 10px 0;
        }
        .dependency-line {
            stroke: #d0d7de;
            stroke-width: 2;
            fill: none;
            marker-end: url(#arrowhead);
        }
        .dependency-line:hover {
            stroke: #0969da;
            stroke-width: 3;
        }
    ''')
    html_parts.append('</style>')
    
    # SVG for drawing lines (will be positioned absolutely)
    html_parts.append('<svg class="ci-graph-svg" id="dependency-svg">')
    # Define arrowhead marker
    html_parts.append('''
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#d0d7de" />
            </marker>
        </defs>
    ''')
    html_parts.append('</svg>')
    
    # Render standalone nodes first (no connections)
    if standalone_nodes:
        html_parts.append('<div class="ci-graph-layer-label">Standalone (No Dependencies)</div>')
        html_parts.append('<div class="ci-graph-layer">')
        
        for job_id in sorted(standalone_nodes):
            node = graph.nodes[job_id]
            status_class = f"status-{node.status}" if node.status != "unknown" else ""
            
            html_parts.append(f'<div class="ci-graph-node {status_class}" id="node-{job_id.replace(" ", "-")}" data-job-id="{job_id}">')
            html_parts.append(f'  <div class="ci-graph-node-name">{node.display_name}</div>')
            html_parts.append(f'  <div class="ci-graph-node-status">{node.status}</div>')
            html_parts.append('  <div style="font-size: 10px; color: #8c959f;">⊗ isolated</div>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    # Render connected layers (top = roots, bottom = leaves)
    for layer_idx, layer in enumerate(layers):
        html_parts.append(f'<div class="ci-graph-layer-label">Layer {layer_idx}</div>')
        html_parts.append(f'<div class="ci-graph-layer" id="layer-{layer_idx}">')
        
        for node_idx, job_id in enumerate(sorted(layer)):
            node = graph.nodes[job_id]
            status_class = f"status-{node.status}" if node.status != "unknown" else ""
            
            html_parts.append(f'<div class="ci-graph-node {status_class}" id="node-{job_id.replace(" ", "-")}" data-job-id="{job_id}">')
            html_parts.append(f'  <div class="ci-graph-node-name">{node.display_name}</div>')
            html_parts.append(f'  <div class="ci-graph-node-status">{node.status}</div>')
            if node.children:
                html_parts.append(f'  <div style="font-size: 10px; color: #8c959f;">↓ depends on: {len(node.children)}</div>')
            if node.parents:
                html_parts.append(f'  <div style="font-size: 10px; color: #8c959f;">↑ depended by: {len(node.parents)}</div>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    # JavaScript to draw lines after DOM is ready
    html_parts.append('''
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        drawDependencyLines();
        
        // Redraw on window resize
        window.addEventListener('resize', drawDependencyLines);
    });
    
    function drawDependencyLines() {
        const svg = document.getElementById('dependency-svg');
        if (!svg) return;
        
        // Clear existing lines
        const existingLines = svg.querySelectorAll('.dependency-line');
        existingLines.forEach(line => line.remove());
        
        // Get all nodes
        const nodes = document.querySelectorAll('.ci-graph-node');
        const nodePositions = new Map();
        
        // Calculate center position for each node
        nodes.forEach(node => {
            const rect = node.getBoundingClientRect();
            const containerRect = svg.parentElement.getBoundingClientRect();
            const jobId = node.getAttribute('data-job-id');
            
            nodePositions.set(jobId, {
                x: rect.left - containerRect.left + rect.width / 2,
                y: rect.top - containerRect.top + rect.height / 2,
                top: rect.top - containerRect.top,
                bottom: rect.top - containerRect.top + rect.height,
            });
        });
        
        // Draw lines for dependencies (parent -> child)
        nodes.forEach(node => {
            const jobId = node.getAttribute('data-job-id');
            const parentPos = nodePositions.get(jobId);
            
            if (!parentPos) return;
            
            // Get children from the graph data (we'll embed it in the HTML)
            const graphData = window.ciGraphData;
            if (!graphData || !graphData[jobId]) return;
            
            const children = graphData[jobId].children || [];
            
            children.forEach(childId => {
                const childPos = nodePositions.get(childId);
                if (!childPos) return;
                
                // Draw a curved line from parent (bottom) to child (top)
                const x1 = parentPos.x;
                const y1 = parentPos.bottom;
                const x2 = childPos.x;
                const y2 = childPos.top;
                
                // Control points for bezier curve
                const midY = (y1 + y2) / 2;
                
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('class', 'dependency-line');
                path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
                path.setAttribute('data-parent', jobId);
                path.setAttribute('data-child', childId);
                
                svg.appendChild(path);
            });
        });
    }
    </script>
    ''')
    
    # Embed graph data as JSON for JavaScript
    graph_data_json = {}
    for job_id, node in graph.nodes.items():
        graph_data_json[job_id] = {
            'children': list(node.children),
            'parents': list(node.parents),
        }
    
    import json
    html_parts.append(f'<script>window.ciGraphData = {json.dumps(graph_data_json)};</script>')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)


if __name__ == "__main__":
    # Test with dynamo repo
    repo_root = Path("/home/keivenc/dynamo/dynamo_latest")
    graph = parse_workflow_yaml_to_graph(repo_root)
    
    print(f"\nTotal nodes: {len(graph.nodes)}")
    print(f"Root nodes (no parents): {[n.job_id for n in graph.get_leaf_nodes()]}")
    print(f"Leaf nodes (no children): {[n.job_id for n in graph.get_root_nodes()]}")
    
    layers, standalone_nodes = graph.get_layers()
    print(f"\nGraph has {len(layers)} connected layers + {len(standalone_nodes)} standalone nodes:")
    for i, layer in enumerate(layers):
        layer_type = "TOP (roots)" if i == 0 else "BOTTOM (leaves)" if i == len(layers) - 1 else "Middle"
        print(f"  Layer {i} ({layer_type}): {len(layer)} nodes - {layer[:5]}{'...' if len(layer) > 5 else ''}")
    if standalone_nodes:
        print(f"  Standalone: {len(standalone_nodes)} nodes - {standalone_nodes[:5]}{'...' if len(standalone_nodes) > 5 else ''}")
    
    # Generate HTML
    html = render_graph_html(graph, pr_number=12345)
    output_file = Path("/home/keivenc/dynamo/speedoflight/users/keivenchang/ci-graph-test.html")
    output_file.write_text(html, encoding="utf-8")
    print(f"\nGenerated test HTML: {output_file}")
