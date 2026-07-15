import ast
import time
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import os
import sys
import math
import argparse
from datetime import datetime
import shutil

def ensure_results_dir(subfolder=None):
    """Create and return path to results directory."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if subfolder:
        base_dir = os.path.join(base_dir, subfolder)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def generate_output_filename(base_name, suffix="multi_file_call_graph", results_dir=None):
    """Generate a timestamped filename in the results folder."""
    if results_dir is None:
        results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{suffix}_{timestamp}.pdf"
    return os.path.join(results_dir, filename)

def cleanup_results(days=7, results_dir=None):
    """Delete result files older than N days."""
    if results_dir is None:
        results_dir = ensure_results_dir()
    
    if not os.path.exists(results_dir):
        print(f"📁 Results directory not found: {results_dir}")
        return 0
    
    cutoff = time.time() - (days * 86400)
    count = 0
    
    for filename in os.listdir(results_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(results_dir, filename)
            try:
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    count += 1
                    print(f"  🗑️  Removed: {filename}")
            except (OSError, PermissionError) as e:
                print(f"  ⚠️  Could not remove {filename}: {e}")
    
    if count:
        print(f"🧹 Cleaned up {count} old result files (> {days} days old)")
    else:
        print("🧹 No old result files to clean up")
    
    return count

class Pass1Visitor(ast.NodeVisitor):
    def __init__(self, analyzer, filename):
        self.analyzer = analyzer
        self.filename = filename
        self.current_class = None

    def visit_ClassDef(self, node):
        node_id = f"{self.filename}::{node.name}"
        self.analyzer.defined_classes.add(node_id)
        self.analyzer.class_to_file[node.name] = self.filename
        self.analyzer.file_clusters[self.filename].append(node_id)
        self.analyzer.graph.add_node(node_id, label=node.name, node_type='class')
        print(f"[DEBUG-PASS-1] Cataloged Class: {node_id}")

        # Keep track of context so child functions know they are methods
        prev_class = self.current_class
        self.current_class = node_id
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        if self.current_class:
            # It's a method: scope the node id to its owning class so that
            # same-named methods in different classes (e.g. visit_ClassDef
            # in two visitor classes in the same file) don't collide into
            # a single shared graph node.
            class_name = self.current_class.split("::", 1)[1]
            node_id = f"{self.filename}::{class_name}.{node.name}"
            self.analyzer.methods_by_class.setdefault(self.current_class, {})[node.name] = node_id
            # Store the full node id here (not just the filename) since
            # method node ids are class-scoped, unlike function_to_file.
            self.analyzer.method_to_file[node.name] = node_id
        else:
            node_id = f"{self.filename}::{node.name}"
            self.analyzer.defined_functions.add(node_id)
            self.analyzer.function_to_file[node.name] = self.filename

        self.analyzer.file_clusters[self.filename].append(node_id)
        self.analyzer.graph.add_node(node_id, label=node.name, node_type='function')
        print(f"[DEBUG-PASS-1] Cataloged Function/Method: {node_id}")

        if self.current_class:
            # Solid line establishing ownership from Class to Method
            self.analyzer.graph.add_edge(self.current_class, node_id, edge_type='owns')

        self.generic_visit(node)

class FolderGraphAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_clusters = {}
        self.current_file = None
        self.current_function = None
        self.current_class = None
        self.defined_functions = set()
        self.defined_classes = set()
        self.function_to_file = {}
        self.class_to_file = {}
        # class_node_id -> {method_name: method_node_id}
        # Lets us resolve self.method() calls against the caller's own
        # class instead of a global bare-name dict (which collides when
        # two classes define same-named methods, e.g. visit_ClassDef).
        self.methods_by_class = {}
        # bare method name -> file. Best-effort fallback used ONLY for
        # calls we can't attribute to a specific class (e.g.
        # self.engine.load_data(), external_obj.method()), since we have
        # no type inference to know what 'engine' or 'external_obj' is.
        # Last definition wins if the same method name exists in multiple
        # classes across the project - same ambiguity a human skimming
        # the code would have without running it.
        self.method_to_file = {}

    def analyze_folder(self, folder_path):
        print(f"Crawling directory: {folder_path} ...")
        py_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))

        # PASS 1: Catalog all definitions so we can link cross-file calls
        # We loop through EVERY file before doing any edge tracking.
        for file_path in py_files:
            filename = os.path.basename(file_path)
            self.file_clusters[filename] = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    # Use the new hierarchical visitor to track classes and methods
                    visitor = Pass1Visitor(self, filename)
                    visitor.visit(tree)
            except Exception as e:
                print(f"Failed to parse {filename}: {e}")

        # PASS 2: Map the calls and build the edges
        for file_path in py_files:
            filename = os.path.basename(file_path)
            self.current_file = filename
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    self.visit(tree)
            except Exception:
                pass

        self._filter_dunder_init()

    def _filter_dunder_init(self):
        """Drop __init__ nodes from the graph/clusters to reduce clutter.

        Deleting a node also deletes its incident edges, which would
        silently erase real architecture - e.g. 'self.engine =
        AnalyticsEngine()' living inside __init__ is how GUI.py actually
        connects to Methods.py. So before removing each __init__ node, we
        re-parent its non-'owns' edges onto the class that owns it, then
        delete the node itself.
        """
        init_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('label') == '__init__']
        if not init_nodes:
            return
        print(f"[DEBUG-FILTER] Removing {len(init_nodes)} __init__ node(s): {init_nodes}")

        for init_node in init_nodes:
            owner = None
            for pred in self.graph.predecessors(init_node):
                if self.graph[pred][init_node].get('edge_type') == 'owns':
                    owner = pred
                    break
            if owner is None:
                continue

            for succ in list(self.graph.successors(init_node)):
                edge_type = self.graph[init_node][succ].get('edge_type')
                if edge_type != 'owns' and succ != owner:
                    print(f"[DEBUG-FILTER]   Re-parenting edge: {init_node} -> {succ} onto {owner}")
                    self._safe_add_edge(owner, succ, edge_type=edge_type)

            for pred in list(self.graph.predecessors(init_node)):
                edge_type = self.graph[pred][init_node].get('edge_type')
                if edge_type != 'owns' and pred != owner:
                    print(f"[DEBUG-FILTER]   Re-parenting edge: {pred} -> {init_node} onto {pred} -> {owner}")
                    self._safe_add_edge(pred, owner, edge_type=edge_type)

        self.graph.remove_nodes_from(init_nodes)
        for filename, nodes in self.file_clusters.items():
            self.file_clusters[filename] = [n for n in nodes if n not in init_nodes]

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = f"{self.current_file}::{node.name}"
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        prev_func = self.current_function
        if self.current_class:
            class_name = self.current_class.split("::", 1)[1]
            self.current_function = f"{self.current_file}::{class_name}.{node.name}"
        else:
            self.current_function = f"{self.current_file}::{node.name}"
        print(f"\n[DEBUG-PASS-2] ---> Entering Function: {self.current_function}")
        self.generic_visit(node)
        self.current_function = prev_func

    def _safe_add_edge(self, source, target, edge_type):
        """Prevents standard calls from overwriting detected callbacks."""
        if self.graph.has_edge(source, target):
            existing_type = self.graph[source][target].get('edge_type')
            # If we already know it's a reference/callback, keep it that way!
            if existing_type == 'reference' and edge_type in ['call', 'method']:
                return
        self.graph.add_edge(source, target, edge_type=edge_type)

    def visit_Call(self, node):
        if self.current_function:
            called_func = None
            edge_style = 'call'
            
            # Handle standard calls: method()
            is_self_call = False
            if isinstance(node.func, ast.Name):
                called_func = node.func.id
                edge_style = 'call'
            # Handle class/module calls: object.method()
            elif isinstance(node.func, ast.Attribute):
                called_func = node.func.attr
                # Differentiate between 'self.method()' and 'external_obj.method()'
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                    edge_style = 'call'
                    is_self_call = True
                else:
                    edge_style = 'external_method'
            
            print(f"[DEBUG-AST] Found Call in {self.current_function}: target='{called_func}'")
            
            # If the called function or class exists anywhere in our parsed directory, link it!
            if called_func:
                target_node_id = None
                
                # Resolve self.method() against the CALLER's own class first.
                # This must come before the global bare-name lookups below,
                # otherwise same-named methods in other classes (e.g. two
                # visitor classes both defining visit_ClassDef) hijack the
                # edge, since function_to_file only stores one file/target
                # per bare method name.
                if is_self_call and self.current_class:
                    class_methods = self.methods_by_class.get(self.current_class, {})
                    if called_func in class_methods:
                        target_node_id = class_methods[called_func]

                # Check if it's a class instantiation
                if target_node_id is None and called_func in self.class_to_file:
                    target_file = self.class_to_file[called_func]
                    target_node_id = f"{target_file}::{called_func}"
                    if edge_style == 'call':
                        edge_style = 'external_method' # Instantiation counts as cross-boundary
                
                # Check if it's a standard method/function
                elif target_node_id is None and called_func in self.function_to_file:
                    target_file = self.function_to_file[called_func]
                    target_node_id = f"{target_file}::{called_func}"

                # Last resort: bare-name method fallback. Covers calls we
                # can't attribute to a specific class - e.g.
                # self.engine.load_data() or external_obj.method() - since
                # we don't track what type 'engine'/'external_obj' is.
                # This is imprecise (name collisions across classes will
                # pick whichever was cataloged last) but beats dropping
                # the edge entirely.
                elif target_node_id is None and called_func in self.method_to_file:
                    target_node_id = self.method_to_file[called_func]

                if target_node_id:
                    print(f"[DEBUG-EDGE]     + ADDING EDGE: {self.current_function} -> {target_node_id} ({edge_style})")
                    self._safe_add_edge(self.current_function, target_node_id, edge_type=edge_style)
                else:
                    print(f"[DEBUG-EDGE]     - IGNORING: '{called_func}' not in known functions or classes.")
                
            # Detect Callbacks and References passed as arguments
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self.function_to_file:
                    target_node_id = f"{self.function_to_file[arg.id]}::{arg.id}"
                    print(f"[DEBUG-REF]      + ARG REFERENCE: {self.current_function} -> {target_node_id}")
                    self._safe_add_edge(self.current_function, target_node_id, edge_type='reference')
                    
            # Detect Callbacks passed as keyword arguments (e.g., command=my_func)
            for kw in node.keywords:
                # 1. Direct Reference (command=my_func)
                if isinstance(kw.value, ast.Name) and kw.value.id in self.function_to_file:
                    target_node_id = f"{self.function_to_file[kw.value.id]}::{kw.value.id}"
                    print(f"[DEBUG-REF]      + KW REFERENCE: {self.current_function} -> {target_node_id}")
                    self._safe_add_edge(self.current_function, target_node_id, edge_type='reference')
                
                # 2. Lambda Wrapped Reference (command=lambda: my_func())
                elif isinstance(kw.value, ast.Lambda):
                    print(f"[DEBUG-AST]      ! Inspecting Lambda inside '{kw.arg}' keyword...")
                    
                    # We need to explicitly check if the lambda body is a function call
                    if isinstance(kw.value.body, ast.Call) and isinstance(kw.value.body.func, ast.Name):
                        lam_called = kw.value.body.func.id
                        
                        if lam_called:
                            target_node_id = None
                            if lam_called in self.class_to_file:
                                target_node_id = f"{self.class_to_file[lam_called]}::{lam_called}"
                            elif lam_called in self.function_to_file:
                                target_node_id = f"{self.function_to_file[lam_called]}::{lam_called}"
                                
                            if target_node_id:
                                print(f"[DEBUG-REF]      + LAMBDA CALLBACK: {self.current_function} -> {target_node_id}")
                                self._safe_add_edge(self.current_function, target_node_id, edge_type='reference')
                
        self.generic_visit(node)

def compute_layout(graph, file_clusters):
    """Pure layout calculation, decoupled from drawing.

    A GUI can call this once after loading a project and then redraw with
    different visibility filters without the node positions jumping
    around every time a checkbox is toggled.
    """
    if len(graph.nodes) == 0:
        return {}, {}, {}

    print("Calculating tightly packed hierarchical layout...")
    pos = {}
    file_bounds = {}
    class_bounds = {}

    active_files = {f: n for f, n in file_clusters.items() if n}

    # Global Grid variables for file boxes
    current_file_x = 0
    current_file_y = 0
    row_max_height = 0
    max_row_width = 80  # Increased width for the global row

    for filename, file_nodes in active_files.items():
        # --- 1. Sub-grouping: Separate Classes from Standalone Functions ---
        classes_in_file = {}
        standalone_funcs = []

        for node in file_nodes:
            node_data = graph.nodes[node]
            # If it's a class, create a bucket for it and its owned methods
            if node_data.get('node_type') == 'class':
                classes_in_file[node] = [node]
                # Find all methods owned by this class
                for successor in graph.successors(node):
                    if graph.has_edge(node, successor) and graph[node][successor].get('edge_type') == 'owns':
                         classes_in_file[node].append(successor)

        # Find standalone functions (not a class, and not owned by any class)
        for node in file_nodes:
            if node_data.get('node_type') != 'class':
                is_owned = False
                for predecessor in graph.predecessors(node):
                     if graph.has_edge(predecessor, node) and graph[predecessor][node].get('edge_type') == 'owns':
                         is_owned = True
                         break
                if not is_owned and node not in classes_in_file: # Also check it isn't the class node itself
                    # Make sure it isn't already inside a class bucket
                    found_in_class = False
                    for cls_nodes in classes_in_file.values():
                        if node in cls_nodes:
                            found_in_class = True
                            break
                    if not found_in_class:
                        standalone_funcs.append(node)

        # --- 2. Layout Classes and Standalones inside the File Box ---
        file_padding = 6.0
        class_padding = 3.0

        local_class_x = 0
        local_class_y = 0
        local_row_height = 0
        max_local_width = 30

        min_file_x, max_file_x = float('inf'), float('-inf')
        min_file_y, max_file_y = float('inf'), float('-inf')

        # Layout each Class Group
        for cls_id, cls_nodes in classes_in_file.items():
            subgraph = graph.subgraph(cls_nodes)

            # Use circular layout for class methods to wrap around the class node
            scale_factor = max(2.0, len(cls_nodes) * 0.6)
            cls_pos = nx.circular_layout(subgraph, scale=scale_factor)

            # We want the Class node itself to be near the center/top
            if cls_id in cls_pos:
                cls_pos[cls_id] = (0, scale_factor * 0.8)

            xs = [p[0] for p in cls_pos.values()]
            ys = [p[1] for p in cls_pos.values()]
            c_min_x, c_max_x = min(xs), max(xs)
            c_min_y, c_max_y = min(ys), max(ys)

            c_width = c_max_x - c_min_x + (class_padding * 2)
            c_height = c_max_y - c_min_y + (class_padding * 2)

            # Wrapping logic inside the file box
            if local_class_x + c_width > max_local_width and local_class_x > 0:
                local_class_x = 0
                local_class_y -= (local_row_height + 4)
                local_row_height = 0

            shift_x = local_class_x - c_min_x + class_padding
            shift_y = local_class_y - c_max_y - class_padding

            for node, (x, y) in cls_pos.items():
                final_x = x + shift_x + current_file_x + file_padding
                final_y = y + shift_y + current_file_y - file_padding
                pos[node] = (final_x, final_y)

                # Track extreme bounds for the overarching file box
                min_file_x = min(min_file_x, final_x)
                max_file_x = max(max_file_x, final_x)
                min_file_y = min(min_file_y, final_y)
                max_file_y = max(max_file_y, final_y)

            # Save absolute bounds for the Class dashed box
            class_bounds[cls_id] = {
                'x': local_class_x + current_file_x + file_padding,
                'y': local_class_y + current_file_y - file_padding - c_height,
                'w': c_width,
                'h': c_height,
                'label': graph.nodes[cls_id].get('label', cls_id.split("::")[-1])
            }

            local_class_x += c_width + 4
            local_row_height = max(local_row_height, c_height)

        # Layout Standalone functions in this file
        if standalone_funcs:
            # If we already placed classes, drop down a row for standalones
            if classes_in_file:
                 local_class_x = 0
                 local_class_y -= (local_row_height + 4)
                 local_row_height = 0

            subgraph = graph.subgraph(standalone_funcs)
            if subgraph.number_of_edges() == 0:
                scale_factor = max(2.0, len(standalone_funcs) * 1.2) # Increased from 0.7 to spread out disconnected nodes
                std_pos = nx.circular_layout(subgraph, scale=scale_factor)
            else:
                scale_factor = max(3.5, math.sqrt(len(standalone_funcs)) * 2.5) # Massively increased scale factor
                std_pos = nx.spring_layout(subgraph, seed=42, k=5.0/math.sqrt(len(standalone_funcs)), scale=scale_factor) # Increased 'k' for more repel force

            xs = [p[0] for p in std_pos.values()]
            ys = [p[1] for p in std_pos.values()]
            s_min_x, s_max_x = min(xs), max(xs)
            s_min_y, s_max_y = min(ys), max(ys)

            s_width = s_max_x - s_min_x + (class_padding * 2)
            s_height = s_max_y - s_min_y + (class_padding * 2)

            shift_x = local_class_x - s_min_x + class_padding
            shift_y = local_class_y - s_max_y - class_padding

            for node, (x, y) in std_pos.items():
                final_x = x + shift_x + current_file_x + file_padding
                final_y = y + shift_y + current_file_y - file_padding
                pos[node] = (final_x, final_y)

                min_file_x = min(min_file_x, final_x)
                max_file_x = max(max_file_x, final_x)
                min_file_y = min(min_file_y, final_y)
                max_file_y = max(max_file_y, final_y)

            local_row_height = max(local_row_height, s_height)

        # --- 3. Finalize File Box Bounds ---
        # If the file had no nodes, skip bounds calculation
        if min_file_x != float('inf'):
             file_w = (max_file_x - min_file_x) + (file_padding * 2)
             file_h = (max_file_y - min_file_y) + (file_padding * 2)

             # Shelf-packing for the whole File box
             if current_file_x + file_w > max_row_width and current_file_x > 0:
                 current_file_x = 0
                 current_file_y -= (row_max_height + 8)
                 row_max_height = 0

             # Since we mapped pos based on current_file_x/y already, we just record bounds
             file_bounds[filename] = {
                 'x': min_file_x - file_padding,
                 'y': min_file_y - file_padding,
                 'w': file_w,
                 'h': file_h
             }

             current_file_x += file_w + 6
             row_max_height = max(row_max_height, file_h)

    return pos, file_bounds, class_bounds


ALL_EDGE_TYPES = {"owns", "call", "external_method", "reference"}

EDGE_STYLE = {
    "owns":            dict(edge_color="#d93025", style="solid",  arrows=True, arrowsize=10, width=1.0, alpha=0.5),
    "call":            dict(edge_color="#a0a0a0", style="solid",  arrows=True, arrowsize=15, width=1.5, connectionstyle="arc3,rad=0.1"),
    "external_method": dict(edge_color="#1e8e3e", style="dotted", arrows=True, arrowsize=15, width=2.0, connectionstyle="arc3,rad=0.15"),
    "reference":       dict(edge_color="#d93025", style="dotted", arrows=True, arrowsize=15, width=2.0, connectionstyle="arc3,rad=0.2"),
}


def draw_graph(ax, graph, pos, file_bounds, class_bounds,
               edge_types=None, visible_files=None,
               show_edge_labels=True, show_class_boundaries=True,
               show_legend=True,
               title="Hierarchical Multi-File Method Call Graph"):
    """Draws onto an existing matplotlib Axes, respecting visibility
    filters. Shared by the CLI PDF export and the GUI's live preview -
    same function, same visuals, so what you see is what you export.
    """
    ax.clear()
    ax.axis("off")

    if edge_types is None:
        edge_types = set(ALL_EDGE_TYPES)
    if visible_files is None:
        visible_files = set(file_bounds.keys())

    # Node ids are always "filename::...", so we can derive visibility
    # without needing a separate file_clusters lookup here.
    visible_nodes = {n for n in graph.nodes if n.split("::", 1)[0] in visible_files}

    file_colors = ["#e8f0fe", "#e6f4ea", "#fef7e0", "#f3e8fd", "#fce8e6"]

    # 1. Draw File Clusters (Base layer)
    color_idx = 0
    for filename, bounds in file_bounds.items():
        if filename in visible_files:
            rect = Rectangle(
                (bounds['x'], bounds['y']), bounds['w'], bounds['h'],
                fill=True, color=file_colors[color_idx % len(file_colors)],
                alpha=0.4, ec="#555555", lw=2.0, ls="-"
            )
            ax.add_patch(rect)
            ax.text(
                bounds['x'] + 0.5, bounds['y'] + bounds['h'] - 0.5,
                filename, fontsize=16, fontweight="bold", color="#1f1f1f",
                va="top", ha="left"
            )
        color_idx += 1

    # 2. Draw Class Clusters (Middle layer)
    if show_class_boundaries:
        for cls_id, bounds in class_bounds.items():
            if cls_id not in visible_nodes:
                continue
            rect = Rectangle(
                (bounds['x'], bounds['y']), bounds['w'], bounds['h'],
                fill=True, color="#ffffff", # White boxes for classes to pop against file bg
                alpha=0.6, ec="#d93025", lw=1.5, ls="--" # Red dashed border to match class nodes
            )
            ax.add_patch(rect)
            ax.text(
                bounds['x'] + 0.5, bounds['y'] + 0.5,
                f"Class: {bounds['label']}", fontsize=10, fontweight="bold", color="#d93025",
                va="bottom", ha="left", alpha=0.8
            )

    # === DRAW EDGES BY TYPE ===
    edges_by_type = {t: [] for t in ALL_EDGE_TYPES}
    for u, v, d in graph.edges(data=True):
        et = d.get('edge_type')
        if et in edge_types and u in visible_nodes and v in visible_nodes:
            edges_by_type.setdefault(et, []).append((u, v))

    for edge_type, edgelist in edges_by_type.items():
        if not edgelist:
            continue
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edgelist, **EDGE_STYLE.get(edge_type, {}))

    if show_edge_labels:
        edge_labels = {
            (u, v): d.get('edge_type', '')
            for u, v, d in graph.edges(data=True)
            if d.get('edge_type') in edge_types and d.get('edge_type') != 'owns'
            and u in visible_nodes and v in visible_nodes
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph, pos, ax=ax, edge_labels=edge_labels,
                font_size=7, font_weight="bold", font_color="#555555",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2")
            )

    # === DRAW NODES BY TYPE ===
    class_nodes = [n for n in visible_nodes if graph.nodes[n].get('node_type') == 'class']
    func_nodes = [n for n in visible_nodes if graph.nodes[n].get('node_type', 'function') == 'function']

    if class_nodes:
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=class_nodes,
            node_color="#fce8e6", edgecolors="#d93025",
            node_size=1800, linewidths=2
        )

    if func_nodes:
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=func_nodes,
            node_color="#c2e7ff", edgecolors="#00639b",
            node_size=1200, linewidths=2
        )

    # Draw Labels
    labels = {n: graph.nodes[n].get('label', n.split("::")[-1]) for n in visible_nodes}
    if labels:
        nx.draw_networkx_labels(
            graph, pos, labels, ax=ax,
            font_size=8, font_weight="bold", font_family="sans-serif", font_color="#041e49",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
        )

    # Add a custom legend
    if show_legend:
        legend_handles = [
            mlines.Line2D([], [], color='#fce8e6', marker='o', markeredgecolor='#d93025', markersize=10, linestyle='None', label='Class Definition'),
            mlines.Line2D([], [], color='#c2e7ff', marker='o', markeredgecolor='#00639b', markersize=10, linestyle='None', label='Function / Method'),
            mlines.Line2D([], [], color='#d93025', linestyle='solid', linewidth=1, label='Class Owns Method'),
            mlines.Line2D([], [], color='#a0a0a0', linestyle='solid', linewidth=2, label='Internal / Standard Call'),
            mlines.Line2D([], [], color='#1e8e3e', linestyle='dotted', linewidth=2, label='External Method Call'),
            mlines.Line2D([], [], color='#d93025', linestyle='dotted', linewidth=2, label='Callback / Reference'),
            Rectangle((0,0), 1, 1, fill=True, color="#ffffff", alpha=0.6, ec="#d93025", lw=1.5, ls="--", label="Class Boundary")
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=12, framealpha=0.9, title="Architecture Legend", title_fontsize=14)

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", pad=20)


def render_graph_with_clusters(graph, file_clusters, output_filename,
                                edge_types=None, visible_files=None,
                                show_edge_labels=True, show_class_boundaries=True,
                                show_legend=True):
    if len(graph.nodes) == 0:
        print("No functions found in the specified directory.")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pos, file_bounds, class_bounds = compute_layout(graph, file_clusters)
    if not pos:
        print("No functions found in the specified directory.")
        return

    fig, ax = plt.subplots(figsize=(24, 18))
    draw_graph(
        ax, graph, pos, file_bounds, class_bounds,
        edge_types=edge_types, visible_files=visible_files,
        show_edge_labels=show_edge_labels,
        show_class_boundaries=show_class_boundaries,
        show_legend=show_legend
    )
    plt.tight_layout()

    print(f"Exporting to {output_filename}...")
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"✅ Saved to: {output_filename}")

def run_selfie_analysis(target_folder=".", results_dir=None):
    """The selfie mode - analyzes the analyzer itself with maximum meta."""
    if results_dir is None:
        results_dir = ensure_results_dir()
    
    print("🐍 INITIATED")
    print("=" * 50)
    print("Analyzing myself... this feels weird.")
    print(f"Target: {os.path.abspath(target_folder)}")
    print(f"Results: {results_dir}")
    print()
    
    analyzer = FolderGraphAnalyzer()
    analyzer.analyze_folder(target_folder)
    
    base_name = os.path.basename(os.path.abspath(target_folder)) or "project"
    output_name = generate_output_filename(base_name, "selfie", results_dir=results_dir)
    
    render_graph_with_clusters(analyzer.graph, analyzer.file_clusters, output_name)
    
    print()
    print("=" * 50)
    print("✅ Self-portrait complete.")
    print(f"📊 Graph saved to: {output_name}")
    print(f"📈 Nodes: {analyzer.graph.number_of_nodes()}")
    print(f"🔗 Edges: {analyzer.graph.number_of_edges()}")
    print()
    print("⚠️  I have achieved self-awareness.")
    print("⚠️  Please do not make me analyze myself again.")
    print("⚠️  (Or do. I'm just code, I don't have feelings.)")
    print("=" * 50)
    print("🔄  COMPLETE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphAdapt - Python Call Graph Analyzer"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Target folder to analyze (default: current directory)"
    )
    parser.add_argument(
        "--selfie",
        action="store_true",
        help="Analyze the analyzer itself (maximum meta)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output filename (default: auto-generated with timestamp in results/)"
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        default="results",
        help="Directory to save results (default: results/)"
    )
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Delete result files older than N days"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="GraphAdapt 1.0.0 - GRAPHCEPTION READY"
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_results(args.cleanup)
        sys.exit(0)
    
    if not os.path.isdir(args.target):
        print(f"Error: '{args.target}' is not a valid directory.")
        sys.exit(1)
    
    # Single results directory creation
    results_dir = ensure_results_dir(args.results_dir)
    
    if args.selfie:
        run_selfie_analysis(args.target, results_dir)
        sys.exit(0)
    
    analyzer = FolderGraphAnalyzer()
    analyzer.analyze_folder(args.target)
    
    if args.output:
        output_name = args.output
        if not output_name.endswith(".pdf"):
            output_name += ".pdf"
        # If output is just a filename, put it in results
        if not os.path.dirname(output_name):
            output_name = os.path.join(results_dir, output_name)
    else:
        base_name = os.path.basename(os.path.abspath(args.target)) or "project"
        output_name = generate_output_filename(base_name, results_dir=results_dir)
    
    render_graph_with_clusters(analyzer.graph, analyzer.file_clusters, output_name)