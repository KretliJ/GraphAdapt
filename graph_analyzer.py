import ast
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import os
import sys
import math
import argparse
import time
import configparser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = "config.ini"
DEFAULT_CONFIG = {
    'paths': {
        'results_dir': 'results',
        'output_prefix': 'GraphAdapt',
    },
    'analysis': {
        'selfie_suffix': 'selfie',
        'default_suffix': 'multi_file_call_graph',
    },
    'logging': {
        'debug': 'false',
        'verbose': 'false',
    },
    'layout': {
        'max_row_width': '80',
        'max_local_width': '30',
        'file_padding': '6.0',
        'class_padding': '3.0',
    }
}

class Config:
    """Singleton configuration manager."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        self.config = configparser.ConfigParser()
        
        # Load defaults
        for section, values in DEFAULT_CONFIG.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for key, val in values.items():
                self.config.set(section, key, str(val))
        
        # Override with config.ini if it exists
        if os.path.exists(CONFIG_FILE):
            self.config.read(CONFIG_FILE)
        
        # Ensure config.ini exists with current values
        self._save()
    
    def _save(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            self.config.write(f)
    
    def get(self, section, key, fallback=None):
        """Get a config value, converting types automatically."""
        try:
            val = self.config.get(section, key)
            # Try to convert to appropriate type
            if val.lower() in ('true', 'false'):
                return val.lower() == 'true'
            try:
                return int(val)
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    return val
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_path(self, key, fallback=None):
        """Get a path from the paths section."""
        return self.get('paths', key, fallback)
    
    def get_analysis(self, key, fallback=None):
        """Get an analysis setting."""
        return self.get('analysis', key, fallback)
    
    def get_layout(self, key, fallback=None):
        """Get a layout setting."""
        return self.get('layout', key, fallback)
    
    def is_debug(self):
        """Check if debug mode is enabled."""
        return self.get('logging', 'debug', False)
    
    def is_verbose(self):
        """Check if verbose mode is enabled."""
        return self.get('logging', 'verbose', False)

config = Config()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_output_path(base_name: str, suffix: str = None, results_dir: str = None) -> str:
    """
    Build the full output path with timestamp.
    
    Args:
        base_name: Base name for the file (e.g., "project")
        suffix: Suffix to add (e.g., "selfie", "multi_file_call_graph")
        results_dir: Directory name (default: from config)
    
    Returns:
        Full path to the output file
    """
    if results_dir is None:
        results_dir = config.get_path('results_dir', 'results')
    
    if suffix is None:
        suffix = config.get_analysis('default_suffix', 'multi_file_call_graph')
    
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)
    os.makedirs(base, exist_ok=True)
    
    prefix = config.get_path('output_prefix', 'GraphAdapt')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{base_name}_{suffix}_{timestamp}.pdf"
    return os.path.join(base, filename)

def cleanup_results(days: int = 7, results_dir: str = None):
    """Delete result files older than N days."""
    if results_dir is None:
        results_dir = config.get_path('results_dir', 'results')
    
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)
    if not os.path.exists(base):
        print(f"📁 Results directory not found: {base}")
        return 0
    
    cutoff = time.time() - (days * 86400)
    count = 0
    
    for filename in os.listdir(base):
        if filename.endswith(".pdf"):
            filepath = os.path.join(base, filename)
            try:
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    count += 1
                    if config.is_verbose():
                        print(f"  🗑️  Removed: {filename}")
            except (OSError, PermissionError) as e:
                print(f"  ⚠️  Could not remove {filename}: {e}")
    
    if count:
        print(f"🧹 Cleaned up {count} old result files (> {days} days old)")
    else:
        print("🧹 No old result files to clean up")
    
    return count

def path_to_module(filepath: str, root_dir: str) -> str:
    """Convert a file path to a Python module name."""
    rel_path = os.path.relpath(filepath, root_dir)
    module = rel_path[:-3] if rel_path.endswith('.py') else rel_path
    module = module.replace(os.sep, '.')
    return module

def debug(msg: str):
    """Print debug message if debug mode is enabled."""
    if config.is_debug():
        print(f"[DEBUG] {msg}")

# ============================================================================
# PASS 1: CATALOG DEFINITIONS
# ============================================================================

class Pass1Visitor(ast.NodeVisitor):
    """First pass: catalog all class, function, and method definitions."""
    
    def __init__(self, analyzer: 'FolderGraphAnalyzer', filepath: str, module_name: str):
        self.analyzer = analyzer
        self.filepath = filepath
        self.module_name = module_name
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Catalog a class definition."""
        node_id = f"{self.module_name}::{node.name}"
        
        self.analyzer.classes[node_id] = self.filepath
        self.analyzer.class_by_name.setdefault(node.name, []).append(node_id)
        self.analyzer.graph.add_node(node_id, label=node.name, node_type='class')
        
        debug(f"Cataloged Class: {node_id}")
        
        prev_class = self.current_class
        self.current_class = node_id
        self.generic_visit(node)
        self.current_class = prev_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Catalog a function or method definition."""
        if self.current_class:
            class_name = self.current_class.split("::")[1]
            node_id = f"{self.module_name}::{class_name}.{node.name}"
            
            self.analyzer.methods_by_class.setdefault(self.current_class, {})[node.name] = node_id
            self.analyzer.method_by_name.setdefault(node.name, []).append(node_id)
        else:
            node_id = f"{self.module_name}::{node.name}"
            self.analyzer.function_by_name.setdefault(node.name, []).append(node_id)
        
        self.analyzer.functions[node_id] = self.filepath
        self.analyzer.graph.add_node(node_id, label=node.name, node_type='function')
        
        debug(f"Cataloged Function/Method: {node_id}")
        
        if self.current_class:
            self.analyzer.graph.add_edge(self.current_class, node_id, edge_type='owns')
        
        prev_func = self.current_function
        self.current_function = node_id
        self.generic_visit(node)
        self.current_function = prev_func

# ============================================================================
# PASS 2: COLLECT IMPORTS
# ============================================================================

@dataclass
class ImportInfo:
    """Information about an imported name."""
    alias: str
    module: str
    original_name: Optional[str] = None
    is_relative: bool = False

class ImportCollector(ast.NodeVisitor):
    """Second pass: collect all imports from a file."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.imports: Dict[str, ImportInfo] = {}
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = ImportInfo(
                alias=name,
                module=alias.name,
                original_name=None,
                is_relative=False
            )
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        is_relative = node.level > 0
        
        if is_relative:
            parts = self.module_name.split('.')
            if node.level <= len(parts):
                base_parts = parts[:-node.level]
            else:
                base_parts = []
            
            if module:
                base_parts.append(module)
            abs_module = '.'.join(base_parts)
        else:
            abs_module = module
        
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = ImportInfo(
                alias=name,
                module=abs_module,
                original_name=alias.name,
                is_relative=is_relative
            )
    
    @classmethod
    def collect(cls, tree: ast.AST, module_name: str) -> Dict[str, ImportInfo]:
        collector = cls(module_name)
        collector.visit(tree)
        return collector.imports

# ============================================================================
# PASS 3: RESOLVE CALLS
# ============================================================================

class CallResolver:
    """Resolve function and class names to their full qualified IDs."""
    
    def __init__(self, analyzer: 'FolderGraphAnalyzer'):
        self.analyzer = analyzer
        self.current_module: Optional[str] = None
        self.current_file: Optional[str] = None
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        self.imports: Dict[str, ImportInfo] = {}
    
    def resolve(self, name: str) -> Optional[str]:
        """Resolve a name to a full qualified ID."""
        if self.current_module is None:
            return None
        
        # 1. Check imports
        if name in self.imports:
            imp = self.imports[name]
            target_name = imp.original_name or imp.alias
            target_id = f"{imp.module}::{target_name}"
            
            if target_id in self.analyzer.functions or target_id in self.analyzer.classes:
                debug(f"Import resolved: {name} → {target_id}")
                return target_id
        
        # 2. Check current module
        local_id = f"{self.current_module}::{name}"
        if local_id in self.analyzer.functions or local_id in self.analyzer.classes:
            debug(f"Local resolved: {name} → {local_id}")
            return local_id
        
        # 3. Check current class methods
        if self.current_class:
            class_id = self.current_class
            if class_id in self.analyzer.methods_by_class:
                methods = self.analyzer.methods_by_class[class_id]
                if name in methods:
                    debug(f"Method resolved: {name} → {methods[name]}")
                    return methods[name]
        
        # 4. Check global definitions
        candidates = []
        candidates.extend(self.analyzer.function_by_name.get(name, []))
        candidates.extend(self.analyzer.class_by_name.get(name, []))
        
        if len(candidates) == 1:
            debug(f"Global resolved: {name} → {candidates[0]}")
            return candidates[0]
        elif len(candidates) > 1:
            # Prefer same module
            for target_id in candidates:
                if target_id.startswith(self.current_module):
                    debug(f"Ambiguous resolved (same module): {name} → {target_id}")
                    return target_id
            
            debug(f"Ambiguous name: {name} has multiple candidates: {candidates}")
            return candidates[0]
        
        # 5. Method fallback
        if name in self.analyzer.method_by_name:
            candidates = self.analyzer.method_by_name[name]
            if len(candidates) == 1:
                debug(f"Method fallback: {name} → {candidates[0]}")
                return candidates[0]
        
        debug(f"Could not resolve: {name}")
        return None

# ============================================================================
# PASS 3: BUILD CALL GRAPH
# ============================================================================

class CallGraphBuilder(ast.NodeVisitor):
    """Third pass: build the call graph by resolving calls."""
    
    def __init__(self, analyzer: 'FolderGraphAnalyzer'):
        self.analyzer = analyzer
        self.resolver = CallResolver(analyzer)
        self.current_module: Optional[str] = None
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        self.imports: Dict[str, ImportInfo] = {}
    
    def visit_ClassDef(self, node: ast.ClassDef):
        prev_class = self.current_class
        self.current_class = f"{self.current_module}::{node.name}"
        self.generic_visit(node)
        self.current_class = prev_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.current_class:
            class_name = self.current_class.split("::")[1]
            self.current_function = f"{self.current_module}::{class_name}.{node.name}"
        else:
            self.current_function = f"{self.current_module}::{node.name}"
        
        self.resolver.current_function = self.current_function
        self.resolver.current_class = self.current_class
        
        debug(f"Entering Function: {self.current_function}")
        self.generic_visit(node)
        self.current_function = None
        self.resolver.current_function = None
    
    def _safe_add_edge(self, source: str, target: str, edge_type: str):
        if self.analyzer.graph.has_edge(source, target):
            existing = self.analyzer.graph[source][target].get('edge_type')
            if existing == 'reference' and edge_type in ['call', 'method', 'external_method']:
                return
        self.analyzer.graph.add_edge(source, target, edge_type=edge_type)
    
    def visit_Call(self, node: ast.Call):
        if self.current_function is None:
            self.generic_visit(node)
            return
        
        called_name = None
        edge_type = 'call'
        is_self_call = False
        is_class_call = False
        
        if isinstance(node.func, ast.Name):
            called_name = node.func.id
            edge_type = 'call'
        
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr
            edge_type = 'external_method'
            
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'self':
                    edge_type = 'call'
                    is_self_call = True
                elif node.func.value.id == 'cls':
                    edge_type = 'external_method'
                    is_class_call = True
        
        if called_name is None:
            self.generic_visit(node)
            return
        
        debug(f"Found Call in {self.current_function}: target='{called_name}'")
        
        target_id = None
        
        # Self method call
        if is_self_call and self.current_class:
            methods = self.analyzer.methods_by_class.get(self.current_class, {})
            if called_name in methods:
                target_id = methods[called_name]
                debug(f"Self method resolved: {called_name} → {target_id}")
        
        # Resolver
        if target_id is None:
            self.resolver.current_module = self.current_module
            self.resolver.current_function = self.current_function
            self.resolver.current_class = self.current_class
            self.resolver.imports = self.imports
            target_id = self.resolver.resolve(called_name)
        
        if target_id:
            debug(f"ADDING EDGE: {self.current_function} -> {target_id} ({edge_type})")
            self._safe_add_edge(self.current_function, target_id, edge_type)
        else:
            debug(f"IGNORING: '{called_name}' not resolved.")
        
        self._detect_callbacks(node)
        self.generic_visit(node)
    
    def _detect_callbacks(self, node: ast.Call):
        if self.current_function is None:
            return
        
        for arg in node.args:
            if isinstance(arg, ast.Name):
                target_id = self.resolver.resolve(arg.id)
                if target_id:
                    debug(f"ARG REFERENCE: {self.current_function} -> {target_id}")
                    self._safe_add_edge(self.current_function, target_id, 'reference')
        
        for kw in node.keywords:
            if isinstance(kw.value, ast.Name):
                target_id = self.resolver.resolve(kw.value.id)
                if target_id:
                    debug(f"KW REFERENCE: {self.current_function} -> {target_id}")
                    self._safe_add_edge(self.current_function, target_id, 'reference')
            
            elif isinstance(kw.value, ast.Lambda):
                if isinstance(kw.value.body, ast.Call):
                    if isinstance(kw.value.body.func, ast.Name):
                        lam_called = kw.value.body.func.id
                        target_id = self.resolver.resolve(lam_called)
                        if target_id:
                            debug(f"LAMBDA CALLBACK: {self.current_function} -> {target_id}")
                            self._safe_add_edge(self.current_function, target_id, 'reference')

# ============================================================================
# MAIN ANALYZER
# ============================================================================

class FolderGraphAnalyzer:
    """Main analyzer that orchestrates all passes."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_clusters: Dict[str, List[str]] = {}
        
        self.module_to_file: Dict[str, str] = {}
        self.file_to_module: Dict[str, str] = {}
        self.root_dir: Optional[str] = None
        
        self.functions: Dict[str, str] = {}
        self.classes: Dict[str, str] = {}
        self.function_by_name: Dict[str, List[str]] = {}
        self.class_by_name: Dict[str, List[str]] = {}
        self.method_by_name: Dict[str, List[str]] = {}
        self.methods_by_class: Dict[str, Dict[str, str]] = {}
        
        self.imports: Dict[str, Dict[str, ImportInfo]] = {}
        self.py_files: List[str] = []
    
    def analyze_folder(self, folder_path: str):
        """Analyze all Python files in a folder recursively."""
        print(f"Crawling directory: {folder_path} ...")
        self.root_dir = os.path.abspath(folder_path)
        
        self.py_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):
                    self.py_files.append(os.path.join(root, file))
        
        if not self.py_files:
            print("No Python files found.")
            return
        
        print(f"Found {len(self.py_files)} Python files.")
        
        # Build module mappings
        for filepath in self.py_files:
            module = path_to_module(filepath, self.root_dir)
            self.module_to_file[module] = filepath
            self.file_to_module[filepath] = module
            self.file_clusters[filepath] = []
            debug(f"Mapping: {filepath} → {module}")
        
        # PASS 1: Catalog definitions
        print("\n=== PASS 1: Cataloging Definitions ===")
        for filepath in self.py_files:
            module = self.file_to_module[filepath]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    visitor = Pass1Visitor(self, filepath, module)
                    visitor.visit(tree)
            except Exception as e:
                print(f"Failed to parse {filepath}: {e}")
        
        # PASS 2: Collect imports
        print("\n=== PASS 2: Collecting Imports ===")
        for filepath in self.py_files:
            module = self.file_to_module[filepath]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    self.imports[module] = ImportCollector.collect(tree, module)
                    if config.is_verbose():
                        print(f"  {module}: {len(self.imports[module])} imports")
            except Exception as e:
                print(f"Failed to parse {filepath}: {e}")
        
        # PASS 3: Build call graph
        print("\n=== PASS 3: Building Call Graph ===")
        for filepath in self.py_files:
            module = self.file_to_module[filepath]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    builder = CallGraphBuilder(self)
                    builder.current_module = module
                    builder.imports = self.imports.get(module, {})
                    builder.resolver.current_module = module
                    builder.resolver.imports = builder.imports
                    builder.visit(tree)
            except Exception as e:
                print(f"Failed to parse {filepath}: {e}")
        
        self._filter_dunder_init()
        self._build_file_clusters()
    
    def _build_file_clusters(self):
        """Build file clusters from module mappings."""
        self.file_clusters = {}
        for module, filepath in self.module_to_file.items():
            if self.root_dir:
                rel_path = os.path.relpath(filepath, self.root_dir)
            else:
                rel_path = os.path.basename(filepath)
            self.file_clusters[rel_path] = []
        
        for node_id in self.graph.nodes:
            if "::" in node_id:
                module = node_id.split("::")[0]
                filepath = self.module_to_file.get(module)
                if filepath and self.root_dir:
                    rel_path = os.path.relpath(filepath, self.root_dir)
                    if rel_path in self.file_clusters:
                        self.file_clusters[rel_path].append(node_id)
    
    def _filter_dunder_init(self):
        """Remove __init__ nodes but preserve their edges."""
        init_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('label') == '__init__']
        if not init_nodes:
            return
        
        print(f"[DEBUG-FILTER] Removing {len(init_nodes)} __init__ node(s)")
        
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
                    self._safe_add_edge(owner, succ, edge_type=edge_type)
            
            for pred in list(self.graph.predecessors(init_node)):
                edge_type = self.graph[pred][init_node].get('edge_type')
                if edge_type != 'owns' and pred != owner:
                    self._safe_add_edge(pred, owner, edge_type=edge_type)
        
        self.graph.remove_nodes_from(init_nodes)
    
    def _safe_add_edge(self, source: str, target: str, edge_type: str):
        if self.graph.has_edge(source, target):
            existing = self.graph[source][target].get('edge_type')
            if existing == 'reference' and edge_type in ['call', 'method', 'external_method']:
                return
        self.graph.add_edge(source, target, edge_type=edge_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'files': len(self.py_files),
            'modules': len(self.module_to_file),
            'functions': len(self.functions),
            'methods': sum(len(m) for m in self.methods_by_class.values()),
            'classes': len(self.classes),
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
        }

# ============================================================================
# LAYOUT COMPUTATION
# ============================================================================

def compute_layout(graph, file_clusters):
    if len(graph.nodes) == 0:
        return {}, {}, {}

    print("Calculating tightly packed hierarchical layout...")
    pos = {}
    file_bounds = {}
    class_bounds = {}

    active_files = {f: n for f, n in file_clusters.items() if n}

    current_file_x = 0
    current_file_y = 0
    row_max_height = 0
    max_row_width = config.get_layout('max_row_width', 80)

    for filename, file_nodes in active_files.items():
        classes_in_file = {}
        standalone_funcs = []

        for node in file_nodes:
            node_data = graph.nodes[node]
            if node_data.get('node_type') == 'class':
                classes_in_file[node] = [node]
                for successor in graph.successors(node):
                    if graph.has_edge(node, successor) and graph[node][successor].get('edge_type') == 'owns':
                        classes_in_file[node].append(successor)

        for node in file_nodes:
            node_data = graph.nodes[node]
            if node_data.get('node_type') != 'class':
                is_owned = False
                for predecessor in graph.predecessors(node):
                    if graph.has_edge(predecessor, node) and graph[predecessor][node].get('edge_type') == 'owns':
                        is_owned = True
                        break
                if not is_owned and node not in classes_in_file:
                    found_in_class = False
                    for cls_nodes in classes_in_file.values():
                        if node in cls_nodes:
                            found_in_class = True
                            break
                    if not found_in_class:
                        standalone_funcs.append(node)

        file_padding = config.get_layout('file_padding', 6.0)
        class_padding = config.get_layout('class_padding', 3.0)

        local_class_x = 0
        local_class_y = 0
        local_row_height = 0
        max_local_width = config.get_layout('max_local_width', 30)

        min_file_x, max_file_x = float('inf'), float('-inf')
        min_file_y, max_file_y = float('inf'), float('-inf')

        for cls_id, cls_nodes in classes_in_file.items():
            subgraph = graph.subgraph(cls_nodes)
            scale_factor = max(2.0, len(cls_nodes) * 0.6)
            cls_pos = nx.circular_layout(subgraph, scale=scale_factor)

            if cls_id in cls_pos:
                cls_pos[cls_id] = (0, scale_factor * 0.8)

            xs = [p[0] for p in cls_pos.values()]
            ys = [p[1] for p in cls_pos.values()]
            c_min_x, c_max_x = min(xs), max(xs)
            c_min_y, c_max_y = min(ys), max(ys)

            c_width = c_max_x - c_min_x + (class_padding * 2)
            c_height = c_max_y - c_min_y + (class_padding * 2)

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

                min_file_x = min(min_file_x, final_x)
                max_file_x = max(max_file_x, final_x)
                min_file_y = min(min_file_y, final_y)
                max_file_y = max(max_file_y, final_y)

            class_bounds[cls_id] = {
                'x': local_class_x + current_file_x + file_padding,
                'y': local_class_y + current_file_y - file_padding - c_height,
                'w': c_width,
                'h': c_height,
                'label': graph.nodes[cls_id].get('label', cls_id.split("::")[-1])
            }

            local_class_x += c_width + 4
            local_row_height = max(local_row_height, c_height)

        if standalone_funcs:
            if classes_in_file:
                local_class_x = 0
                local_class_y -= (local_row_height + 4)
                local_row_height = 0

            subgraph = graph.subgraph(standalone_funcs)
            if subgraph.number_of_edges() == 0:
                scale_factor = max(2.0, len(standalone_funcs) * 1.2)
                std_pos = nx.circular_layout(subgraph, scale=scale_factor)
            else:
                scale_factor = max(3.5, math.sqrt(len(standalone_funcs)) * 2.5)
                std_pos = nx.spring_layout(subgraph, seed=42, k=5.0/math.sqrt(len(standalone_funcs)), scale=scale_factor)

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

        if min_file_x != float('inf'):
            file_w = (max_file_x - min_file_x) + (file_padding * 2)
            file_h = (max_file_y - min_file_y) + (file_padding * 2)

            if current_file_x + file_w > max_row_width and current_file_x > 0:
                current_file_x = 0
                current_file_y -= (row_max_height + 8)
                row_max_height = 0

            file_bounds[filename] = {
                'x': min_file_x - file_padding,
                'y': min_file_y - file_padding,
                'w': file_w,
                'h': file_h
            }

            current_file_x += file_w + 6
            row_max_height = max(row_max_height, file_h)

    return pos, file_bounds, class_bounds

# ============================================================================
# EDGE STYLES AND DRAWING
# ============================================================================

ALL_EDGE_TYPES = {"owns", "call", "external_method", "reference"}

EDGE_STYLE = {
    "owns": dict(edge_color="#d93025", style="solid", arrows=True, arrowsize=10, width=1.0, alpha=0.5),
    "call": dict(edge_color="#a0a0a0", style="solid", arrows=True, arrowsize=15, width=1.5, connectionstyle="arc3,rad=0.1"),
    "external_method": dict(edge_color="#1e8e3e", style="dotted", arrows=True, arrowsize=15, width=2.0, connectionstyle="arc3,rad=0.15"),
    "reference": dict(edge_color="#d93025", style="dotted", arrows=True, arrowsize=15, width=2.0, connectionstyle="arc3,rad=0.2"),
}

def draw_graph(ax, graph, pos, file_bounds, class_bounds,
               edge_types=None, visible_files=None,
               show_edge_labels=True, show_class_boundaries=True,
               show_legend=True,
               title="Hierarchical Multi-File Method Call Graph"):
    ax.clear()
    ax.axis("off")

    if edge_types is None:
        edge_types = set(ALL_EDGE_TYPES)
    if visible_files is None:
        visible_files = set(file_bounds.keys())

    visible_nodes = set()
    for n in graph.nodes:
        if "::" in n:
            module = n.split("::")[0]
            for filename in file_bounds.keys():
                if filename in n or n.startswith(module):
                    if filename in visible_files:
                        visible_nodes.add(n)
                        break
        else:
            for filename in file_bounds.keys():
                if filename in n:
                    if filename in visible_files:
                        visible_nodes.add(n)
                        break

    file_colors = ["#e8f0fe", "#e6f4ea", "#fef7e0", "#f3e8fd", "#fce8e6"]

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

    if show_class_boundaries:
        for cls_id, bounds in class_bounds.items():
            if cls_id not in visible_nodes:
                continue
            rect = Rectangle(
                (bounds['x'], bounds['y']), bounds['w'], bounds['h'],
                fill=True, color="#ffffff",
                alpha=0.6, ec="#d93025", lw=1.5, ls="--"
            )
            ax.add_patch(rect)
            ax.text(
                bounds['x'] + 0.5, bounds['y'] + 0.5,
                f"Class: {bounds['label']}", fontsize=10, fontweight="bold", color="#d93025",
                va="bottom", ha="left", alpha=0.8
            )

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

    labels = {n: graph.nodes[n].get('label', n.split("::")[-1]) for n in visible_nodes}
    if labels:
        nx.draw_networkx_labels(
            graph, pos, labels, ax=ax,
            font_size=8, font_weight="bold", font_family="sans-serif", font_color="#041e49",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
        )

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

# ============================================================================
# RENDER FUNCTION
# ============================================================================

def render_graph_with_clusters(graph, file_clusters, output_filename,
                               edge_types=None, visible_files=None,
                               show_edge_labels=True, show_class_boundaries=True,
                               show_legend=True):
    if len(graph.nodes) == 0:
        print("No functions found in the specified directory.")
        return

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

# ============================================================================
# SELFIE MODE
# ============================================================================

def run_selfie_analysis(target_folder=".", results_dir=None):
    """Selfie mode: analyze the analyzer itself."""
    if results_dir is None:
        results_dir = config.get_path('results_dir', 'results')

    print("🐍 GRAPHCEPTION INITIATED")
    print("=" * 50)
    print("Analyzing myself... this feels weird.")
    print(f"Target: {os.path.abspath(target_folder)}")
    print(f"Results: {results_dir}")
    print()

    analyzer = FolderGraphAnalyzer()
    analyzer.analyze_folder(target_folder)

    stats = analyzer.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Files: {stats['files']}")
    print(f"  Modules: {stats['modules']}")
    print(f"  Functions: {stats['functions']}")
    print(f"  Methods: {stats['methods']}")
    print(f"  Classes: {stats['classes']}")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")

    base_name = os.path.basename(os.path.abspath(target_folder)) or "project"
    suffix = config.get_analysis('selfie_suffix', 'selfie')
    output_name = get_output_path(base_name, suffix, results_dir)

    render_graph_with_clusters(analyzer.graph, analyzer.file_clusters, output_name)

    print()
    print("=" * 50)
    print("✅ Self-portrait complete.")
    print(f"📊 Graph saved to: {output_name}")
    print()
    print("⚠️  I have achieved self-awareness.")
    print("⚠️  Please do not make me analyze myself again.")
    print("⚠️  (Or do. I'm just code, I don't have feelings.)")
    print("=" * 50)
    print("🔄 GRAPHCEPTION COMPLETE")

# ============================================================================
# MAIN
# ============================================================================

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
        default=None,
        help="Directory to save results (default: from config.ini)"
    )
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Delete result files older than N days"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="GraphAdapt 1.0.0 - GRAPHCEPTION READY"
    )

    args = parser.parse_args()

    # Override config with CLI flags
    if args.debug:
        config.config.set('logging', 'debug', 'true')
    if args.verbose:
        config.config.set('logging', 'verbose', 'true')

    if args.cleanup:
        cleanup_results(args.cleanup, args.results_dir)
        sys.exit(0)

    if not os.path.isdir(args.target):
        print(f"Error: '{args.target}' is not a valid directory.")
        sys.exit(1)

    if args.selfie:
        run_selfie_analysis(args.target, args.results_dir)
        sys.exit(0)

    analyzer = FolderGraphAnalyzer()
    analyzer.analyze_folder(args.target)

    if args.output:
        output_name = args.output
        if not output_name.endswith(".pdf"):
            output_name += ".pdf"
        if not os.path.dirname(output_name):
            # Output is just a filename, put it in results
            results_dir = args.results_dir or config.get_path('results_dir', 'results')
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_dir)
            os.makedirs(base, exist_ok=True)
            output_name = os.path.join(base, output_name)
    else:
        base_name = os.path.basename(os.path.abspath(args.target)) or "project"
        output_name = get_output_path(base_name, results_dir=args.results_dir)

    render_graph_with_clusters(analyzer.graph, analyzer.file_clusters, output_name)

    stats = analyzer.get_statistics()
    print(f"\n📊 Analysis complete:")
    print(f"  Files: {stats['files']}")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Graph saved to: {output_name}")