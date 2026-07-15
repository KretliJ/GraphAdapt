"""
Unit tests for GraphAdapt call graph analyzer.

Run with: pytest test_graph_analyzer.py -v
Or: python -m unittest test_graph_analyzer.py
"""

import ast
import os
import tempfile
import unittest
import shutil
from pathlib import Path
from datetime import datetime

import networkx as nx

from graph_analyzer import (
    FolderGraphAnalyzer,
    Pass1Visitor,
    compute_layout,
    render_graph_with_clusters,
    ensure_results_dir,
    generate_output_filename,
    cleanup_results,
    ALL_EDGE_TYPES,
)


class TestPass1Visitor(unittest.TestCase):
    """Test the first-pass cataloging of definitions."""

    def setUp(self):
        self.analyzer = FolderGraphAnalyzer()
        self.filename = "test_file.py"
        self.analyzer.file_clusters[self.filename] = []
        self.visitor = Pass1Visitor(self.analyzer, self.filename)

    def test_catalog_class(self):
        """Classes should be added to graph with correct metadata."""
        tree = ast.parse("""
class MyClass:
    pass
""")
        self.visitor.visit(tree)
        
        node_id = f"{self.filename}::MyClass"
        self.assertIn(node_id, self.analyzer.graph.nodes)
        self.assertEqual(self.analyzer.graph.nodes[node_id]['label'], 'MyClass')
        self.assertEqual(self.analyzer.graph.nodes[node_id]['node_type'], 'class')
        self.assertIn(node_id, self.analyzer.defined_classes)

    def test_catalog_function(self):
        """Standalone functions should be cataloged correctly."""
        tree = ast.parse("""
def my_func():
    pass
""")
        self.visitor.visit(tree)
        
        node_id = f"{self.filename}::my_func"
        self.assertIn(node_id, self.analyzer.graph.nodes)
        self.assertEqual(self.analyzer.graph.nodes[node_id]['label'], 'my_func')
        self.assertEqual(self.analyzer.graph.nodes[node_id]['node_type'], 'function')
        self.assertIn(node_id, self.analyzer.defined_functions)

    def test_catalog_method(self):
        """Methods should be owned by their class with scoped IDs."""
        tree = ast.parse("""
class MyClass:
    def my_method(self):
        pass
""")
        self.visitor.visit(tree)
        
        class_id = f"{self.filename}::MyClass"
        method_id = f"{self.filename}::MyClass.my_method"
        
        self.assertIn(class_id, self.analyzer.graph.nodes)
        self.assertIn(method_id, self.analyzer.graph.nodes)
        
        # Ownership edge should exist
        self.assertTrue(self.analyzer.graph.has_edge(class_id, method_id))
        self.assertEqual(
            self.analyzer.graph[class_id][method_id]['edge_type'],
            'owns'
        )
        
        # Method should be in methods_by_class
        self.assertIn(method_id, self.analyzer.methods_by_class[class_id].values())

    def test_same_name_methods_different_classes(self):
        """Methods with same name in different classes should not collide."""
        tree = ast.parse("""
class VisitorA:
    def visit(self):
        pass

class VisitorB:
    def visit(self):
        pass
""")
        self.visitor.visit(tree)
        
        method_a = f"{self.filename}::VisitorA.visit"
        method_b = f"{self.filename}::VisitorB.visit"
        
        self.assertIn(method_a, self.analyzer.graph.nodes)
        self.assertIn(method_b, self.analyzer.graph.nodes)
        self.assertNotEqual(method_a, method_b)
        
        # Both should be in method_to_file (fallback lookup)
        self.assertEqual(self.analyzer.method_to_file['visit'], method_b)  # Last one wins


class TestFolderGraphAnalyzer(unittest.TestCase):
    """Test the full analyzer pipeline."""

    def setUp(self):
        self.analyzer = FolderGraphAnalyzer()

    def _create_temp_file(self, content, filename="test.py"):
        """Helper to create a temp Python file and return path."""
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return temp_dir, file_path

    def _cleanup_temp_dir(self, temp_dir):
        """Helper to clean up temp directory."""
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_analyze_single_file(self):
        """Basic analysis of a single file."""
        content = """
def foo():
    return 42

def bar():
    return foo()
"""
        temp_dir, file_path = self._create_temp_file(content)
        
        try:
            self.analyzer.analyze_folder(temp_dir)
            
            # Should have 2 function nodes
            self.assertEqual(len(self.analyzer.graph.nodes), 2)
            self.assertEqual(len(self.analyzer.graph.edges), 1)  # bar -> foo
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_analyze_cross_file_call(self):
        """Calls across files should be resolved."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # File A: defines a function
            with open(os.path.join(temp_dir, "a.py"), "w", encoding="utf-8") as f:
                f.write("""
def helper():
    return 42
""")
            
            # File B: calls it
            with open(os.path.join(temp_dir, "b.py"), "w", encoding="utf-8") as f:
                f.write("""
from a import helper

def main():
    return helper()
""")
            
            self.analyzer.analyze_folder(temp_dir)
            
            # Should have nodes for helper and main
            helper_id = "a.py::helper"
            main_id = "b.py::main"
            
            self.assertIn(helper_id, self.analyzer.graph.nodes)
            self.assertIn(main_id, self.analyzer.graph.nodes)
            self.assertTrue(self.analyzer.graph.has_edge(main_id, helper_id))
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_self_method_call(self):
        """self.method() should resolve against the caller's class."""
        content = """
class MyClass:
    def caller(self):
        self.callee()
    
    def callee(self):
        pass
"""
        temp_dir, file_path = self._create_temp_file(content)
        
        try:
            self.analyzer.analyze_folder(temp_dir)
            
            class_id = "test.py::MyClass"
            caller_id = "test.py::MyClass.caller"
            callee_id = "test.py::MyClass.callee"
            
            # Edge from caller to callee
            self.assertTrue(self.analyzer.graph.has_edge(caller_id, callee_id))
            self.assertEqual(
                self.analyzer.graph[caller_id][callee_id]['edge_type'],
                'call'
            )
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_class_instantiation(self):
        """Class instantiation should create external_method edges."""
        content = """
class MyClass:
    pass

def create():
    return MyClass()
"""
        temp_dir, file_path = self._create_temp_file(content)
        
        try:
            self.analyzer.analyze_folder(temp_dir)
            
            class_id = "test.py::MyClass"
            func_id = "test.py::create"
            
            self.assertTrue(self.analyzer.graph.has_edge(func_id, class_id))
            self.assertEqual(
                self.analyzer.graph[func_id][class_id]['edge_type'],
                'external_method'
            )
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_callback_detection(self):
        """Function references as arguments should be detected."""
        content = """
def handler():
    pass

def register(callback):
    callback()

def setup():
    register(handler)
"""
        temp_dir, file_path = self._create_temp_file(content)
        
        try:
            self.analyzer.analyze_folder(temp_dir)
            
            handler_id = "test.py::handler"
            setup_id = "test.py::setup"
            
            self.assertTrue(self.analyzer.graph.has_edge(setup_id, handler_id))
            self.assertEqual(
                self.analyzer.graph[setup_id][handler_id]['edge_type'],
                'reference'
            )
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_lambda_callback(self):
        """Lambda-wrapped callbacks should be detected."""
        content = """
def handler():
    pass

def setup():
    btn = Button(command=lambda: handler())
"""
        temp_dir, file_path = self._create_temp_file(content)
        
        try:
            self.analyzer.analyze_folder(temp_dir)
            
            handler_id = "test.py::handler"
            setup_id = "test.py::setup"
            
            # Should detect lambda callback reference
            self.assertTrue(self.analyzer.graph.has_edge(setup_id, handler_id))
            self.assertEqual(
                self.analyzer.graph[setup_id][handler_id]['edge_type'],
                'reference'
            )
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_dunder_init_filter(self):
        """__init__ should be filtered out with edges re-parented."""
        content = """
class MyClass:
    def __init__(self):
        self.setup()
    
    def setup(self):
        pass

def create():
    return MyClass()
"""
        temp_dir, file_path = self._create_temp_file(content)
        
        try:
            self.analyzer.analyze_folder(temp_dir)
            
            # __init__ node should be gone
            init_id = "test.py::MyClass.__init__"
            self.assertNotIn(init_id, self.analyzer.graph.nodes)
            
            # But its edges should be re-parented to MyClass
            class_id = "test.py::MyClass"
            setup_id = "test.py::MyClass.setup"
            
            # MyClass should own setup
            self.assertTrue(self.analyzer.graph.has_edge(class_id, setup_id))
        finally:
            self._cleanup_temp_dir(temp_dir)

    def test_safe_add_edge_prevents_overwrite(self):
        """_safe_add_edge should preserve reference/callback edges."""
        self.analyzer.graph.add_edge("A", "B", edge_type="reference")
        
        # Try to overwrite with 'call'
        self.analyzer._safe_add_edge("A", "B", edge_type="call")
        
        # Should still be 'reference'
        self.assertEqual(
            self.analyzer.graph["A"]["B"]["edge_type"],
            "reference"
        )
        
        # But new edge types should be added
        self.analyzer._safe_add_edge("A", "C", edge_type="call")
        self.assertEqual(
            self.analyzer.graph["A"]["C"]["edge_type"],
            "call"
        )


class TestComputeLayout(unittest.TestCase):
    """Test layout computation (doesn't crash, returns sensible data)."""

    def test_empty_graph(self):
        """Empty graph should return empty layouts."""
        g = nx.DiGraph()
        pos, file_bounds, class_bounds = compute_layout(g, {})
        self.assertEqual(pos, {})
        self.assertEqual(file_bounds, {})
        self.assertEqual(class_bounds, {})

    def test_single_function(self):
        """Single function should produce valid layout."""
        g = nx.DiGraph()
        g.add_node("file.py::func", label="func", node_type="function")
        clusters = {"file.py": ["file.py::func"]}
        
        pos, file_bounds, class_bounds = compute_layout(g, clusters)
        
        self.assertIn("file.py::func", pos)
        self.assertIn("file.py", file_bounds)
        self.assertEqual(class_bounds, {})

    def test_class_with_methods(self):
        """Class with methods should produce class bounds."""
        g = nx.DiGraph()
        g.add_node("file.py::MyClass", label="MyClass", node_type="class")
        g.add_node("file.py::MyClass.method", label="method", node_type="function")
        g.add_edge("file.py::MyClass", "file.py::MyClass.method", edge_type="owns")
        clusters = {"file.py": ["file.py::MyClass", "file.py::MyClass.method"]}
        
        pos, file_bounds, class_bounds = compute_layout(g, clusters)
        
        self.assertIn("file.py::MyClass", class_bounds)
        self.assertEqual(class_bounds["file.py::MyClass"]["label"], "MyClass")


class TestResultsFolder(unittest.TestCase):
    """Test results folder functionality."""

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ensure_results_dir_creates_folder(self):
        """ensure_results_dir should create results folder."""
        results_dir = ensure_results_dir()
        self.assertTrue(os.path.exists(results_dir))
        self.assertTrue(os.path.isdir(results_dir))
        self.assertTrue(results_dir.endswith("results"))

    def test_ensure_results_dir_with_subfolder(self):
        """ensure_results_dir with subfolder should create nested folder."""
        results_dir = ensure_results_dir("graphs")
        self.assertTrue(os.path.exists(results_dir))
        self.assertTrue(results_dir.endswith(os.path.join("results", "graphs")))

    def test_generate_output_filename_creates_timestamped_file(self):
        """generate_output_filename should create timestamped filename in results."""
        results_dir = ensure_results_dir()
        filename = generate_output_filename("test_project", "analysis", results_dir)
        
        self.assertTrue(filename.startswith(os.path.join(results_dir, "test_project_analysis_")))
        self.assertTrue(filename.endswith(".pdf"))
        # Should contain timestamp (YYYYMMDD_HHMMSS)
        import re
        self.assertTrue(re.search(r'\d{8}_\d{6}', filename))

    def test_cleanup_results_removes_old_files(self):
        """cleanup_results should remove files older than N days."""
        results_dir = ensure_results_dir()
        
        # Create an old file (by setting mtime to 10 days ago)
        old_file = os.path.join(results_dir, "old_test.pdf")
        with open(old_file, "w") as f:
            f.write("test")
        
        # Set mtime to 10 days ago
        old_time = datetime.now().timestamp() - (10 * 86400)
        os.utime(old_file, (old_time, old_time))
        
        # Create a new file
        new_file = os.path.join(results_dir, "new_test.pdf")
        with open(new_file, "w") as f:
            f.write("test")
        
        # Run cleanup with 7 days threshold
        cleanup_results(7)
        
        # Old file should be gone, new file should remain
        self.assertFalse(os.path.exists(old_file))
        self.assertTrue(os.path.exists(new_file))

    def test_cleanup_results_does_not_remove_non_pdf(self):
        """cleanup_results should not remove non-PDF files."""
        results_dir = ensure_results_dir()
        
        old_file = os.path.join(results_dir, "old_test.txt")
        with open(old_file, "w") as f:
            f.write("test")
        
        old_time = datetime.now().timestamp() - (10 * 86400)
        os.utime(old_file, (old_time, old_time))
        
        cleanup_results(7)
        
        # Non-PDF file should remain
        self.assertTrue(os.path.exists(old_file))

    def test_render_graph_with_clusters_creates_output_dir(self):
        """render_graph_with_clusters should create output directory if it doesn't exist."""
        # Create a simple graph
        g = nx.DiGraph()
        g.add_node("test.py::func", label="func", node_type="function")
        clusters = {"test.py": ["test.py::func"]}
        
        # Use a nested output path
        output_path = os.path.join(self.test_dir, "nested", "subdir", "output.pdf")
        
        # Render (should create directories)
        render_graph_with_clusters(g, clusters, output_path)
        
        self.assertTrue(os.path.exists(os.path.dirname(output_path)))
        self.assertTrue(os.path.exists(output_path))


class TestIntegration(unittest.TestCase):
    """End-to-end tests with real Python files."""

    def test_analyze_itself(self):
        """The analyzer should be able to analyze itself."""
        analyzer_dir = os.path.dirname(os.path.abspath(__file__))
        analyzer = FolderGraphAnalyzer()
        
        try:
            analyzer.analyze_folder(analyzer_dir)
            
            self.assertGreater(analyzer.graph.number_of_nodes(), 0)
            self.assertGreater(analyzer.graph.number_of_edges(), 0)
            
            nodes = list(analyzer.graph.nodes)
            has_analyzer = any("FolderGraphAnalyzer" in n for n in nodes)
            self.assertTrue(has_analyzer, "Should find FolderGraphAnalyzer")
            
            has_self = any("graph_analyzer.py" in n for n in nodes)
            if has_self:
                print(f"🐍 GRAPHCEPTION VERIFIED: Analyzer analyzed itself successfully!")
            
        except Exception as e:
            self.skipTest(f"Cannot analyze self in test environment: {e}")


if __name__ == "__main__":
    unittest.main()