# GraphAdapt

A static call graph analyzer for Python that maps code architecture across files, with interactive visualization and self-analysis capabilities.

## Features

- Cross-file call resolution - Tracks function calls across module boundaries
- Class method disambiguation - Properly scopes methods to their owning classes
- Callback detection - Identifies function references passed as arguments, including lambda-wrapped callbacks
- Hierarchical visualization - Nested file/class clustering with color-coded edge types
- Interactive GUI - Filter by edge type, file, toggle labels and class boundaries
- Self-analysis - Run --selfie to generate a call graph of the analyzer itself

## Installation

pip install networkx matplotlib

## Usage

### CLI

# Analyze a project
python graph_analyzer.py /path/to/project

# Analyze current directory
python graph_analyzer.py

# Self-analysis
python graph_analyzer.py --selfie

# Custom output location
python graph_analyzer.py /path/to/project -o my_graph.pdf

# Clean up results older than 7 days
python graph_analyzer.py --cleanup 7

# Show help
python graph_analyzer.py --help

### GUI

python graph_gui.py

### Tests

pytest test_graph_analyzer.py -v

## Output

All PDFs are saved to a results/ folder with timestamps:
- project_multi_file_call_graph_20260115_143022.pdf
- project_selfie_20260115_143045.pdf

## Edge Types

| Type | Color | Style | Description |
| owns | Red | Solid | Class owns its methods |
| call | Gray | Solid | Internal/standard function call |
| external_method | Green | Dotted | Cross-file or class instantiation call |
| reference | Red | Dotted | Callback or function reference |

## Project Structure

GraphAdapt/
├── graph_analyzer.py          # Core analyzer + CLI
├── graph_gui.py               # Interactive Tkinter GUI
├── test_graph_analyzer.py     # Unit and integration tests
├── pytest.ini                 # Test configuration
└── results/                   # Generated PDFs

## How It Works

1. Pass 1: Catalog all class, function, and method definitions across all files
2. Pass 2: Traverse the AST to resolve calls and build graph edges
3. Layout: Compute hierarchical positions with file and class clustering
4. Render: Generate PDF or display in interactive GUI

## Limitations

- Does not resolve imported aliases (e.g., from module import func as f)
- Does not infer types for external objects (self.engine.method() falls back to global lookup)
- Static analysis only - no runtime information

## Issues

- Vizualization is a problem to be solved in itself
- Tests are incomplete

## Requirements

- Python 3.10+
- networkx
- matplotlib
- tkinter (included with Python)

## License

MIT