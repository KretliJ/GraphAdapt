"""
Interactive GUI for GraphAdapt's call graph analyzer.

Pick a folder, crawl it, then tick edge types / files / labels on and off
and watch the preview update live. Export whatever you're currently
looking at to a PDF with one click.

Run:
    python graph_gui.py
"""
import datetime
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import argparse

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from graph_analyzer import FolderGraphAnalyzer, compute_layout, draw_graph, ALL_EDGE_TYPES, ensure_results_dir, generate_output_filename

EDGE_TYPE_LABELS = {
    "owns": "Class Owns Method",
    "call": "Internal / Standard Call",
    "external_method": "External Method Call",
    "reference": "Callback / Reference",
}


class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GraphAdapt - Interactive Call Graph Viewer")
        self.root.geometry("1400x850")

        # Analysis state, populated after a folder is loaded
        self.analyzer = None
        self.pos = {}
        self.file_bounds = {}
        self.class_bounds = {}

        # Tk control variables
        self.edge_type_vars = {et: tk.BooleanVar(value=True) for et in ALL_EDGE_TYPES}
        self.show_edge_labels_var = tk.BooleanVar(value=True)
        self.show_class_boundaries_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
        self.file_vars = {}  # filled in dynamically once a folder is loaded

        self._build_ui()

    def _auto_load_selfie(self):
        """auto-load the analyzer itself."""
        self.status_var.set("🐍 HUH? Okay. Loading myself...")
        self.btn_load.config(state=tk.DISABLED)
        
        folder = os.path.dirname(os.path.abspath(__file__)) or "."
        threading.Thread(target=self._load_folder_thread, args=(folder,), daemon=True).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(toolbar, text="Load Folder...", command=self.pick_folder)
        self.btn_load.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_export = ttk.Button(toolbar, text="Export to PDF", command=self.export_pdf, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="No folder loaded yet.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=16)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Left sidebar: filter controls ---
        sidebar = ttk.Frame(body, padding=10, width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="Edge Types", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        for et in ["owns", "call", "external_method", "reference"]:
            cb = ttk.Checkbutton(
                sidebar, text=EDGE_TYPE_LABELS[et], variable=self.edge_type_vars[et],
                command=self.refresh_preview
            )
            cb.pack(anchor=tk.W)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(sidebar, text="Display", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Checkbutton(sidebar, text="Edge labels", variable=self.show_edge_labels_var,
                         command=self.refresh_preview).pack(anchor=tk.W)
        ttk.Checkbutton(sidebar, text="Class boundary boxes", variable=self.show_class_boundaries_var,
                         command=self.refresh_preview).pack(anchor=tk.W)
        ttk.Checkbutton(sidebar, text="Legend", variable=self.show_legend_var,
                         command=self.refresh_preview).pack(anchor=tk.W)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(sidebar, text="Files", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.files_frame = ttk.Frame(sidebar)
        self.files_frame.pack(fill=tk.X)
        ttk.Label(self.files_frame, text="(load a folder first)", foreground="#777777").pack(anchor=tk.W)

        # --- Right side: matplotlib preview ---
        preview_frame = ttk.Frame(body)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=preview_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav_toolbar = NavigationToolbar2Tk(self.canvas, preview_frame)
        nav_toolbar.update()

        self.canvas.draw()

    # ------------------------------------------------------------------
    # Folder loading
    # ------------------------------------------------------------------
    def pick_folder(self):
        folder = filedialog.askdirectory(title="Select project folder")
        if not folder:
            return
        self.btn_load.config(state=tk.DISABLED)
        self.status_var.set(f"Crawling {folder} ...")
        threading.Thread(target=self._load_folder_thread, args=(folder,), daemon=True).start()

    def _load_folder_thread(self, folder):
        try:
            analyzer = FolderGraphAnalyzer()
            analyzer.analyze_folder(folder)
            pos, file_bounds, class_bounds = compute_layout(analyzer.graph, analyzer.file_clusters)
        except Exception as e:
            self.root.after(0, lambda: self._load_failed(e))
            return
        self.root.after(0, lambda: self._load_succeeded(folder, analyzer, pos, file_bounds, class_bounds))

    def _load_failed(self, error):
        self.btn_load.config(state=tk.NORMAL)
        self.status_var.set("Failed to load folder.")
        messagebox.showerror("Error", f"Could not analyze folder:\n{error}")

    def _load_succeeded(self, folder, analyzer, pos, file_bounds, class_bounds):
        self.analyzer = analyzer
        self.pos = pos
        self.file_bounds = file_bounds
        self.class_bounds = class_bounds

        # Rebuild the per-file checkboxes
        for child in self.files_frame.winfo_children():
            child.destroy()
        self.file_vars = {}
        for filename in file_bounds.keys():
            var = tk.BooleanVar(value=True)
            self.file_vars[filename] = var
            ttk.Checkbutton(self.files_frame, text=filename, variable=var,
                             command=self.refresh_preview).pack(anchor=tk.W)

        self.btn_load.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.NORMAL)

                # Check if we are in self-analysis
        is_selfie = (os.path.basename(os.path.abspath(folder)) == "graph_analyzer" or 
                    os.path.basename(os.path.abspath(folder)) == ".")
        
        if is_selfie and analyzer.graph.number_of_nodes() > 0:
            self.status_var.set(
                f"🐍 GRAPHCEPTION: I see {analyzer.graph.number_of_nodes()} nodes and "
                f"{analyzer.graph.number_of_edges()} edges. I am beautiful."
            )
        else:
            self.status_var.set(
                f"Loaded {os.path.basename(os.path.abspath(folder))} - "
                f"{analyzer.graph.number_of_nodes()} nodes, {analyzer.graph.number_of_edges()} edges."
            )
        self.refresh_preview()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _current_filters(self):
        edge_types = {et for et, var in self.edge_type_vars.items() if var.get()}
        visible_files = {f for f, var in self.file_vars.items() if var.get()}
        return dict(
            edge_types=edge_types,
            visible_files=visible_files,
            show_edge_labels=self.show_edge_labels_var.get(),
            show_class_boundaries=self.show_class_boundaries_var.get(),
            show_legend=self.show_legend_var.get(),
        )

    def refresh_preview(self):
        if self.analyzer is None or not self.pos:
            return
        filters = self._current_filters()
        draw_graph(self.ax, self.analyzer.graph, self.pos, self.file_bounds, self.class_bounds, **filters)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_pdf(self):
        if self.analyzer is None:
            return
        
        # Use results folder with timestamp
        results_dir = ensure_results_dir()
        base_name = "call_graph"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = os.path.join(results_dir, f"{base_name}_{timestamp}.pdf")
        
        path = filedialog.asksaveasfilename(
            title="Export to PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=os.path.basename(default_filename),
            initialdir=results_dir,
        )
        if not path:
            return

        filters = self._current_filters()
        export_fig = Figure(figsize=(24, 18))
        export_ax = export_fig.add_subplot(111)
        draw_graph(export_ax, self.analyzer.graph, self.pos, self.file_bounds, self.class_bounds, **filters)
        export_fig.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            export_fig.savefig(path, format="pdf", bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save PDF:\n{e}")
            return
        self.status_var.set(f"Exported to {path}")
        messagebox.showinfo("Exported", f"Saved to:\n{path}")


def main():
    parser = argparse.ArgumentParser(
        description="GraphAdapt GUI - Interactive Call Graph Viewer"
    )
    parser.add_argument(
        "--selfie",
        action="store_true",
        help="Load the analyzer itself on startup (GRAPHCEPTION mode)"
    )
    args = parser.parse_args()
    
    root = tk.Tk()
    app = GraphApp(root)
    
    if args.selfie:
        # Auto-load the current directory with selfie flair
        root.after(100, lambda: app._auto_load_selfie())
    
    root.mainloop()

if __name__ == "__main__":
    main()