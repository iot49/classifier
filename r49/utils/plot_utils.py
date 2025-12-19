try:
    import tkinter as tk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    tk = None
    FigureCanvasTkAgg = None

class ScrollablePlot:
    def __init__(self, fig):
        if tk is None or FigureCanvasTkAgg is None:
            raise ImportError("Tkinter or matplotlib.backends.backend_tkagg not available.")
            
        self.root = tk.Tk()
        self.root.wm_title("R49 Results")
        
        # Sizing
        w, h = fig.get_size_inches()
        dpi = fig.get_dpi()
        # pixel size:
        # pw = int(w * dpi)  # Unused
        ph = int(h * dpi)
        
        # Screen size check (optional, but good for scrollbar need)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        
        # Canvas frame
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbars for both axes
        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Embed figure
        canvas_fig = FigureCanvasTkAgg(fig, master=canvas)
        canvas_fig.draw()
        fig_widget = canvas_fig.get_tk_widget()
        
        # Add to canvas
        canvas.create_window((0, 0), window=fig_widget, anchor="nw")
        
        # Configure scrolling
        fig_widget.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Set window size (limited by screen)
        # Add some padding for window borders/scrollbars
        width = int(sw * 0.8)
        height = min(ph + 50, sh - 100)
        self.root.geometry(f"{width}x{height}")
        
    def show(self):
        self.root.mainloop()
