# integrated_aq10_app.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import platform
import os
import sys
from pathlib import Path
from datetime import datetime

# ML imports
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
        roc_curve, confusion_matrix  # <-- added confusion_matrix
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# matplotlib for embedding
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ------------------------
# PATH HELPERS (robust for .py and PyInstaller EXE)
# ------------------------
def _resource_base_dir():
    """Folder containing this script, or the EXE when frozen."""
    if getattr(sys, "frozen", False):            # PyInstaller
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = _resource_base_dir()

# ------------------------
# CONFIG
# ------------------------
NEW_DATA_PATH = "new_data.csv"
DATASET_BASENAME = "train.csv"
TRAIN_CLEANED = "train_cleaned.csv"   # used for training SVM/XGBoost
BACKUP_BASENAME  = "train.backup.csv"

DATASET_PATH = os.path.join(BASE_DIR, DATASET_BASENAME)
TRAIN_CLEANED_PATH = os.path.join(BASE_DIR, TRAIN_CLEANED)
NEW_DATA_ABS = os.path.join(BASE_DIR, NEW_DATA_PATH)
BACKUP_PATH  = os.path.join(BASE_DIR, BACKUP_BASENAME)

REQUIRED_HEADERS = [
    "ID",
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "age","gender","ethnicity","jaundice","austim","contry_of_res",
    "used_app_before","result","age_desc","relation","Class/ASD"
]
EXTRA_HEADERS = ["screening_result"]

# ------------------------ ONE-TIME MERGE ------------------------
def merge_new_data_once():
    master_file = Path(DATASET_PATH)
    new_file = Path(NEW_DATA_ABS)

    if not new_file.exists():
        return

    if SKLEARN_AVAILABLE:
        if master_file.exists():
            try:
                df_master = pd.read_csv(master_file)
            except Exception:
                df_master = pd.DataFrame(columns=REQUIRED_HEADERS + EXTRA_HEADERS)
        else:
            df_master = pd.DataFrame(columns=REQUIRED_HEADERS + EXTRA_HEADERS)
            df_master.to_csv(master_file, index=False)
    else:
        if not master_file.exists():
            with open(master_file, "w", encoding="utf-8") as f:
                f.write(",".join(REQUIRED_HEADERS + EXTRA_HEADERS) + "\n")

    if not SKLEARN_AVAILABLE:
        try:
            messagebox.showwarning(
                "Merge skipped",
                "pandas is not installed, so the automatic merge is skipped.\n\n"
                "Install with: pip install pandas\nThen run again to merge your new CSV."
            )
        except Exception:
            print("Merge skipped: pandas not installed.")
        return

    df_master = pd.read_csv(master_file)
    df_new = pd.read_csv(new_file)
    Path(BACKUP_PATH).write_bytes(master_file.read_bytes())

    master_cols = list(df_master.columns)
    new_cols = [c for c in df_new.columns if c not in master_cols]
    all_cols = master_cols + new_cols

    for c in all_cols:
        if c not in df_master.columns:
            df_master[c] = ""
        if c not in df_new.columns:
            df_new[c] = ""

    df_master = df_master[all_cols]
    df_new = df_new[all_cols]

    if "ID" not in all_cols:
        df_master.insert(0, "ID", range(1, len(df_master) + 1))
        all_cols = ["ID"] + [c for c in all_cols if c != "ID"]

    max_id = pd.to_numeric(df_master["ID"], errors="coerce").max()
    if pd.isna(max_id):
        max_id = len(df_master)
    start_id = int(max_id) + 1
    df_new["ID"] = range(start_id, start_id + len(df_new))

    if "age_desc" in df_new.columns:
        df_new["age_desc"] = (
            df_new["age_desc"].astype(str).str.strip()
            .replace({"18>": "18 and more", "18+": "18 and more"})
        )
    if "jaundice" in df_new.columns:
        df_new["jaundice"] = (
            df_new["jaundice"].astype(str).str.strip().str.lower()
            .replace({"yes": "Yes", "no": "No"})
        )

    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    df_combined.to_csv(master_file, index=False)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    merged_name = new_file.with_suffix(f".merged_{ts}.csv")
    new_file.rename(merged_name)

    print("✅ Merge complete.")
    print(f"   Added rows:     {len(df_new)}")
    print(f"   New total:      {len(df_combined)}")
    print(f"   Last ID:        {int(df_combined['ID'].iloc[-1])}")
    print(f"   Next ID:        {int(df_combined['ID'].iloc[-1]) + 1}")
    print(f"   Backup saved:   {BACKUP_PATH}")
    print(f"   Renamed source: {merged_name.name}")

# ------------------------ THEMING ------------------------
WINDOW_BG = "#0a0a0a"
FRAME_BG  = "#121212"
CARD_BG   = "#171717"
ACCENT    = "#00b4d8"
TEXT_PRIMARY   = "#ffffff"
TEXT_SECONDARY = "#cfcfcf"

FONT_TITLE    = ("Helvetica", 20, "bold")
FONT_SUBTITLE = ("Helvetica", 12)
FONT_TEXT     = ("Helvetica", 11)
FONT_QUESTION = ("Helvetica", 12, "bold")

PUZZLE_COLORS = ["#00b4d8", "#90e0ef", "#0077b6", "#48cae4"]

# ------------------------ APP ------------------------
class AQ10App:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 AQ-10 Autism Test (Integrated Models)")
        self.root.configure(bg=WINDOW_BG)
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        # Background canvas
        self.bg_canvas = tk.Canvas(root, bg=WINDOW_BG, highlightthickness=0, bd=0)
        self.bg_canvas.place(relwidth=1, relheight=1)
        self._draw_static_puzzle_background()

        # Pages
        self.page1 = tk.Frame(root, bg=FRAME_BG)
        self.page2 = tk.Frame(root, bg=FRAME_BG)
        self.page3 = tk.Frame(root, bg=FRAME_BG)
        self._place_pages()
        self.current_page = self.page1

        # Data & ML holders
        self.df = None
        self.model_note = ""
        self.model_ready = False
        self.train_cleaned_override = None  # user-selected file path if they browse

        # user latest submission stored for Page3 prediction
        self.latest_submission = None

        # Load dataset (basic)
        self._load_dataset()

        # Questions
        self.questions = [
            "I often notice small sounds when others do not.",
            "I usually concentrate more on the whole picture, rather than the small details.",
            "I find it easy to do more than one thing at once.",
            "If there is an interruption, I can switch back to what I was doing very quickly.",
            "I find it easy to ‘read between the lines’ when someone is talking to me.",
            "I know how to tell if someone listening to me is getting bored.",
            "When I’m reading a story, I find it difficult to work out the characters’ intentions.",
            "I like to collect information about categories of things (e.g., cars, birds, trains, plants, etc.).",
            "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
            "I find it difficult to work out people’s intentions."
        ]
        # Items 2,3,4,5,6,9 are reverse-scored (score 1 for DISAGREE)
        self.disagree_score_items = {2, 3, 4, 5, 6, 9}
        self.responses = {}

        # UI
        self._init_styles()
        self._create_page1()
        self._create_page2()
        self._create_page3()
        self.show_page(self.page1)

        self.root.bind("<Configure>", self._on_resize)

    def _init_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=FRAME_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Header.TLabel", background=FRAME_BG, foreground=TEXT_PRIMARY, font=FONT_TITLE)
        style.configure("Subheader.TLabel", background=FRAME_BG, foreground=TEXT_SECONDARY, font=FONT_SUBTITLE)
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=TEXT_PRIMARY, font=FONT_QUESTION)
        style.configure("CardText.TLabel", background=CARD_BG, foreground=TEXT_SECONDARY, font=FONT_TEXT)
        style.configure("Primary.TButton", font=FONT_TEXT, padding=8)
        style.map("Primary.TButton", background=[("!disabled", ACCENT)], foreground=[("!disabled", "#ffffff")])
        style.configure("Secondary.TButton", font=FONT_TEXT, padding=8, background="#2a2a2a", foreground="#ffffff")
        style.map("Secondary.TButton", background=[("!disabled", "#2a2a2a")], foreground=[("!disabled", "#ffffff")])
        style.configure("AQ.TRadiobutton", background=CARD_BG, foreground=TEXT_PRIMARY, font=FONT_TEXT)
        style.configure("Entry.TEntry", fieldbackground="#1e1e1e", background="#1e1e1e", foreground="#ffffff")

    def _draw_static_puzzle_background(self):
        self.bg_canvas.delete("all")
        w = max(800, self.root.winfo_width() or self.root.winfo_screenwidth())
        h = max(600, self.root.winfo_height() or self.root.winfo_screenheight())
        rng = random.Random(42)
        for _ in range(22):
            size = rng.randint(28, 56)
            x = rng.randint(40, w - 40)
            y = rng.randint(40, h - 40)
            color = rng.choice(PUZZLE_COLORS)
            self._draw_piece(x, y, size, color)

    def _draw_piece(self, x, y, size, color):
        r = max(6, int(size * 0.18))
        left, top = x - size // 2, y - size // 2
        right, bottom = x + size // 2, y + size // 2
        c = self.bg_canvas
        c.create_oval(left, top, left + 2*r, top + 2*r, fill=color, outline=color)
        c.create_oval(right - 2*r, top, right, top + 2*r, fill=color, outline=color)
        c.create_oval(left, bottom - 2*r, left + 2*r, bottom, fill=color, outline=color)
        c.create_oval(right - 2*r, bottom - 2*r, right, bottom, fill=color, outline=color)
        c.create_rectangle(left + r, top, right - r, bottom, fill=color, outline=color)
        c.create_rectangle(left, top + r, right, bottom - r, fill=color, outline=color)

    def _place_pages(self):
        w = self.root.winfo_width() or self.root.winfo_screenwidth()
        max_width = min(1100, int(w * 0.86))
        for p in (self.page1, self.page2, self.page3):
            p.place(relx=0.5, rely=0, anchor="n", y=20, width=max_width)

    def _on_resize(self, _):
        self._draw_static_puzzle_background()
        self._place_pages()
        self.current_page.tkraise()

    # ---------- Page 1 ----------
    def _create_page1(self):
        container = ttk.Frame(self.page1, padding=20, style="TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="🧠 Autism Spectrum Quotient (AQ-10)", style="Header.TLabel").pack(anchor="w", pady=(0,6))
        ttk.Label(container, text="First, some basic information.", style="Subheader.TLabel").pack(anchor="w", pady=(0,12))

        card = ttk.Frame(container, padding=12, style="Card.TFrame")
        card.pack(fill="x", expand=False, pady=(0,12))

        self.gender_var    = tk.StringVar(value="Select…")
        self.ethnicity_var = tk.StringVar(value="Select…")
        self.age_var       = tk.StringVar(value="")
        self.country_var   = tk.StringVar(value="")
        self.jaundice_var  = tk.StringVar(value="Select…")
        self.relation_var  = tk.StringVar(value="Select…")

        for i in range(4):
            card.columnconfigure(i, weight=1)

        ttk.Label(card, text="Gender", style="CardText.TLabel").grid(row=0, column=0, sticky="w")
        gender_menu = ttk.OptionMenu(card, self.gender_var, self.gender_var.get(), "Male", "Female")
        gender_menu.grid(row=1, column=0, sticky="we", padx=(0,8), pady=(6,6))

        ttk.Label(card, text="Ethnicity", style="CardText.TLabel").grid(row=0, column=1, sticky="w")
        ethnicity_menu = ttk.OptionMenu(card, self.ethnicity_var, self.ethnicity_var.get(),
                                        "White-European","Black","Asian","Latino","Middle Eastern",
                                        "South Asian","Hispanic","Pasifika","Other")
        ethnicity_menu.grid(row=1, column=1, sticky="we", padx=(0,8), pady=(6,6))

        ttk.Label(card, text="Age (years)", style="CardText.TLabel").grid(row=0, column=2, sticky="w")
        age_entry = ttk.Entry(card, textvariable=self.age_var, style="Entry.TEntry")
        age_entry.grid(row=1, column=2, sticky="we", padx=(0,8), pady=(6,6))

        ttk.Label(card, text="Country of residence", style="CardText.TLabel").grid(row=2, column=0, sticky="w", pady=(8,0))
        ttk.Entry(card, textvariable=self.country_var, style="Entry.TEntry").grid(row=3, column=0, sticky="we", padx=(0,8), pady=(6,6))

        ttk.Label(card, text="Jaundice (as an infant)", style="CardText.TLabel").grid(row=2, column=1, sticky="w", pady=(8,0))
        jaundice_menu = ttk.OptionMenu(card, self.jaundice_var, self.jaundice_var.get(), "Yes","No")
        jaundice_menu.grid(row=3, column=1, sticky="we", padx=(0,8), pady=(6,6))

        ttk.Label(card, text="Relation", style="CardText.TLabel").grid(row=2, column=2, sticky="w", pady=(8,0))
        relation_menu = ttk.OptionMenu(card, self.relation_var, self.relation_var.get(), "Self","Parent","Guardian")
        relation_menu.grid(row=3, column=2, sticky="we", padx=(0,8), pady=(6,6))

        # Questions area (scrollable)
        sc = ttk.Frame(container, style="TFrame")
        sc.pack(fill="both", expand=True)

        self.q_canvas = tk.Canvas(sc, bg=FRAME_BG, highlightthickness=0, bd=0)
        self.q_canvas.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(sc, orient="vertical", command=self.q_canvas.yview)
        sb.pack(side="right", fill="y")
        self.q_canvas.configure(yscrollcommand=sb.set)
        self.scroll_frame = ttk.Frame(self.q_canvas, style="TFrame")
        self.scroll_window = self.q_canvas.create_window((0,0), window=self.scroll_frame, anchor="nw")
        self.q_canvas.bind("<Configure>", lambda e: self.q_canvas.itemconfigure(self.scroll_window, width=e.width))
        self.scroll_frame.bind("<Configure>", lambda e: self.q_canvas.configure(scrollregion=self.q_canvas.bbox("all")))

        # enable wheel scrolling when hovering
        self.scroll_frame.bind("<Enter>", lambda e: self._bind_to_mousewheel())
        self.scroll_frame.bind("<Leave>", lambda e: self._unbind_from_mousewheel())

        for i, q in enumerate(self.questions, start=1):
            card_q = ttk.Frame(self.scroll_frame, padding=10, style="Card.TFrame")
            card_q.pack(fill="x", expand=True, pady=(6,6))
            ttk.Label(card_q, text=f"{i}. {q}", style="CardTitle.TLabel", wraplength=900, justify="left").pack(anchor="w")
            var = tk.StringVar(value="")
            self.responses[i] = var

            # update counter whenever a choice is made
            var.trace_add("write", lambda *_: self._update_progress())

            row = ttk.Frame(card_q, style="Card.TFrame")
            row.pack(anchor="w", pady=(6,0))

            # 4-point AQ-10 options; scoring collapses to Agree/Disagree later
            ttk.Radiobutton(row, text="Definitely agree",   value="DA", variable=var, style="AQ.TRadiobutton").pack(side="left", padx=(0,12))
            ttk.Radiobutton(row, text="Slightly agree",     value="SA", variable=var, style="AQ.TRadiobutton").pack(side="left", padx=(0,12))
            ttk.Radiobutton(row, text="Slightly disagree",  value="SD", variable=var, style="AQ.TRadiobutton").pack(side="left", padx=(0,12))
            ttk.Radiobutton(row, text="Definitely disagree",value="DD", variable=var, style="AQ.TRadiobutton").pack(side="left")

        footer = ttk.Frame(container, padding=8, style="TFrame")
        footer.pack(fill="x")
        ttk.Button(footer, text="Submit →", style="Primary.TButton", command=self._submit_and_show_results).pack(side="right", padx=(6,0))
        ttk.Button(footer, text="Quit", style="Secondary.TButton", command=self.root.destroy).pack(side="left")

        self.progress_var = tk.StringVar(value="Answered: 0 / 10")
        ttk.Label(container, textvariable=self.progress_var, style="Subheader.TLabel").pack(anchor="w", pady=(6,0))

    # mouse wheel support
    def _bind_to_mousewheel(self):
        sysname = platform.system()
        if sysname in ("Windows", "Darwin"):  # Windows & macOS
            self.q_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        else:  # Linux/X11
            self.q_canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.q_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_from_mousewheel(self):
        self.q_canvas.unbind_all("<MouseWheel>")
        self.q_canvas.unbind_all("<Button-4>")
        self.q_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        sysname = platform.system()
        if sysname == "Windows":
            delta = -1 * int(event.delta / 120)
            self.q_canvas.yview_scroll(delta, "units")
        elif sysname == "Darwin":
            # macOS delta is small but sign is the same
            delta = -1 if event.delta > 0 else 1
            self.q_canvas.yview_scroll(delta, "units")
        else:
            # Linux uses button numbers
            delta = -1 if getattr(event, "num", None) == 4 else 1
            self.q_canvas.yview_scroll(delta, "units")

    # ---------- Page 2 ----------
    def _create_page2(self):
        container = ttk.Frame(self.page2, padding=20, style="TFrame")
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="🧾 Personalized Screening Result", style="Header.TLabel").pack(anchor="w", pady=(0,6))
        ttk.Label(container, text="A quick visual summary of your AQ-10 result.", style="Subheader.TLabel").pack(anchor="w", pady=(0,8))

        self.results_card = ttk.Frame(container, padding=12, style="Card.TFrame")
        self.results_card.pack(fill="both", expand=True, pady=(6,12))

        self.chart_holder = ttk.Frame(self.results_card, style="Card.TFrame")
        self.chart_holder.pack(fill="both", expand=True)

        self.details_var = tk.StringVar(value="")
        ttk.Label(self.results_card, textvariable=self.details_var, style="CardText.TLabel", wraplength=900, justify="left").pack(anchor="w", pady=(8,6))

        actions = ttk.Frame(container, style="TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="← Back", style="Secondary.TButton", command=lambda: self.show_page(self.page1)).pack(side="left")
        ttk.Button(actions, text="Next → Models", style="Primary.TButton", command=self._go_to_models_page).pack(side="right")

    # ---------- Page 3 ----------
    def _create_page3(self):
        container = ttk.Frame(self.page3, padding=12, style="TFrame")
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="🔬 Predictive Model Comparison", style="Header.TLabel").pack(anchor="w", pady=(0,6))
        ttk.Label(container, text="SVM vs XGBoost — ROC curves and predictions for your submission", style="Subheader.TLabel").pack(anchor="w", pady=(0,8))

        self.models_card = ttk.Frame(container, padding=10, style="Card.TFrame")
        self.models_card.pack(fill="both", expand=True, pady=(6,12))

        # Plot holder
        self.models_plot_holder = ttk.Frame(self.models_card, style="Card.TFrame")
        self.models_plot_holder.pack(fill="both", expand=True, pady=(6,6))

        # Metrics & predictions text
        self.models_text = tk.Text(self.models_card, height=12, bg="#171717", fg="#cfcfcf", wrap="word")
        self.models_text.pack(fill="x", padx=6, pady=(6,6))
        self.models_text.configure(state="disabled")

        # actions
        actions = ttk.Frame(container, style="TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="Restart → Back to start", style="Secondary.TButton", command=self._restart).pack(side="left")
        ttk.Button(actions, text="Close", style="Primary.TButton", command=self.root.destroy).pack(side="right")

        # helper row if training csv is missing
        self.missing_csv_row = ttk.Frame(container, padding=6, style="TFrame")
        self.missing_csv_row.pack(fill="x")
        ttk.Label(self.missing_csv_row, text="", style="Subheader.TLabel").pack(side="left", padx=(0,8))
        self.browse_btn = ttk.Button(self.missing_csv_row, text="Browse training CSV…", style="Secondary.TButton",
                                     command=self._browse_train_csv)
        self.browse_btn.pack(side="left")
        self.missing_csv_row.pack_forget()  # hidden by default

    def _browse_train_csv(self):
        path = filedialog.askopenfilename(
            title="Select train_cleaned.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.train_cleaned_override = path
            messagebox.showinfo("Training file set", f"Using:\n{path}\n\nYou can change it by browsing again.")
            # re-run model training if we're already on the models page
            self._train_and_display_models()

    # ---------- load basic dataset ----------
    def _load_dataset(self):
        if SKLEARN_AVAILABLE:
            if os.path.exists(DATASET_PATH):
                try:
                    self.df = pd.read_csv(DATASET_PATH)
                except Exception:
                    self.df = pd.DataFrame(columns=REQUIRED_HEADERS + EXTRA_HEADERS)
            else:
                self.df = pd.DataFrame(columns=REQUIRED_HEADERS + EXTRA_HEADERS)
                self.df.to_csv(DATASET_PATH, index=False)
        else:
            if not os.path.exists(DATASET_PATH):
                with open(DATASET_PATH, "w", encoding="utf-8") as f:
                    f.write(",".join(REQUIRED_HEADERS + EXTRA_HEADERS) + "\n")
            self.df = None

    # ---------- helper: append to dataset ----------
    def _append_to_dataset(self, row_dict):
        if not SKLEARN_AVAILABLE:
            max_id = 0
            lines = []
            if os.path.exists(DATASET_PATH):
                try:
                    with open(DATASET_PATH, "r", encoding="utf-8") as f:
                        lines = f.read().splitlines()
                    for ln in lines[1:]:
                        parts = ln.split(",")
                        if parts:
                            try:
                                max_id = max(max_id, int(parts[0]))
                            except Exception:
                                pass
                    current_rows = max(0, len(lines) - 1)
                except Exception:
                    lines = []
                    current_rows = 0
            else:
                current_rows = 0

            next_id = max_id + 1
            row_dict = dict(row_dict)
            if str(row_dict.get("ID", "")).strip() == "":
                row_dict["ID"] = next_id
            assigned_id = int(row_dict["ID"])

            if not os.path.exists(DATASET_PATH):
                with open(DATASET_PATH, "w", encoding="utf-8") as f:
                    f.write(",".join(REQUIRED_HEADERS + EXTRA_HEADERS) + "\n")
            with open(DATASET_PATH, "a", encoding="utf-8") as f:
                ordered = [str(row_dict.get(h, "")) for h in REQUIRED_HEADERS + EXTRA_HEADERS]
                f.write(",".join(ordered) + "\n")
            return assigned_id, current_rows + 1

        try:
            if os.path.exists(DATASET_PATH):
                df_now = pd.read_csv(DATASET_PATH)
            else:
                df_now = pd.DataFrame(columns=REQUIRED_HEADERS + EXTRA_HEADERS)
        except Exception:
            df_now = pd.DataFrame(columns=REQUIRED_HEADERS + EXTRA_HEADERS)

        for col in REQUIRED_HEADERS + EXTRA_HEADERS:
            if col not in df_now.columns:
                df_now[col] = ""

        n_old = len(df_now)
        try:
            max_id = pd.to_numeric(df_now["ID"], errors="coerce").max()
            next_id = int(max_id) + 1 if pd.notna(max_id) else 1
        except Exception:
            next_id = n_old + 1

        row_dict = dict(row_dict)
        if str(row_dict.get("ID", "")).strip() == "":
            row_dict["ID"] = next_id
        assigned_id = int(row_dict["ID"])

        for col in df_now.columns:
            if col not in row_dict:
                row_dict[col] = ""

        df_now.loc[len(df_now)] = {c: row_dict.get(c, "") for c in df_now.columns}

        ordered_cols = [c for c in REQUIRED_HEADERS] + [c for c in EXTRA_HEADERS] + [c for c in df_now.columns if c not in REQUIRED_HEADERS + EXTRA_HEADERS]
        df_now = df_now[ordered_cols]
        df_now.to_csv(DATASET_PATH, index=False)
        self.df = df_now
        return assigned_id, n_old + 1

    # ---------- submission ----------
    def _submit_and_show_results(self):
        errors = []
        country = self.country_var.get().strip()
        age_raw = self.age_var.get().strip()
        if country == "":
            errors.append("Please enter your country of residence on Page 1.")
        if age_raw == "" or not age_raw.isdigit() or int(age_raw) < 0:
            errors.append("Please enter a valid non-negative whole number for age on Page 1.")
        if self.gender_var.get() not in ("Male", "Female"):
            errors.append("Please select a gender (Male or Female) on Page 1.")
        if self.ethnicity_var.get() not in (
            "White-European","Black","Asian","Latino","Middle Eastern","South Asian","Hispanic","Pasifika","Other"):
            errors.append("Please select an ethnicity on Page 1.")
        if self.jaundice_var.get() not in ("Yes", "No"):
            errors.append("Please select jaundice Yes/No on Page 1.")
        if self.relation_var.get() not in ("Self", "Parent", "Guardian"):
            errors.append("Please select a relation on Page 1.")
        if errors:
            messagebox.showwarning("Missing info", "\n".join(errors))
            return

        age_num = int(age_raw)
        age_desc = "0-18" if age_num < 18 else "18 and more"

        # Ensure all questions answered and compute AQ-10 scoring
        A_scores = []
        for i in range(1, 11):
            r = self.responses[i].get()
            if r not in ("DA", "SA", "SD", "DD"):
                messagebox.showwarning("Incomplete", "Please answer all questions on Page 1 before submitting.")
                return
            is_agree = (r in ("DA", "SA"))
            is_disagree = not is_agree
            if i in self.disagree_score_items:
                A_scores.append(1 if is_disagree else 0)
            else:
                A_scores.append(1 if is_agree else 0)

        aq10_total = sum(A_scores)
        screening_class = 1 if aq10_total >= 6 else 0

        # Optional: keep raw selections for audit
        raw_answers = {f"A{i}_Raw": self.responses[i].get() for i in range(1, 11)}

        row = {
            "ID": "",
            "A1_Score": A_scores[0], "A2_Score": A_scores[1], "A3_Score": A_scores[2],
            "A4_Score": A_scores[3], "A5_Score": A_scores[4], "A6_Score": A_scores[5],
            "A7_Score": A_scores[6], "A8_Score": A_scores[7], "A9_Score": A_scores[8],
            "A10_Score": A_scores[9],
            "age": age_num,
            "gender": self.gender_var.get().strip(),
            "ethnicity": self.ethnicity_var.get().strip(),
            "jaundice": self.jaundice_var.get().strip(),
            "austim": "",
            "contry_of_res": country,
            "used_app_before": "yes",
            "result": aq10_total,
            "age_desc": age_desc,
            "relation": self.relation_var.get().strip(),
            "Class/ASD": ""
        }
        row["screening_result"] = screening_class
        row.update(raw_answers)

        try:
            assigned_id, saved_row_num = self._append_to_dataset(row)
        except Exception as e:
            messagebox.showwarning("Save error", f"Could not save to dataset:\n{e}")
            return

        # save latest submission for predictions later
        self.latest_submission = {
            "A_scores": A_scores,
            "age_desc": age_desc,
            "jaundice": self.jaundice_var.get().strip(),
            "ID": assigned_id
        }

        # Show page 2 results
        about = "AQ-10 scoring per NICE: score 1 on items 1,7,8,10 for Agree; items 2,3,4,5,6,9 for Disagree."
        status_text = ("Score ≥ 6 — consider referral for a comprehensive autism assessment."
                       if screening_class == 1 else
                       "Score < 6 — below referral cut-off (use clinical judgement).")
        details = (
            f"AQ-10 Score: {aq10_total}/10\n"
            f"{about}\n\n"
            f"Recommendation: {status_text}\n\n"
            f"This is a screening tool, not a diagnosis. If you have concerns, consult a qualified clinician.\n\n"
            f"Saved to: {DATASET_BASENAME}\nRow: {saved_row_num}   ID: {assigned_id}"
        )
        self._display_results_page(aq10_total, details, screening_class)

    def _display_results_page(self, aq10_total, details, screening_class):
        self.details_var.set(details)
        for w in self.chart_holder.winfo_children():
            w.destroy()

        if MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(6, 3), dpi=100)
            ax = fig.add_subplot(111)
            ax.bar(["Your AQ-10 Score"], [aq10_total], color=ACCENT)
            ax.axhline(6, color="red", linestyle="--", label="Threshold (6)")
            ax.set_ylim(0, 10)
            ax.set_ylabel("AQ-10 Total Score")
            ax.set_title("Autism Screening Result")
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=self.chart_holder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(self.chart_holder, text=f"Your AQ-10 Score: {aq10_total}/10 (Threshold: 6)", style="CardText.TLabel").pack(anchor="center", pady=20)

        self.show_page(self.page2)

    # ---------- NAV TO MODELS ----------
    def _go_to_models_page(self):
        self.show_page(self.page3)
        self.root.after(100, self._train_and_display_models)

    # ---------- TRAIN & DISPLAY MODELS ----------
    def _train_and_display_models(self):
        for w in self.models_plot_holder.winfo_children():
            w.destroy()
        self.models_text.configure(state="normal")
        self.models_text.delete("1.0", tk.END)
        self.models_text.configure(state="disabled")

        train_path = self.train_cleaned_override or TRAIN_CLEANED_PATH

        def show_missing_row(msg):
            for child in self.missing_csv_row.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(text=msg)
                    break
            self.missing_csv_row.pack(fill="x")

        if not SKLEARN_AVAILABLE:
            messagebox.showwarning("ML not available", "pandas/scikit-learn not installed. Models cannot be trained.")
            return

        if not os.path.exists(train_path):
            msg = (f"Training file '{TRAIN_CLEANED}' not found.\n\n"
                   f"I looked here:\n{train_path}\n\n"
                   f"Place the CSV next to this program (folder: {BASE_DIR}) or click 'Browse training CSV…'.")
            show_missing_row(msg)
            messagebox.showwarning("Training data missing", msg)
            return
        else:
            self.missing_csv_row.pack_forget()

        try:
            train_df = pd.read_csv(train_path)
        except Exception as e:
            messagebox.showwarning("Read error", f"Could not read training CSV:\n{e}")
            return

        target_col = "Class/ASD"
        if target_col not in train_df.columns:
            messagebox.showwarning("Target missing", f"Target column '{target_col}' not found in training CSV.")
            return

        # ---- Robust target binarization
        y_series = train_df[target_col].copy()
        def to_bin(val):
            try:
                if pd.isna(val): return 0
                if isinstance(val, (int, float)) and int(val) in (0,1): return int(val)
                s = str(val).strip().lower()
                if s in ("yes","1","true","asd","1.0","y"): return 1
                if s in ("no","0","false","non-asd","non_asd","n"): return 0
                return 1 if float(s) >= 0.5 else 0
            except Exception:
                return 0
        y_series = y_series.map(to_bin).astype(int)

        mask = y_series.isin([0,1])
        Xdf = train_df.loc[mask].copy()
        y = y_series[mask].values
        if Xdf.empty or len(y) < 10:
            messagebox.showwarning("Not enough data", "Not enough labeled rows to train models. Need at least ~10 labeled rows.")
            return

        # Preferred features (AQ-10 items) + helpful context if present
        feature_cols = [f"A{i}_Score" for i in range(1, 11)]
        extra_cols = [c for c in ("age_desc","jaundice","age","gender","ethnicity") if c in Xdf.columns]
        used_cols = [c for c in feature_cols if c in Xdf.columns] + extra_cols

        # Fallback if some A*_Score missing
        if len([c for c in feature_cols if c in Xdf.columns]) < 10:
            numeric_cols_fallback = Xdf.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols_fallback = [c for c in numeric_cols_fallback if c != target_col]
            used_cols = numeric_cols_fallback if numeric_cols_fallback else used_cols

        if not used_cols:
            messagebox.showwarning("No features", "No usable feature columns were found in the training CSV.")
            return

        X = Xdf[used_cols].copy()

        # Identify numeric vs categorical from training data
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        # Preprocessor
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]) if cat_cols else None

        if cat_cols:
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, cat_cols)
            ])
        else:
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_cols)
            ])

        # ---- Split RAW X and fit pipelines on raw data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Pipelines
        svm_pipeline = Pipeline([
            ("pre", preprocessor),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        ])

        svm_metrics = {}
        try:
            svm_pipeline.fit(X_train, y_train)
            svm_probs = svm_pipeline.predict_proba(X_test)[:, 1]
            svm_pred  = (svm_probs >= 0.5).astype(int)
            fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
            svm_cm = confusion_matrix(y_test, svm_pred, labels=[0,1])   # <-- CM
            svm_metrics = {
                "accuracy": accuracy_score(y_test, svm_pred),
                "precision": precision_score(y_test, svm_pred, zero_division=0),
                "recall": recall_score(y_test, svm_pred, zero_division=0),
                "f1": f1_score(y_test, svm_pred, zero_division=0),
                "auc": roc_auc_score(y_test, svm_probs),
                "fpr": fpr_svm, "tpr": tpr_svm,
                "cm": svm_cm
            }
        except Exception as e:
            svm_pipeline = None
            svm_metrics = {"error": str(e)}

        xgb_pipeline = None
        xgb_metrics = {}
        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    learning_rate=0.05,
                    max_depth=5,
                    n_estimators=200,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method="hist"
                )
                xgb_pipeline = Pipeline([
                    ("pre", preprocessor),
                    ("clf", xgb)
                ])
                xgb_pipeline.fit(X_train, y_train)
                xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
                xgb_pred  = (xgb_probs >= 0.5).astype(int)
                fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
                xgb_cm = confusion_matrix(y_test, xgb_pred, labels=[0,1])  # <-- CM
                xgb_metrics = {
                    "accuracy": accuracy_score(y_test, xgb_pred),
                    "precision": precision_score(y_test, xgb_pred, zero_division=0),
                    "recall": recall_score(y_test, xgb_pred, zero_division=0),
                    "f1": f1_score(y_test, xgb_pred, zero_division=0),
                    "auc": roc_auc_score(y_test, xgb_probs),
                    "fpr": fpr_xgb, "tpr": tpr_xgb,
                    "cm": xgb_cm
                }
            except Exception as e:
                xgb_pipeline = None
                xgb_metrics = {"error": str(e)}
        else:
            xgb_metrics = {"error": "xgboost not installed."}

        # ---- Plots (ROC top row, Confusion Matrices bottom row)
        if MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(10, 8), dpi=100)
            ax_roc_svm = fig.add_subplot(221)
            ax_roc_xgb = fig.add_subplot(222)
            ax_cm_svm  = fig.add_subplot(223)
            ax_cm_xgb  = fig.add_subplot(224)

            # --- SVM ROC ---
            if "error" not in svm_metrics:
                ax_roc_svm.plot(svm_metrics["fpr"], svm_metrics["tpr"], label=f"SVM (AUC={svm_metrics['auc']:.2f})")
                ax_roc_svm.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax_roc_svm.set_title("SVM ROC")
                ax_roc_svm.set_xlabel("False Positive Rate")
                ax_roc_svm.set_ylabel("True Positive Rate")
                ax_roc_svm.legend(loc="lower right")
            else:
                ax_roc_svm.text(0.5, 0.5, f"SVM ROC unavailable\n{svm_metrics['error']}", ha='center', va='center')
                ax_roc_svm.set_axis_off()

            # --- XGBoost ROC ---
            if "error" not in xgb_metrics:
                ax_roc_xgb.plot(xgb_metrics["fpr"], xgb_metrics["tpr"], label=f"XGBoost (AUC={xgb_metrics['auc']:.2f})")
                ax_roc_xgb.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax_roc_xgb.set_title("XGBoost ROC")
                ax_roc_xgb.set_xlabel("False Positive Rate")
                ax_roc_xgb.set_ylabel("True Positive Rate")
                ax_roc_xgb.legend(loc="lower right")
            else:
                ax_roc_xgb.text(0.5, 0.5, "XGBoost ROC unavailable", ha='center', va='center')
                ax_roc_xgb.set_axis_off()

            # --- SVM Confusion Matrix ---
            if "error" not in svm_metrics and "cm" in svm_metrics:
                cm = svm_metrics["cm"]
                ax_cm_svm.imshow(cm, interpolation="nearest", cmap="Blues")
                ax_cm_svm.set_title("SVM Confusion Matrix")
                ax_cm_svm.set_xlabel("Predicted")
                ax_cm_svm.set_ylabel("True")
                ax_cm_svm.set_xticks([0,1]); ax_cm_svm.set_xticklabels(["Non-ASD","ASD"])
                ax_cm_svm.set_yticks([0,1]); ax_cm_svm.set_yticklabels(["Non-ASD","ASD"])
                for (i, j), v in np.ndenumerate(cm):
                    ax_cm_svm.text(j, i, str(v), ha="center", va="center")
            else:
                ax_cm_svm.text(0.5, 0.5, "SVM confusion matrix unavailable", ha='center', va='center')
                ax_cm_svm.set_axis_off()

            # --- XGB Confusion Matrix ---
            if "error" not in xgb_metrics and "cm" in xgb_metrics:
                cm = xgb_metrics["cm"]
                ax_cm_xgb.imshow(cm, interpolation="nearest", cmap="Greens")
                ax_cm_xgb.set_title("XGBoost Confusion Matrix")
                ax_cm_xgb.set_xlabel("Predicted")
                ax_cm_xgb.set_ylabel("True")
                ax_cm_xgb.set_xticks([0,1]); ax_cm_xgb.set_xticklabels(["Non-ASD","ASD"])
                ax_cm_xgb.set_yticks([0,1]); ax_cm_xgb.set_yticklabels(["Non-ASD","ASD"])
                for (i, j), v in np.ndenumerate(cm):
                    ax_cm_xgb.text(j, i, str(v), ha="center", va="center")
            else:
                ax_cm_xgb.text(0.5, 0.5, "XGBoost confusion matrix unavailable", ha='center', va='center')
                ax_cm_xgb.set_axis_off()

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.models_plot_holder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(self.models_plot_holder, text="matplotlib not installed — plots unavailable", style="CardText.TLabel").pack()

        # ---- Metrics & predictions (user submission)
        output_lines = []
        output_lines.append("Model training complete. Summary metrics (on test split):\n")

        def append_metrics(name, m):
            if not m: return
            if "error" in m:
                output_lines.append(f"{name}: ERROR — {m['error']}\n")
                return
            output_lines.append(f"{name}:\n")
            output_lines.append(f"  Accuracy:  {m.get('accuracy', 0):.3f}\n")
            output_lines.append(f"  Precision: {m.get('precision', 0):.3f}\n")
            output_lines.append(f"  Recall:    {m.get('recall', 0):.3f}\n")
            output_lines.append(f"  F1 Score:  {m.get('f1', 0):.3f}\n")
            output_lines.append(f"  AUC:       {m.get('auc', 0):.3f}\n\n")

        def append_cm(name, m):
            if m and "cm" in m and "error" not in m:
                tn, fp, fn, tp = m["cm"].ravel()
                output_lines.append(f"{name} Confusion Matrix (rows=True, cols=Pred):\n")
                output_lines.append(f"  TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}\n\n")

        append_metrics("SVM", svm_metrics)
        append_metrics("XGBoost", xgb_metrics)
        append_cm("SVM", svm_metrics)
        append_cm("XGBoost", xgb_metrics)

        if self.latest_submission is None:
            output_lines.append("No recent submission found to predict. Return to the questionnaire and submit first.\n")
        else:
            # Build a row with EXACTLY the training columns
            user_row = {}
            for col in X.columns:
                if col.startswith("A") and col.endswith("_Score"):
                    idx = int(col[1:col.index("_")])  # A1_Score -> 1
                    user_row[col] = self.latest_submission["A_scores"][idx-1]
                elif col == "age_desc":
                    user_row[col] = self.latest_submission.get("age_desc", np.nan)
                elif col == "jaundice":
                    user_row[col] = self.latest_submission.get("jaundice", np.nan)
                else:
                    user_row[col] = np.nan

            user_df = pd.DataFrame([user_row], columns=X.columns)

            # Ensure numeric cols are numeric (NaN for missing), categorical are object/NaN
            for col in numeric_cols:
                user_df[col] = pd.to_numeric(user_df[col], errors="coerce")
            for col in cat_cols:
                user_df[col] = user_df[col].astype("object")

            if svm_pipeline is not None and "error" not in svm_metrics:
                try:
                    prob = svm_pipeline.predict_proba(user_df)[0, 1]
                    pred = int(prob >= 0.5)
                    label = "ASD" if pred == 1 else "Non-ASD"
                    output_lines.append(f"SVM Prediction for your submission: {label}  (prob={prob:.3f})\n")
                except Exception as e:
                    output_lines.append(f"SVM prediction failed: {e}\n")
            else:
                output_lines.append("SVM pipeline unavailable — cannot predict.\n")

            if xgb_pipeline is not None and "error" not in xgb_metrics:
                try:
                    prob = xgb_pipeline.predict_proba(user_df)[0, 1]
                    pred = int(prob >= 0.5)
                    label = "ASD" if pred == 1 else "Non-ASD"
                    output_lines.append(f"XGBoost Prediction for your submission: {label}  (prob={prob:.3f})\n")
                except Exception as e:
                    output_lines.append(f"XGBoost prediction failed: {e}\n")
            else:
                output_lines.append("XGBoost model unavailable — cannot predict.\n")

        self.models_text.configure(state="normal")
        self.models_text.delete("1.0", tk.END)
        self.models_text.insert(tk.END, "".join(output_lines))
        self.models_text.configure(state="disabled")

    # ---------- progress counter ----------
    def _update_progress(self):
        answered = sum(1 for i in range(1, 11) if self.responses[i].get() in ("DA","SA","SD","DD"))
        self.progress_var.set(f"Answered: {answered} / 10")

    # ---------- restart ----------
    def _restart(self):
        for i in range(1, 11):
            self.responses[i].set("")
        self.age_var.set("")
        self.country_var.set("")
        self.gender_var.set("Select…")
        self.ethnicity_var.set("Select…")
        self.jaundice_var.set("Select…")
        self.relation_var.set("Select…")
        self.latest_submission = None
        self._update_progress()
        self.show_page(self.page1)

    # ---------- nav ----------
    def show_page(self, page):
        self.current_page = page
        self.current_page.tkraise()

# ------------------------ MAIN ------------------------
if __name__ == "__main__":
    merge_new_data_once()

    root = tk.Tk()
    app = AQ10App(root)
    root.mainloop()
