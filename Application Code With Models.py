# integrated_aq10_app.py
import tkinter as tk
from tkinter import ttk, messagebox
import random
import platform
import os
from pathlib import Path
from datetime import datetime

# ML imports
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, classification_report
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
# CONFIG
# ------------------------
NEW_DATA_PATH = "new_data.csv"
DATASET_BASENAME = "train.csv"
TRAIN_CLEANED = "train_cleaned.csv"   # used for training SVM/XGBoost
BACKUP_BASENAME  = "train.backup.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# ------------------------ ONE-TIME MERGE (keeps your original function) ------------------------
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
        self.disagree_score_items = {2,3,4,5,6,9}
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
        for i in range(22):
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

        for i, q in enumerate(self.questions, start=1):
            card_q = ttk.Frame(self.scroll_frame, padding=10, style="Card.TFrame")
            card_q.pack(fill="x", expand=True, pady=(6,6))
            ttk.Label(card_q, text=f"{i}. {q}", style="CardTitle.TLabel", wraplength=900, justify="left").pack(anchor="w")
            var = tk.StringVar(value="")
            self.responses[i] = var
            row = ttk.Frame(card_q, style="Card.TFrame")
            row.pack(anchor="w", pady=(6,0))
            ttk.Radiobutton(row, text="Agree  (1)", value="1", variable=var, style="AQ.TRadiobutton").pack(side="left", padx=(0,12))
            ttk.Radiobutton(row, text="Disagree  (0)", value="0", variable=var, style="AQ.TRadiobutton").pack(side="left")

        footer = ttk.Frame(container, padding=8, style="TFrame")
        footer.pack(fill="x")
        ttk.Button(footer, text="Submit →", style="Primary.TButton", command=self._submit_and_show_results).pack(side="right", padx=(6,0))
        ttk.Button(footer, text="Quit", style="Secondary.TButton", command=self.root.destroy).pack(side="left")

        self.progress_var = tk.StringVar(value="Answered: 0 / 10")
        ttk.Label(container, textvariable=self.progress_var, style="Subheader.TLabel").pack(anchor="w", pady=(6,0))

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
        self.models_text = tk.Text(self.models_card, height=10, bg="#171717", fg="#cfcfcf", wrap="word")
        self.models_text.pack(fill="x", padx=6, pady=(6,6))
        self.models_text.configure(state="disabled")

        actions = ttk.Frame(container, style="TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="Restart → Back to start", style="Secondary.TButton", command=self._restart).pack(side="left")
        ttk.Button(actions, text="Close", style="Primary.TButton", command=self.root.destroy).pack(side="right")

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

        for i in range(1, 11):
            if self.responses[i].get() == "":
                messagebox.showwarning("Incomplete", "Please answer all questions on Page 1 before submitting.")
                return

        raw = [int(self.responses[i].get()) for i in range(1, 11)]
        A_scores = []
        for i, val in enumerate(raw, start=1):
            A_scores.append((1 - val) if i in self.disagree_score_items else val)
        aq10_total = sum(A_scores)

        screening_class = 1 if aq10_total >= 6 else 0

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
        percent = aq10_total * 10.0  # fallback percent if no model used here
        about = "Rule-based estimate (not ML)."
        status_text = "Consider specialist assessment." if screening_class == 1 else "Low immediate concern."
        details = (
            f"AQ-10 Score: {aq10_total}/10\n"
            f"Estimated likelihood (rule): {percent:.1f}%\n"
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
        # Models train and show when page 3 opens
        self.show_page(self.page3)
        self.root.after(100, self._train_and_display_models)  # small delay to allow UI to update

    # ---------- TRAIN & DISPLAY MODELS ----------
    def _train_and_display_models(self):
        # Clear old plot/text
        for w in self.models_plot_holder.winfo_children():
            w.destroy()
        self.models_text.configure(state="normal")
        self.models_text.delete("1.0", tk.END)
        self.models_text.configure(state="disabled")

        if not SKLEARN_AVAILABLE:
            messagebox.showwarning("ML not available", "pandas/scikit-learn not installed. Models cannot be trained.")
            return

        if not os.path.exists(TRAIN_CLEANED_PATH):
            messagebox.showwarning("Training data missing", f"Training file '{TRAIN_CLEANED}' not found. Place it next to the script.")
            return

        try:
            train_df = pd.read_csv(TRAIN_CLEANED_PATH)
        except Exception as e:
            messagebox.showwarning("Read error", f"Could not read training CSV: {e}")
            return

        target_col = "Class/ASD"
        if target_col not in train_df.columns:
            messagebox.showwarning("Target missing", f"Target column '{target_col}' not found in training CSV.")
            return

        # Convert target to binary (robust)
        y = train_df[target_col].copy()
        def to_bin(val):
            try:
                if pd.isna(val): return 0
                if isinstance(val, (int, float)) and int(val) in (0,1): return int(val)
                s = str(val).strip().lower()
                if s in ("yes","1","true","asd","1.0","y"):
                    return 1
                if s in ("no","0","false","non-asd","non_asd","n"):
                    return 0
                # fallback: try numeric
                try:
                    if float(s) >= 0.5:
                        return 1
                    else:
                        return 0
                except Exception:
                    return 0
            except Exception:
                return 0
        y = y.map(to_bin).astype(int)

        # Keep only rows with valid target if possible
        mask = y.isin([0,1])
        Xdf = train_df.loc[mask].copy()
        y = y[mask].values
        if Xdf.empty or len(y) < 10:
            messagebox.showwarning("Not enough data", "Not enough labeled rows to train models. Need at least ~10 labeled rows.")
            return

        # Build feature set: try detect A1_Score..A10_Score and additional small features
        feature_cols = [f"A{i}_Score" for i in range(1, 11)]
        extra_cols = []
        for col in ("age_desc","jaundice","age","gender","ethnicity"):
            if col in Xdf.columns:
                extra_cols.append(col)
        used_cols = [c for c in feature_cols if c in Xdf.columns] + extra_cols
        if len([c for c in feature_cols if c in Xdf.columns]) < 10:
            # If 10 A* not present, try using all numeric except target
            numeric_cols = Xdf.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != target_col]
            used_cols = numeric_cols if numeric_cols else used_cols

        X = Xdf[used_cols].copy()

        # Preprocess - numeric pipeline + cat pipeline
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]) if cat_cols else None

        if cat_cols:
            preprocessor = ColumnTransformer([("num", num_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)])
        else:
            preprocessor = ColumnTransformer([("num", num_pipeline, numeric_cols)])

        # Train-test split for metric estimation
        try:
            X_proc = preprocessor.fit_transform(X)
        except Exception as e:
            messagebox.showwarning("Preprocess error", f"Could not preprocess training data: {e}")
            return

        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, stratify=y, random_state=42)

        # ---- SVM ----
        svm_model = None
        svm_metrics = {}
        try:
            svm_model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
            svm_model.fit(X_train, y_train)
            svm_probs = svm_model.predict_proba(X_test)[:, 1]
            svm_pred = (svm_probs >= 0.5).astype(int)
            svm_metrics = {
                "accuracy": accuracy_score(y_test, svm_pred),
                "precision": precision_score(y_test, svm_pred, zero_division=0),
                "recall": recall_score(y_test, svm_pred, zero_division=0),
                "f1": f1_score(y_test, svm_pred, zero_division=0),
                "auc": roc_auc_score(y_test, svm_probs),
                "fpr_tpr": roc_curve(y_test, svm_probs)
            }
        except Exception as e:
            svm_model = None
            svm_metrics = {"error": str(e)}

        # ---- XGBoost ----
        xgb_model = None
        xgb_metrics = {}
        if XGBOOST_AVAILABLE:
            try:
                # Build full pipeline: preprocessor + xgb
                xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                    random_state=42, learning_rate=0.05, max_depth=5,
                                    n_estimators=200, use_label_encoder=False)
                xgb_pipeline = Pipeline([("pre", preprocessor), ("clf", xgb)])
                xgb_pipeline.fit(X, y)  # fit on the whole X (we will show metrics on split)
                # For test metrics, transform X_test (we already have X_test from preprocessor)
                xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
                xgb_pred = (xgb_probs >= 0.5).astype(int)
                xgb_metrics = {
                    "accuracy": accuracy_score(y_test, xgb_pred),
                    "precision": precision_score(y_test, xgb_pred, zero_division=0),
                    "recall": recall_score(y_test, xgb_pred, zero_division=0),
                    "f1": f1_score(y_test, xgb_pred, zero_division=0),
                    "auc": roc_auc_score(y_test, xgb_probs),
                    "fpr_tpr": roc_curve(y_test, xgb_probs),
                    "pipeline": xgb_pipeline
                }
                xgb_model = xgb_pipeline
            except Exception as e:
                xgb_model = None
                xgb_metrics = {"error": str(e)}
        else:
            xgb_model = None
            xgb_metrics = {"error": "xgboost not installed."}

        # For SVM we need full pipeline with preprocessor to predict user data later.
        svm_pipeline = None
        try:
            # Wrap preprocessor and svm into pipeline
            from sklearn.base import clone
            svm_clone = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
            svm_pipeline = Pipeline([("pre", preprocessor), ("clf", svm_clone)])
            svm_pipeline.fit(X, y)
        except Exception as e:
            svm_pipeline = None

        # Display plots (embedded)
        if MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(9, 4), dpi=100)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            # SVM ROC
            if "fpr_tpr" in svm_metrics and isinstance(svm_metrics["fpr_tpr"], tuple):
                fpr, tpr, _ = svm_metrics["fpr_tpr"]
                roc_auc = svm_metrics.get("auc", 0.0)
                ax1.plot(fpr, tpr, label=f"SVM (AUC={roc_auc:.2f})")
                ax1.plot([0,1],[0,1], linestyle="--", color="gray")
                ax1.set_title("SVM ROC")
                ax1.set_xlabel("False Positive Rate")
                ax1.set_ylabel("True Positive Rate")
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, "SVM ROC unavailable", horizontalalignment='center', verticalalignment='center')

            # XGB ROC
            if "fpr_tpr" in xgb_metrics and isinstance(xgb_metrics["fpr_tpr"], tuple):
                fpr, tpr, _ = xgb_metrics["fpr_tpr"]
                roc_auc = xgb_metrics.get("auc", 0.0)
                ax2.plot(fpr, tpr, label=f"XGBoost (AUC={roc_auc:.2f})")
                ax2.plot([0,1],[0,1], linestyle="--", color="gray")
                ax2.set_title("XGBoost ROC")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "XGBoost ROC unavailable", horizontalalignment='center', verticalalignment='center')

            canvas = FigureCanvasTkAgg(fig, master=self.models_plot_holder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(self.models_plot_holder, text="matplotlib not installed — plots unavailable", style="CardText.TLabel").pack()

        # Prepare metrics and predict user's submission if available
        output_lines = []
        output_lines.append("Model training complete. Summary metrics (on test split):\n")
        def append_metrics(name, metrics):
            if not metrics:
                return
            if "error" in metrics:
                output_lines.append(f"{name}: ERROR — {metrics['error']}\n")
                return
            output_lines.append(f"{name}:\n")
            output_lines.append(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}\n")
            output_lines.append(f"  Precision: {metrics.get('precision', 0):.3f}\n")
            output_lines.append(f"  Recall:    {metrics.get('recall', 0):.3f}\n")
            output_lines.append(f"  F1 Score:  {metrics.get('f1', 0):.3f}\n")
            output_lines.append(f"  AUC:       {metrics.get('auc', 0):.3f}\n\n")

        append_metrics("SVM", svm_metrics)
        append_metrics("XGBoost", xgb_metrics)

        # Predict user's submission
        if self.latest_submission is None:
            output_lines.append("No recent submission found to predict. Return to the questionnaire and submit first.\n")
        else:
            # Build a DataFrame row consistent with training features (used_cols)
            # We will attempt to create same columns used in X (above).
            user_row = {}
            # first A1-A10 if present in training used_cols
            for i in range(1, 11):
                col = f"A{i}_Score"
                if col in used_cols:
                    user_row[col] = self.latest_submission["A_scores"][i-1]
            # extra columns: age_desc,jaundice etc.
            for c in extra_cols:
                if c == "age_desc":
                    user_row[c] = self.latest_submission.get("age_desc", "")
                elif c == "jaundice":
                    user_row[c] = self.latest_submission.get("jaundice", "")
                else:
                    user_row[c] = ""  # blank fallback

            user_df = pd.DataFrame([user_row], columns=used_cols)

            # use svm_pipeline and xgb_model to predict
            if svm_pipeline is not None:
                try:
                    prob = svm_pipeline.predict_proba(user_df)[0,1]
                    pred = int(prob >= 0.5)
                    label = "ASD" if pred==1 else "Non-ASD"
                    output_lines.append(f"SVM Prediction for your submission: {label}  (prob={prob:.3f})\n")
                except Exception as e:
                    output_lines.append(f"SVM prediction failed: {e}\n")
            else:
                output_lines.append("SVM pipeline unavailable — cannot predict.\n")

            if xgb_model is not None:
                try:
                    prob = xgb_model.predict_proba(user_df)[0,1]
                    pred = int(prob >= 0.5)
                    label = "ASD" if pred==1 else "Non-ASD"
                    output_lines.append(f"XGBoost Prediction for your submission: {label}  (prob={prob:.3f})\n")
                except Exception as e:
                    output_lines.append(f"XGBoost prediction failed: {e}\n")
            else:
                output_lines.append("XGBoost model unavailable — cannot predict.\n")

        # Display metrics and predictions
        self.models_text.configure(state="normal")
        self.models_text.delete("1.0", tk.END)
        self.models_text.insert(tk.END, "".join(output_lines))
        self.models_text.configure(state="disabled")

    # ---------- restart ----------
    def _restart(self):
        # clear responses & go back to page1
        for i in range(1, 11):
            self.responses[i].set("")
        self.age_var.set("")
        self.country_var.set("")
        self.gender_var.set("Select…")
        self.ethnicity_var.set("Select…")
        self.jaundice_var.set("Select…")
        self.relation_var.set("Select…")
        self.latest_submission = None
        self.show_page(self.page1)

    # ---------- nav ----------
    def show_page(self, page):
        self.current_page = page
        self.current_page.tkraise()

# ------------------------ MAIN ------------------------
if __name__ == "__main__":
    # Merge optional new data if present
    merge_new_data_once()

    root = tk.Tk()
    app = AQ10App(root)
    root.mainloop()
