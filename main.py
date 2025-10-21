import tkinter as tk
from tkinter import ttk, messagebox
import random
import platform
import os
from pathlib import Path
from datetime import datetime

# =========================
# CONFIG — set your new CSV here
# =========================
NEW_DATA_PATH = "new_data.csv"   # <- change this to your extra CSV file name
DATASET_BASENAME = "train.csv"   # your master dataset
BACKUP_BASENAME  = "train.backup.csv"

# Optional ML for on-screen % (NOT required for saving to CSV)
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ------------------------ PATHS / DATASET ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, DATASET_BASENAME)
NEW_DATA_ABS = os.path.join(BASE_DIR, NEW_DATA_PATH)
BACKUP_PATH  = os.path.join(BASE_DIR, BACKUP_BASENAME)

# EXACT headings (order preserved when creating/aligning)
REQUIRED_HEADERS = [
    "ID",
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "age","gender","ethnicity","jaundice","austim","contry_of_res",
    "used_app_before","result","age_desc","relation","Class/ASD"
]

# Extra columns we add while keeping REQUIRED_HEADERS first
EXTRA_HEADERS = [
    "screening_result"  # 1 if AQ-10 >= 6 else 0 (screening outcome, not diagnosis)
]

# ------------------------ ONE-TIME MERGE ------------------------
def merge_new_data_once():
    """
    If NEW_DATA_PATH exists, merge it into train.csv safely:
      - backup train.csv
      - align columns
      - continue IDs correctly
      - save back to train.csv
      - rename NEW_DATA_PATH to *.merged_TIMESTAMP.csv (prevents double-merge)
    """
    master_file = Path(DATASET_PATH)
    new_file = Path(NEW_DATA_ABS)

    if not new_file.exists():
        # Nothing to merge; just proceed to app
        return

    # Ensure master exists (create empty with headers if needed)
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
        # Minimal fallback: ensure the master file exists with headers
        if not master_file.exists():
            with open(master_file, "w", encoding="utf-8") as f:
                f.write(",".join(REQUIRED_HEADERS + EXTRA_HEADERS) + "\n")

    if not SKLEARN_AVAILABLE:
        # Without pandas, we won’t perform an automatic merge to avoid corrupting CSV.
        # Tell user and exit gracefully.
        messagebox.showwarning(
            "Merge skipped",
            "pandas is not installed, so the automatic merge is skipped.\n\n"
            "Install with: pip install pandas\n"
            "Then run again to merge your new CSV."
        )
        return

    # With pandas: proceed with safe merge
    df_master = pd.read_csv(master_file)
    df_new = pd.read_csv(new_file)

    # Backup master
    Path(BACKUP_PATH).write_bytes(master_file.read_bytes())

    # Harmonize columns (keep all; master columns first)
    master_cols = list(df_master.columns)
    new_cols = [c for c in df_new.columns if c not in master_cols]
    all_cols = master_cols + new_cols

    # Add missing columns
    for c in all_cols:
        if c not in df_master.columns:
            df_master[c] = ""
        if c not in df_new.columns:
            df_new[c] = ""

    # Reorder
    df_master = df_master[all_cols]
    df_new = df_new[all_cols]

    # Continue IDs
    if "ID" not in all_cols:
        df_master.insert(0, "ID", range(1, len(df_master) + 1))
        all_cols = ["ID"] + [c for c in all_cols if c != "ID"]
        df_master = df_master[all_cols]

    max_id = pd.to_numeric(df_master["ID"], errors="coerce").max()
    if pd.isna(max_id):
        max_id = len(df_master)
    start_id = int(max_id) + 1
    df_new["ID"] = range(start_id, start_id + len(df_new))

    # Mild normalization for known columns if present
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

    # Concatenate and save
    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    df_combined.to_csv(master_file, index=False)

    # Rename the new file so it won't merge again next run
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    merged_name = new_file.with_suffix(f".merged_{ts}.csv")
    new_file.rename(merged_name)

    # Optional: UI pop to confirm success (if desktop)
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

FONT_TITLE    = ("SF Pro Display", 28, "bold")
FONT_SUBTITLE = ("SF Pro Text", 16)
FONT_TEXT     = ("SF Pro Text", 14)
FONT_QUESTION = ("SF Pro Display", 16, "bold")

PUZZLE_COLORS = ["#00b4d8", "#90e0ef", "#0077b6", "#48cae4"]

# ------------------------ APP ------------------------
class AQ10App:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 AQ-10 Autism Test")
        self.root.configure(bg=WINDOW_BG)
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass

        # Background canvas with static puzzle pieces
        self.bg_canvas = tk.Canvas(root, bg=WINDOW_BG, highlightthickness=0, bd=0)
        self.bg_canvas.place(relwidth=1, relheight=1)
        self._draw_static_puzzle_background()

        # Pages
        self.page1 = tk.Frame(root, bg=FRAME_BG, highlightthickness=0, bd=0)
        self.page2 = tk.Frame(root, bg=FRAME_BG, highlightthickness=0, bd=0)
        self._place_pages()
        self.current_page = self.page1

        # Data/model holders
        self.df = None
        self.model = None
        self.model_ready = False
        self.model_note  = ""

        # Load dataset (and prep quick ML if possible)
        self._load_dataset_and_model()

        # AQ-10 questions
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
        # Reverse-scored items (score 1 when Disagree/0)
        self.disagree_score_items = {2,3,4,5,6,9}
        self.responses = {}

        # UI
        self._init_styles()
        self._create_page1()   # ALWAYS starts on Page 1
        self._create_page2()
        self.show_page(self.page1)

        self.root.bind("<Configure>", self._on_resize)

    # ---------- styles ----------
    def _init_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=FRAME_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Header.TLabel", background=FRAME_BG, foreground=TEXT_PRIMARY, font=FONT_TITLE)
        style.configure("Subheader.TLabel", background=FRAME_BG, foreground=TEXT_SECONDARY, font=FONT_SUBTITLE)
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=TEXT_PRIMARY, font=FONT_QUESTION)
        style.configure("CardText.TLabel", background=CARD_BG, foreground=TEXT_SECONDARY, font=FONT_TEXT)
        style.configure("Primary.TButton", font=FONT_TEXT, padding=10)
        style.map("Primary.TButton", background=[("!disabled", ACCENT)], foreground=[("!disabled", "#ffffff")])
        style.configure("Secondary.TButton", font=FONT_TEXT, padding=10, background="#2a2a2a", foreground="#ffffff")
        style.map("Secondary.TButton", background=[("!disabled", "#2a2a2a")], foreground=[("!disabled", "#ffffff")])
        style.configure("AQ.TRadiobutton", background=CARD_BG, foreground=TEXT_PRIMARY, font=FONT_TEXT)
        style.configure("Entry.TEntry", fieldbackground="#1e1e1e", background="#1e1e1e", foreground="#ffffff")
        style.configure("TMenubutton", background="#1e1e1e", foreground="#ffffff")

    # ---------- background ----------
    def _draw_static_puzzle_background(self):
        self.bg_canvas.delete("all")
        w = max(800, self.root.winfo_width() or self.root.winfo_screenwidth())
        h = max(600, self.root.winfo_height() or self.root.winfo_screenheight())
        rng = random.Random(42)
        for i in range(26):
            size = rng.randint(28, 56)
            x = rng.randint(40, w - 40)
            y = rng.randint(40, h - 40)
            color = rng.choice(PUZZLE_COLORS)
            self._draw_piece(x, y, size, color, f"p_{i}")

    def _draw_piece(self, x, y, size, color, tag):
        r = max(6, int(size * 0.18))
        left, top = x - size // 2, y - size // 2
        right, bottom = x + size // 2, y + size // 2
        c = self.bg_canvas
        # rounded rect
        c.create_oval(left, top, left + 2*r, top + 2*r, fill=color, outline=color, tags=tag)
        c.create_oval(right - 2*r, top, right, top + 2*r, fill=color, outline=color, tags=tag)
        c.create_oval(left, bottom - 2*r, left + 2*r, bottom, fill=color, outline=color, tags=tag)
        c.create_oval(right - 2*r, bottom - 2*r, right, bottom, fill=color, outline=color, tags=tag)
        c.create_rectangle(left + r, top, right - r, bottom, fill=color, outline=color, tags=tag)
        c.create_rectangle(left, top + r, right, bottom - r, fill=color, outline=color, tags=tag)
        # simple highlights
        c.create_line(left + r, top + r, right - r, top + r, fill="#cfefff", width=1, tags=tag)
        c.create_line(left + r, top + r, left + r, bottom - r, fill="#b8e0ff", width=1, tags=tag)

    # ---------- layout ----------
    def _place_pages(self):
        w = self.root.winfo_width() or self.root.winfo_screenwidth()
        max_width = min(1000, int(w * 0.88))
        self.page1.place(relx=0.5, rely=0, anchor="n", y=24, width=max_width)
        self.page2.place(relx=0.5, rely=0, anchor="n", y=24, width=max_width)

    def _on_resize(self, _):
        self._draw_static_puzzle_background()
        self._place_pages()
        self.current_page.tkraise()  # keep current page on top

    # ---------- page 1 (personal info) ----------
    def _create_page1(self):
        container = ttk.Frame(self.page1, padding=30, style="TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="🧠 Autism Spectrum Quotient (AQ-10)", style="Header.TLabel").pack(anchor="w", pady=(0,6))
        ttk.Label(container, text="First, some basic information.", style="Subheader.TLabel").pack(anchor="w", pady=(0,20))

        card = ttk.Frame(container, padding=20, style="Card.TFrame")
        card.pack(fill="x", expand=False)

        # Vars
        self.gender_var    = tk.StringVar(value="Select…")
        self.ethnicity_var = tk.StringVar(value="Select…")
        self.age_var       = tk.StringVar(value="")
        self.country_var   = tk.StringVar(value="")
        self.jaundice_var  = tk.StringVar(value="Select…")
        self.relation_var  = tk.StringVar(value="Select…")

        for i in range(4):
            card.columnconfigure(i, weight=1)

        # Gender
        ttk.Label(card, text="Gender", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=(4,2))
        gender_menu = ttk.OptionMenu(card, self.gender_var, self.gender_var.get(), "Male", "Female")
        gender_menu.grid(row=1, column=0, sticky="we", padx=(0,20), pady=(0,10))

        # Ethnicity
        ttk.Label(card, text="Ethnicity", style="CardText.TLabel").grid(row=0, column=1, sticky="w", pady=(4,2))
        ethnicity_menu = ttk.OptionMenu(
            card, self.ethnicity_var, self.ethnicity_var.get(),
            "White-European","Black","Asian","Latino","Middle Eastern",
            "South Asian","Hispanic","Pasifika","Other"
        )
        ethnicity_menu.grid(row=1, column=1, sticky="we", padx=(0,20), pady=(0,10))

        # Age (numeric)
        ttk.Label(card, text="Age (years)", style="CardText.TLabel").grid(row=0, column=2, sticky="w", pady=(4,2))
        age_entry = ttk.Entry(card, textvariable=self.age_var, style="Entry.TEntry")
        age_entry.grid(row=1, column=2, sticky="we", padx=(0,20), pady=(0,10))

        # Country
        ttk.Label(card, text="Country of residence", style="CardText.TLabel").grid(row=2, column=0, sticky="w", pady=(4,2))
        ttk.Entry(card, textvariable=self.country_var, style="Entry.TEntry").grid(row=3, column=0, sticky="we", padx=(0,20), pady=(0,10))

        # Jaundice
        ttk.Label(card, text="Jaundice (as an infant)", style="CardText.TLabel").grid(row=2, column=1, sticky="w", pady=(4,2))
        jaundice_menu = ttk.OptionMenu(card, self.jaundice_var, self.jaundice_var.get(), "Yes","No")
        jaundice_menu.grid(row=3, column=1, sticky="we", padx=(0,20), pady=(0,10))

        # Relation
        ttk.Label(card, text="Relation", style="CardText.TLabel").grid(row=2, column=2, sticky="w", pady=(4,2))
        relation_menu = ttk.OptionMenu(card, self.relation_var, self.relation_var.get(), "Self","Parent","Guardian")
        relation_menu.grid(row=3, column=2, sticky="we", padx=(0,20), pady=(0,10))

        # Actions
        actions = ttk.Frame(container, style="TFrame")
        actions.pack(fill="x", pady=20)
        ttk.Button(actions, text="Continue →", style="Primary.TButton",
                   command=self._validate_page1_and_continue).pack(side="right")

    # ---------- page 2 (questions + scroll) ----------
    def _create_page2(self):
        outer = ttk.Frame(self.page2, padding=0, style="TFrame")
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, padding=30, style="TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Answer Each Question", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="Choose what fits you best. (1 = Agree, 0 = Disagree)", style="Subheader.TLabel").pack(anchor="w", pady=(6,0))
        self.progress_var = tk.StringVar(value="Answered: 0 / 10")
        ttk.Label(header, textvariable=self.progress_var, style="Subheader.TLabel").pack(anchor="w", pady=(8,0))

        # scrollable area
        sc = ttk.Frame(outer, style="TFrame")
        sc.pack(fill="both", expand=True, padx=30, pady=(0,20))
        self.q_canvas = tk.Canvas(sc, bg=FRAME_BG, highlightthickness=0, bd=0)
        self.q_canvas.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(sc, orient="vertical", command=self.q_canvas.yview)
        sb.pack(side="right", fill="y")
        self.q_canvas.configure(yscrollcommand=sb.set)
        self.scroll_frame = ttk.Frame(self.q_canvas, style="TFrame")
        self.scroll_window = self.q_canvas.create_window((0,0), window=self.scroll_frame, anchor="nw")
        self.q_canvas.bind("<Configure>", lambda e: self.q_canvas.itemconfigure(self.scroll_window, width=e.width))
        self.scroll_frame.bind("<Configure>", lambda e: self.q_canvas.configure(scrollregion=self.q_canvas.bbox("all")))
        self._bind_mousewheel(self.q_canvas)

        for i, q in enumerate(self.questions, start=1):
            card = ttk.Frame(self.scroll_frame, padding=16, style="Card.TFrame")
            card.pack(fill="x", expand=True, pady=(0,12))
            ttk.Label(card, text=f"{i}. {q}", style="CardTitle.TLabel", wraplength=1200, justify="left").pack(anchor="w", pady=(0,8))
            var = tk.StringVar(value="")
            self.responses[i] = var
            row = ttk.Frame(card, style="Card.TFrame")
            row.pack(anchor="w", pady=(4,0))
            ttk.Radiobutton(row, text="Agree  (1)", value="1", variable=var, style="AQ.TRadiobutton",
                            command=self._update_progress).pack(side="left", padx=(0,16))
            ttk.Radiobutton(row, text="Disagree  (0)", value="0", variable=var, style="AQ.TRadiobutton",
                            command=self._update_progress).pack(side="left")

        footer = ttk.Frame(outer, padding=20, style="TFrame")
        footer.pack(fill="x")
        ttk.Button(footer, text="← Back", style="Secondary.TButton",
                   command=lambda: self.show_page(self.page1)).pack(side="left")
        ttk.Button(footer, text="Submit Test", style="Primary.TButton",
                   command=self._submit_and_exit).pack(side="right")

    def _bind_mousewheel(self, widget):
        sysname = platform.system()
        widget.bind("<Enter>", lambda e: self._wheel_bind(sysname))
        widget.bind("<Leave>", lambda e: self._wheel_unbind(sysname))

    def _wheel_bind(self, sysname):
        if sysname == "Linux":
            self.q_canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.q_canvas.bind_all("<Button-5>", self._on_mousewheel)
        else:
            self.q_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _wheel_unbind(self, sysname):
        if sysname == "Linux":
            self.q_canvas.unbind_all("<Button-4>")
            self.q_canvas.unbind_all("<Button-5>")
        else:
            self.q_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self.q_canvas.yview_scroll(-2, "units")
        elif getattr(event, "num", None) == 5:
            self.q_canvas.yview_scroll(2, "units")
        else:
            delta = -1 if event.delta > 0 else 1
            self.q_canvas.yview_scroll(delta * 3, "units")

    def _update_progress(self):
        answered = sum(1 for i in range(1, 11) if self.responses[i].get() != "")
        self.progress_var.set(f"Answered: {answered} / 10")

    def _validate_page1_and_continue(self):
        errors = []
        country = self.country_var.get().strip()
        age_raw = self.age_var.get().strip()
        if country == "":
            errors.append("Please enter your country of residence.")
        if age_raw == "" or not age_raw.isdigit() or int(age_raw) < 0:
            errors.append("Please enter a valid non-negative whole number for age.")
        if self.gender_var.get() not in ("Male", "Female"):
            errors.append("Please select a gender (Male or Female).")
        if self.ethnicity_var.get() not in (
            "White-European","Black","Asian","Latino","Middle Eastern","South Asian","Hispanic","Pasifika","Other"):
            errors.append("Please select an ethnicity.")
        if self.jaundice_var.get() not in ("Yes", "No"):
            errors.append("Please select jaundice Yes/No.")
        if self.relation_var.get() not in ("Self", "Parent", "Guardian"):
            errors.append("Please select a relation.")

        if errors:
            messagebox.showwarning("Missing information", "\n".join(errors))
            return
        self.show_page(self.page2)

    # ---------- dataset & ML ----------
    def _load_dataset_and_model(self):
        # ensure CSV exists
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

        # quick ML (only for on-screen %)
        self.model_ready = False
        self.model = None
        self.model_note = ""
        if not SKLEARN_AVAILABLE or self.df is None:
            self.model_note = "Install pandas & scikit-learn for ML (pip install pandas scikit-learn)."
            return
        if "Class/ASD" not in self.df.columns:
            self.model_note = "No 'Class/ASD' labels; using rule-based estimate."
            return

        train_df = self.df.copy()
        y = pd.to_numeric(train_df.get("Class/ASD"), errors="coerce")
        mask = y.isin([0,1])
        train_df = train_df[mask]
        y = y[mask].astype(int)

        if train_df.empty or y.nunique() < 2:
            self.model_note = "Not enough labelled rows; using rule-based estimate."
            return

        needed = [f"A{i}_Score" for i in range(1, 11)] + ["age_desc", "jaundice"]
        if any(col not in train_df.columns for col in needed):
            self.model_note = "Missing features for ML; using rule-based estimate."
            return

        X = self._build_features(train_df)

        if len(y) < 20:
            self.model_note = f"Too few labelled rows ({len(y)}); using rule-based estimate."
            return

        try:
            base = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced", random_state=42)
            model = CalibratedClassifierCV(base, method="isotonic", cv=3)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y_test, proba)
            f1 = f1_score(y_test, pred)

            self.model = model
            self.model_ready = True
            self.model_note = f"Model trained on {len(y_train)} rows (AUC={auc:.3f}, F1={f1:.3f})."
        except Exception as e:
            self.model_note = f"Model error: {e}; using rule-based estimate."

    def _build_features(self, df_like):
        A_cols = [f"A{i}_Score" for i in range(1, 11)]
        X_A = df_like[A_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        age_norm = df_like["age_desc"].astype(str).str.lower().str.strip()
        age18plus = age_norm.str.contains("18") | age_norm.str.contains("more") | age_norm.str.contains("18+")
        jaundice_bin = df_like["jaundice"].astype(str).str.lower().map({"yes": 1.0, "no": 0.0}).fillna(0.0)
        X = pd.concat([
            X_A,
            age18plus.rename("age18plus").astype(float),
            jaundice_bin.rename("jaundice")
        ], axis=1)
        return X.values

    def _predict_percentage(self, A_scores_0_1, age_desc, jaundice_str):
        aq10_total = sum(A_scores_0_1)
        if not (self.model_ready and SKLEARN_AVAILABLE):
            return float(aq10_total * 10.0), f"Rule-based screening % (not ML). {self.model_note}"
        try:
            age_str = str(age_desc).lower().strip()
            age18 = 1.0 if ("18" in age_str or "more" in age_str or "+" in age_str) else 0.0
            j = 1.0 if str(jaundice_str).strip().lower() == "yes" else 0.0
            x = np.array(A_scores_0_1 + [age18, j], dtype=float).reshape(1, -1)
            proba = self.model.predict_proba(x)[0, 1]
            return float(proba * 100.0), f"Model estimate. {self.model_note}"
        except Exception as e:
            return float(aq10_total * 10.0), f"Fallback (rule-based). Model error: {e}"

    # ---------- saving ----------
    def _append_to_dataset(self, row_dict):
        """
        Append a row to THE SAME CSV, ensuring exact columns and continuing ID.
        """
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

        # With pandas
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

        ordered_cols = [c for c in REQUIRED_HEADERS] + \
                       [c for c in EXTRA_HEADERS] + \
                       [c for c in df_now.columns if c not in REQUIRED_HEADERS + EXTRA_HEADERS]
        df_now = df_now[ordered_cols]

        df_now.to_csv(DATASET_PATH, index=False)
        self.df = df_now

        return assigned_id, n_old + 1

    # ---------- submit ----------
    def _submit_and_exit(self):
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
                messagebox.showwarning("Incomplete", "Please answer all questions on Page 2 before submitting.")
                return

        raw = [int(self.responses[i].get()) for i in range(1, 11)]
        A_scores = []
        for i, val in enumerate(raw, start=1):
            A_scores.append((1 - val) if i in self.disagree_score_items else val)
        aq10_total = sum(A_scores)

        percent, about = self._predict_percentage(
            A_scores_0_1=A_scores,
            age_desc=age_desc,
            jaundice_str=self.jaundice_var.get()
        )

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

        advisory = (
            "This is a screening tool, not a diagnosis. "
            "If you have concerns, please consult a qualified clinician."
        )
        details_line = f"\nSaved to: {DATASET_BASENAME}\nRow: {saved_row_num}   ID: {assigned_id}"

        if screening_class == 1:
            messagebox.showwarning(
                "Result (Consider Assessment)",
                f"AQ-10 Score: {aq10_total}/10\n"
                f"Estimated likelihood: {percent:.1f}%\n\n"
                f"{about}\n\n"
                f"Recommendation: A specialist assessment may be appropriate.\n\n"
                f"{advisory}{details_line}"
            )
        else:
            messagebox.showinfo(
                "Result",
                f"AQ-10 Score: {aq10_total}/10\n"
                f"Estimated likelihood: {percent:.1f}%\n\n"
                f"{about}\n\n"
                f"{advisory}{details_line}"
            )

        self.root.after(50, self.root.destroy)

    # ---------- nav ----------
    def show_page(self, page):
        self.current_page = page
        self.current_page.tkraise()

# ------------------------ MAIN (merge once, then run app) ------------------------
if __name__ == "__main__":
    # 1) Merge once if NEW_DATA_PATH exists
    merge_new_data_once()

    # 2) Launch the app
    root = tk.Tk()
    app = AQ10App(root)
    root.mainloop()