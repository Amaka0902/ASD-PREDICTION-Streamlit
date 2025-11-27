import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ------------------------
# ML IMPORTS
# ------------------------
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        roc_curve, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ------------------------
# CONFIGURATION
# ------------------------
st.set_page_config(
    page_title="SpectrumLight Risk Tool",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# PATHS & DATA HANDLING
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "train.csv")

REQUIRED_HEADERS = [
    "ID", "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "ethnicity", "jaundice", "contry_of_res",
    "relation", "Class/ASD", "screening_result"
]


def init_dataset():
    """Create CSV if it doesn't exist."""
    if not os.path.exists(DATASET_PATH):
        df = pd.DataFrame(columns=REQUIRED_HEADERS)
        df.to_csv(DATASET_PATH, index=False)


def save_submission(data):
    """Save data to CSV and return the assigned ID."""
    try:
        df = pd.read_csv(DATASET_PATH)
    except:
        df = pd.DataFrame(columns=REQUIRED_HEADERS)

    # Generate ID
    if not df.empty and 'ID' in df.columns:
        # Handle cases where ID might be non-numeric briefly
        clean_ids = pd.to_numeric(df['ID'], errors='coerce').fillna(0)
        new_id = int(clean_ids.max()) + 1
    else:
        new_id = 1

    data['ID'] = new_id

    # Create new row dataframe
    new_row = pd.DataFrame([data])

    # Align columns
    for col in REQUIRED_HEADERS:
        if col not in new_row.columns:
            new_row[col] = ""

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATASET_PATH, index=False)
    return new_id


# ------------------------
# ML LOGIC (ROBUST VERSION)
# ------------------------
@st.cache_resource
def train_models(data_path):
    if not SKLEARN_AVAILABLE:
        return None, "Scikit-learn not installed."

    # --- LOAD DATA ---
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
        except:
            df = pd.DataFrame(columns=REQUIRED_HEADERS)
    else:
        df = pd.DataFrame(columns=REQUIRED_HEADERS)

    # --- THE FIX: INJECT DUMMY DATA FOR STABILITY ---
    # This ensures we always have at least one 'Yes' and one 'No' for the code to run
    # We create a fake High Risk and a fake Low Risk row just for training (not saved to CSV)

    # 1. Create Dummy High Risk (All Agree)
    dummy_high = {col: 1 for col in [f"A{i}_Score" for i in range(1, 11)]}
    dummy_high["Class/ASD"] = "Yes"

    # 2. Create Dummy Low Risk (All Disagree)
    dummy_low = {col: 0 for col in [f"A{i}_Score" for i in range(1, 11)]}
    dummy_low["Class/ASD"] = "No"

    # Add to dataframe strictly for training memory
    df = pd.concat([df, pd.DataFrame([dummy_high]), pd.DataFrame([dummy_low])], ignore_index=True)

    # ------------------------------------------------

    # Basic cleaning
    target_col = "Class/ASD"
    if target_col not in df.columns:
        return None, "Target column missing."

    def clean_target(x):
        return 1 if str(x).lower() in ['yes', '1', 'true', 'asd'] else 0

    df['target'] = df[target_col].apply(clean_target)

    # Features
    feature_cols = [f"A{i}_Score" for i in range(1, 11)]

    # Fill missing columns with 0 if they don't exist yet
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0

    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), feature_cols)
    ])

    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback for very small datasets
        X_train, X_test, y_train, y_test = X, X, y, y

    models = {}

# SVM
    try:
        svm = Pipeline([('pre', preprocessor), ('clf', SVC(probability=True, random_state=42))])
        svm.fit(X_train, y_train)
        models['SVM'] = {'model': svm, 'test_data': (X_test, y_test)}
    except Exception as e:
        return None, f"SVM Error: {e}"

    # XGBoost
    if XGBOOST_AVAILABLE:
        try:
            # Everything inside 'try' must be indented
            xgb = Pipeline([
                ('pre', preprocessor),
                ('clf', XGBClassifier(eval_metric='logloss'))
            ])
            xgb.fit(X_train, y_train)
            models['XGBoost'] = {'model': xgb, 'test_data': (X_test, y_test)}
        except Exception:
            pass
            
    return models, None
# ------------------------
# UI COMPONENTS
# ------------------------

def sidebar_nav():
    # --- Check if image exists before trying to load it ---
    logo_path = "Spectrum Light Logo.jpg"

    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=250)
    else:
        # Fallback to text if image is missing
        st.sidebar.title("🧩 SpectrumLight")

    st.sidebar.markdown("## Risk Scoring Tool")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", ["1. Assessment", "2. Model Analytics"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Note on Scrolling:**\n"
        "This app uses native browser scrolling. "
        "It works automatically with your laptop trackpad or mouse wheel."
    )
    return page


def page_assessment():
    st.title("Autism Pre-Screening (AQ-10)")

    # 1. CONSENT
    with st.expander("⚠️ Disclaimer & Consent (Click to expand)", expanded=False):
        st.warning("This is NOT a diagnostic tool. Consult a professional for diagnosis.")
        consent = st.checkbox("I understand and agree to proceed.")

    if not consent:
        st.stop()

        # 2. DEMOGRAPHICS
    st.subheader("Demographics")

    with st.form("assessment_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", min_value=1, max_value=100, step=1)
        with c2:
            ethnicity = st.selectbox("Ethnicity", ["White-European", "Asian", "Black", "Latino", "Others"])
            jaundice = st.selectbox("Born with Jaundice?", ["No", "Yes"])
        with c3:
            country = st.text_input("Country of Residence", "South Africa")
            relation = st.selectbox("Who is taking this test?", ["Self", "Parent", "Guardian"])

        st.markdown("---")
        st.subheader("AQ-10 Questions")
        st.caption("Please select the option that best describes you.")

        # Questions Logic
        questions = [
            "1. I often notice small sounds when others do not.",
            "2. I usually concentrate more on the whole picture, rather than the small details.",
            "3. I find it easy to do more than one thing at once.",
            "4. If there is an interruption, I can switch back to what I was doing very quickly.",
            "5. I find it easy to 'read between the lines' when someone is talking to me.",
            "6. I know how to tell if someone listening to me is getting bored.",
            "7. When I'm reading a story, I find it difficult to work out the characters' intentions.",
            "8. I like to collect information about categories of things (e.g., cars, birds, trains, plants).",
            "9. I find it easy to work out what someone is thinking or feeling just by looking at their face.",
            "10. I find it difficult to work out people's intentions."
        ]

        # AQ-10 Scoring Key:
        # Agree gives 1 point for: 1, 7, 8, 10
        # Disagree gives 1 point for: 2, 3, 4, 5, 6, 9
        agree_points = {1, 7, 8, 10}

        responses = {}
        options = ["Definitely Agree", "Slightly Agree", "Slightly Disagree", "Definitely Disagree"]

        for i, q in enumerate(questions, 1):
            st.markdown(f"**{q}**")
            responses[i] = st.radio(f"Q{i}", options, index=2, horizontal=True, label_visibility="collapsed",
                                    key=f"q{i}")
            st.markdown("")  # Spacer

        submit_btn = st.form_submit_button("Submit Assessment", type="primary", use_container_width=True)

    if submit_btn:
        # Scoring Logic
        score = 0
        scores_list = []

        for i in range(1, 11):
            ans = responses[i]
            is_agree = "Agree" in ans

            if i in agree_points:
                point = 1 if is_agree else 0
            else:
                point = 1 if not is_agree else 0

            score += point
            scores_list.append(point)

        # Save Logic
        row_data = {
            "age": age, "gender": gender, "ethnicity": ethnicity,
            "jaundice": jaundice, "contry_of_res": country, "relation": relation,
            "Class/ASD": "Yes" if score >= 6 else "No",
            "screening_result": score
        }
        # Add individual scores
        for idx, val in enumerate(scores_list):
            row_data[f"A{idx + 1}_Score"] = val

        new_id = save_submission(row_data)

        # Save to Session State
        st.session_state['last_score'] = score
        st.session_state['last_data'] = pd.DataFrame([row_data])
        # Ensure numeric types for ML
        for col in [f"A{x}_Score" for x in range(1, 11)]:
            st.session_state['last_data'][col] = st.session_state['last_data'][col].astype(float)

        st.success("Assessment Submitted Successfully!")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Total AQ-10 Score", value=f"{score}/10")
        with res_col2:
            if score >= 6:
                st.error("**Result:** High likelihood of ASD traits.")
                st.write("Recommendation: Consider a formal clinical referral.")
            else:
                st.success("**Result:** Low likelihood of ASD traits.")
                st.write("Recommendation: No immediate referral indicated.")


def page_analytics():
    st.title("🔬 Predictive Models")

    if 'last_data' in st.session_state:
        st.info(f"Analyzing submission for ID: {st.session_state['last_data']['ID'].values[0]}")
    else:
        st.warning("No recent submission found. Please complete the Assessment first.")

    # Train/Load Models
    with st.spinner("Training models on dataset..."):
        models, error = train_models(DATASET_PATH)

    if error:
        st.error(error)
        st.write("👉 **Action Required:** Go to the 'Assessment' tab and submit more data to fix this.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📈 ROC Curves", "🤖 Live Prediction"])

    with tab1:
        st.subheader("Model Accuracy on Test Data")
        col1, col2 = st.columns(2)

        def show_metrics(name, m_data, col):
            model = m_data['model']
            X_test, y_test = m_data['test_data']
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)

            with col:
                st.markdown(f"### {name}")
                st.write(f"**Accuracy:** {acc:.2%}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(y_test, preds, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
                plt.xlabel('Preds')
                plt.ylabel('True')
                st.pyplot(fig)

        if 'SVM' in models: show_metrics("SVM", models['SVM'], col1)
        if 'XGBoost' in models: show_metrics("XGBoost", models['XGBoost'], col2)

    with tab2:
        st.subheader("ROC Curve Analysis")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

        for name, m_data in models.items():
            model = m_data['model']
            X_test, y_test = m_data['test_data']
            if len(np.unique(y_test)) > 1:
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc = roc_auc_score(y_test, probs)
                ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

        ax.legend(loc="lower right")
        st.pyplot(fig)

    with tab3:
        st.subheader("Predicting YOUR Result")
        if 'last_data' not in st.session_state:
            st.write("Submit the form in the 'Assessment' tab to see predictions here.")
        else:
            user_row = st.session_state['last_data']
            cols = st.columns(len(models))

            for idx, (name, m_data) in enumerate(models.items()):
                with cols[idx]:
                    model = m_data['model']
                    try:
                        prob = model.predict_proba(user_row)[0, 1]
                        st.markdown(f"### {name}")
                        st.metric("ASD Probability", f"{prob:.1%}")
                        if prob > 0.5:
                            st.error("Prediction: ASD Traits Detected")
                        else:
                            st.success("Prediction: Non-ASD")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")


# ------------------------
# MAIN EXECUTION
# ------------------------
def main():
    init_dataset()
    page = sidebar_nav()

    if page == "1. Assessment":
        page_assessment()
    elif page == "2. Model Analytics":
        page_analytics()


if __name__ == "__main__":
    main()


