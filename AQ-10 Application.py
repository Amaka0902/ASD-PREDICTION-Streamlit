import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AQ-10 Autism Test", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #caf0f8;
        color: #03045e;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00b4d8;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧩 Autism Spectrum Quotient (AQ-10) Test for Adults")
st.write("""
This short questionnaire helps identify whether you may benefit from a more detailed autism assessment.  
Please answer each statement below by selecting **1 for Strongly Agree** or **0 for Strongly Disagree**.
""")

st.header("👤 Personal Information")

with st.form("user_info"):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    country = st.text_input("Country of Residence")
    jaundice = st.radio("Have you ever been diagnosed with jaundice?", ["Yes", "No"])
    next_step = st.form_submit_button("Continue to AQ-10 Test")

if next_step:
    st.success(f"Welcome {name}! Please complete the AQ-10 questions below 👇")


# AQ-10 Questions
questions = [
    "I often notice small sounds when others do not.",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I find it easy to do more than one thing at once.",
    "If there is an interruption, I can switch back to what I was doing very quickly.",
    "I find it easy to ‘read between the lines’ when someone is talking to me.",
    "I know how to tell if someone listening to me is getting bored.",
    "When I’m reading a story, I find it difficult to work out the characters’ intentions.",
    "I like to collect information about categories of things (e.g., types of cars, birds, trains, plants, etc.).",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "I find it difficult to work out people’s intentions."
]

# Scoring guide
agree_score_items = [1, 7, 8, 10]   # Agree gives 1
disagree_score_items = [2, 3, 4, 5, 6, 9]  # Disagree gives 1

responses = {}
with st.form("aq10_form"):
    for q in questions:
        responses[q] = st.radio(q, ["1 - Strongly Agree", "0 - Strongly Disagree"], index=None)
    submitted = st.form_submit_button("Submit Test")

if submitted:
    score_data = []
    total_score = 0

    for i, q in enumerate(questions, start=1):
        # Convert response to numeric value
        raw_val = 1 if "1" in responses[q] else 0
        if i in disagree_score_items:
            raw_val = 1 - raw_val
        score_data.append({"Question": f"Q{i}", "Response": responses[q], "Score": raw_val})
        total_score += raw_val

    # Convert to DataFrame
    df = pd.DataFrame(score_data)

    # --- Display Results ---
    st.header("🧮 Results Summary")
    st.subheader(f"Total AQ-10 Score: **{total_score}/10**")

    if total_score >= 6:
        st.warning("⚠️ A score of 6 or above suggests that a specialist diagnostic assessment may be beneficial.")
    else:
        st.success("✅ Your responses do not indicate significant autistic traits.")

    # --- Visuals ---
    st.write("### 📊 Visual Breakdown")

    fig_bar = px.bar(
        df,
        x="Question",
        y="Score",
        text="Score",
        color="Score",
        color_continuous_scale=["#8338ec", "#90e0ef"],
        title="Per-Question Scores (1 = Indicates Autistic Trait)"
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(
        names=["Score", "Remaining"],
        values=[total_score, 10 - total_score],
        color_discrete_sequence=["#0077b6", "#ffb703"],
        title="Overall AQ-10 Result Breakdown"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Save or Download Results ---
    user_data = pd.DataFrame({
        "Name": [name],
        "Email": [email],
        "Age": [age],
        "Country": [country],
        "Jaundice": [jaundice],
        "Total Score": [total_score]
    })

    combined = pd.concat([user_data, df], axis=1)
    csv_data = combined.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv_data,
        file_name=f"{name}_AQ10_results.csv",
        mime="text/csv"
    )