"""
AI-Powered Student Performance Prediction & Study Recommendation System
Run: pip install streamlit pandas scikit-learn matplotlib
     streamlit run app.py

Educational demo. The included dataset is synthetic, not real student data.
Predictions are estimates and must not be used for high-stakes decisions.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Success AI", page_icon="🎓", layout="wide")
st.title("🎓 Student Success AI")
st.caption("Machine-learning prediction • Data visualization • Personalized study guidance")
st.info("Demo project using a small synthetic sample dataset. Results are illustrative, not a guarantee of academic performance.")

@st.cache_data
def load_data():
    return pd.read_csv("sample_student_data.csv")

data = load_data()
features = ["study_hours_per_day", "absences", "attendance_percent",
            "assignment_score", "previous_test_score"]
X, y = data[features], data["passed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predict a student", "📘 About the project"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample records", len(data))
    c2.metric("Pass rate in sample", f"{data['passed'].mean()*100:.1f}%")
    c3.metric("Demo model accuracy", f"{accuracy*100:.1f}%")
    left, right = st.columns(2)
    with left:
        st.subheader("Study hours vs. test score")
        fig, ax = plt.subplots()
        ax.scatter(data["study_hours_per_day"], data["previous_test_score"])
        ax.set_xlabel("Study hours per day")
        ax.set_ylabel("Previous test score")
        ax.grid(alpha=.25)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Pass / not-pass distribution")
        counts = data["passed"].map({1:"Pass", 0:"Not pass"}).value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of students")
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Sample data")
    st.dataframe(data, use_container_width=True)

with tab2:
    st.subheader("Enter student details")
    a, b = st.columns(2)
    with a:
        hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
        absences = st.slider("Number of absences", 0, 30, 3)
        attendance = st.slider("Attendance (%)", 0, 100, 75)
    with b:
        assignment = st.slider("Assignment score (%)", 0, 100, 70)
        previous = st.slider("Previous test score (%)", 0, 100, 65)
    if st.button("Predict performance", type="primary"):
        row = pd.DataFrame([[hours, absences, attendance, assignment, previous]], columns=features)
        probability = model.predict_proba(row)[0][1]
        prediction = int(model.predict(row)[0])
        if prediction:
            st.success(f"Model estimate: PASS ({probability*100:.1f}% estimated probability)")
        else:
            st.warning(f"Model estimate: AT-RISK ({probability*100:.1f}% estimated probability)")
        advice = []
        if hours < 3: advice.append("Build a consistent study routine, starting with 1–2 focused sessions.")
        if attendance < 75: advice.append("Improve attendance and revise missed classroom topics.")
        if assignment < 60: advice.append("Practise assignment questions and ask for feedback.")
        if previous < 50: advice.append("Revise foundational concepts and solve previous question papers.")
        if absences > 5: advice.append("Plan ahead to reduce avoidable absences.")
        if not advice: advice.append("Maintain your routine; use weekly revision and practice tests.")
        st.subheader("Personalized study recommendations")
        for item in advice:
            st.write("• " + item)
        st.caption("Recommendations are rule-based examples. The prediction is produced by a Random Forest classifier.")

with tab3:
    st.subheader("Project overview")
    st.write("This project demonstrates how Python and machine learning can explore learning patterns and provide study guidance.")
    st.markdown("""
    **Objectives**
    - Analyse student learning indicators using Python and Pandas.
    - Visualize sample data with Matplotlib.
    - Train a Random Forest classification model.
    - Estimate whether a sample student may pass.
    - Provide simple, explainable study recommendations.

    **Technologies:** Python, Pandas, Matplotlib, Scikit-learn and Streamlit.

    **Method:** Load data → clean/select features → split into training and testing sets
    → train model → evaluate → predict → show recommendations.

    **Limitations:** The sample dataset is small and synthetic. Accuracy can change with
    different data splits. Real deployment requires a larger, representative, consented
    dataset, validation, privacy safeguards and review by educators.
    """)
    st.subheader("Model feature importance")
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Relative importance")
    st.pyplot(fig)
    plt.close(fig)
