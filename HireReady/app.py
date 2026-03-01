import streamlit as st
import PyPDF2
from rag_pipeline import ingest_text, generate_response, evaluate_answer

st.set_page_config(page_title="AI Interview Simulator", layout="wide")

st.title("RAG-Powered AI Interview Simulator")
st.write("Upload Resume & Job Description, then generate interview questions.")

# Resume input
uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

resume_text = ""

if uploaded_resume is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_resume)
    for page in pdf_reader.pages:
        resume_text += page.extract_text() or ""

    st.success("Resume uploaded successfully!")
else:
    resume_text = st.text_area("Or Paste Resume Text")

# JD input
jd_text = st.text_area("Paste Job Description")

if st.button("Analyze & Generate Questions"):
    if resume_text and jd_text:
        ingest_text(resume_text, "resume_doc")
        ingest_text(jd_text, "jd_doc")

        response = generate_response(
            "Create technical interview questions."
        )

        st.subheader("Generated Interview Questions")
        st.write(response)
    else:
        st.warning("Please provide both resume and job description.")

st.markdown("---")

st.subheader("Mock Interview Mode")

user_answer = st.text_area("Answer the question here")

if st.button("Evaluate Answer"):
    if user_answer:
        feedback = evaluate_answer(user_answer)
        st.write(feedback)
    else:
        st.warning("Please provide an answer to evaluate.")
