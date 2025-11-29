import os
import streamlit as st
import pdfplumber
import re
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

# --------- CONFIG ---------
# Load API key from .env (DO NOT hard-code it in this file)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Question Paper Pattern Predictor")

st.title("📊 University Question Paper Pattern Predictor")
st.write("Upload your syllabus and previous question papers to get smart exam insights.")

# ------------ FILE UPLOADS ------------

question_papers = st.file_uploader(
    "Upload previous question papers (PDF format)",
    type=["pdf"],
    accept_multiple_files=True
)

syllabus_file = st.file_uploader(
    "Upload syllabus (PDF format)",
    type=["pdf"]
)

# ------------ HELPERS ------------

def extract_text_from_pdf(uploaded_file):
    """Read all text from a PDF uploaded via Streamlit."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_into_questions(text: str):
    """
    Very simple splitter:
    assumes questions are numbered like:
    1. Question text
    2. Another question
    """
    # Normalize line breaks
    text = re.sub(r'\r', '', text)

    # Regex: number + dot + space, capture blocks until next number
    pattern = r"\n?\s*\d+[\).\]]\s+"
    parts = re.split(pattern, text)

    # First part before Q1 will be junk header, skip it
    questions = []
    for q in parts[1:]:
        q_clean = q.strip()
        # Ignore extremely short lines
        if len(q_clean) > 15:
            questions.append(q_clean)
    return questions


def call_llm(prompt: str) -> str:
    """Call OpenAI chat model and return plain text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

# ------------ SYLLABUS ANALYSIS ------------

if syllabus_file:
    st.success("Syllabus file uploaded.")

    if st.button("🔍 Analyze Syllabus"):
        with st.spinner("Analyzing syllabus and generating questions..."):
            syllabus_text = extract_text_from_pdf(syllabus_file)

            syllabus_prompt = f"""
You are an exam-focused assistant for a university BCA subject.
Use ONLY the syllabus below to prepare student-friendly output.

Do ALL of the following:

1) Give a short summary of the syllabus in 3–5 bullet points.
2) List the MOST important topics to study (5–10 bullet points).
3) Generate 8–10 expected theory questions (mix of long-answer and short notes).
   - Do NOT write answers, only questions.
4) Generate 8–10 multiple choice questions (MCQs) based on the syllabus.
   - Each MCQ should have 4 options: a), b), c), d)
   - Clearly mark the correct option, like: **Correct: b)**

Keep the language simple and clear, as if explaining to an average BCA student.

Syllabus:
{syllabus_text}
"""
            syllabus_output = call_llm(syllabus_prompt)

        st.subheader("📘 Syllabus Insights, Expected Questions & MCQs")
        st.markdown(syllabus_output)

# ------------ QUESTION PAPER ANALYSIS ------------

if question_papers:
    st.success(f"{len(question_papers)} question paper(s) uploaded.")

    if st.button("📊 Analyze Question Papers"):
        with st.spinner("Reading question papers and finding patterns..."):
            all_questions = []

            # Extract questions from each paper
            for qp in question_papers:
                text = extract_text_from_pdf(qp)
                questions = split_into_questions(text)
                all_questions.extend(questions)

            if not all_questions:
                st.error(
                    "Could not find questions. "
                    "Check if the PDFs are scanned images instead of text."
                )
            else:
                # Count repeated questions (exact match, basic)
                normalized = [q.lower().strip() for q in all_questions]
                counts = Counter(normalized)

                repeated = [(q, c) for q, c in counts.items() if c > 1]
                # Map back to original text for display
                repeated_readable = []
                for q_norm, c in repeated:
                    orig = next(q for q in all_questions if q.lower().strip() == q_norm)
                    repeated_readable.append((orig, c))

        if all_questions:
            st.subheader("❓ Total questions found")
            st.write(len(all_questions))

            # Show most repeated questions (if any)
            if repeated_readable:
                st.subheader("🔁 Most Repeated Questions")
                for q, c in sorted(repeated_readable, key=lambda x: x[1], reverse=True):
                    st.markdown(f"**({c} papers)** {q}")
            else:
                st.info("No exactly repeated questions found across the uploaded papers.")

            # Ask AI to guess high-chance topics based on all questions
            joined_questions = "\n".join(all_questions[:50])  # limit prompt size
            pattern_prompt = (
                "You are helping a student prepare for exams.\n"
                "Below are many previous exam questions for one subject.\n"
                "1) Identify the main topics that appear again and again.\n"
                "2) List 5–10 'high chance' topics for the next exam.\n"
                "3) Suggest a short study plan in simple language.\n\n"
                f"Questions:\n{joined_questions}"
            )
            with st.spinner("Asking AI for high-chance topics..."):
                pattern_insights = call_llm(pattern_prompt)

            st.subheader("📈 High-Chance Topics & Study Plan")
            st.write(pattern_insights)

# ------------ BOTH TOGETHER (OPTIONAL NOTE) ------------
if syllabus_file and question_papers:
    st.markdown("---")
    st.caption(
        "Tip: When both syllabus and question papers are uploaded, "
        "you can first analyze the syllabus, then analyze question papers "
        "to get a full picture."
    )
