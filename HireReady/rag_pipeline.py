from embedding import generate_embedding
from vector_store import upsert_vector, search_vector
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------
# Load Instruction Model (Local)
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


# -------------------------------
# Ingest Resume / JD into Endee
# -------------------------------

def ingest_text(text, doc_id):
    embedding = generate_embedding(text)
    upsert_vector(doc_id, embedding, {"content": text})


# -------------------------------
# Retrieve Context from Endee
# -------------------------------

def retrieve_context(query, top_k=3):
    query_embedding = generate_embedding(query)
    results = search_vector(query_embedding, top_k=top_k)

    contexts = []

    for match in results:
        if "meta" in match and "content" in match["meta"]:
            contexts.append(match["meta"]["content"])

    return "\n".join(contexts)


# -------------------------------
# Local LLM Generation
# -------------------------------

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=256,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# RAG Response
# -------------------------------

def generate_response(query):
    context = retrieve_context(query)

    questions = []

    for i in range(5):
        prompt = f"""
You are a senior technical interviewer.

Candidate Resume and Job Description:
{context}

Generate ONE advanced technical interview question.

Rules:
- The question must reference specific technologies from the context (e.g., BERT, RAG, vector databases, PyTorch).
- Avoid generic questions like "What skills do you have?"
- Focus on system design, embeddings, model optimization, or deployment.
- Output only the question.
"""

        q = generate_text(prompt).strip()

        # Avoid duplicate questions
        if q not in questions:
            questions.append(q)
        else:
            # If duplicate, slightly modify prompt
            alt_prompt = prompt + "\nMake it different from previous questions."
            q = generate_text(alt_prompt).strip()
            questions.append(q)

    # Number them properly
    formatted = [f"{idx+1}. {q}" for idx, q in enumerate(questions)]

    return "\n\n".join(formatted)

import re

def evaluate_answer(user_answer):
    # Get resume + JD context from retrieval system
    context = retrieve_context(user_answer)

    # Extract important words from context (simple keyword extraction)
    context_words = re.findall(r'\b[a-zA-Z]{4,}\b', context.lower())
    unique_keywords = list(set(context_words))

    # Remove very common words
    stopwords = {
        "with", "from", "that", "this", "have", "will",
        "your", "about", "using", "into", "work", "role",
        "skills", "experience", "years"
    }

    keywords = [w for w in unique_keywords if w not in stopwords][:15]

    # Count keyword matches in user answer
    matches = sum(1 for k in keywords if k in user_answer.lower())

    # Length score
    length_score = min(len(user_answer.split()) // 10, 3)

    score = min(matches + length_score, 10)

    strengths = []
    weaknesses = []

    if matches >= 3:
        strengths.append("References relevant concepts from the resume/job description.")
    else:
        weaknesses.append("Does not strongly reference key concepts from the role.")

    if len(user_answer.split()) > 20:
        strengths.append("Provides reasonably detailed explanation.")
    else:
        weaknesses.append("Answer lacks sufficient depth.")

    if not weaknesses:
        weaknesses.append("Could include measurable results or impact.")

    return f"""
Strengths:
- {strengths[0] if strengths else "Relevant to the role."}
- {strengths[1] if len(strengths) > 1 else "Technically aligned response."}

Weaknesses:
- {weaknesses[0] if weaknesses else "Minor areas for improvement."}
- {weaknesses[1] if len(weaknesses) > 1 else "Could provide more detail."}

Final Score: {score}/10
"""
