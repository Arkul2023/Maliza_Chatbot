import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
import random
import time
import os

# Function to load data and models
@st.cache_resource
def load_data_and_models(csv_file):
    df = pd.read_csv(csv_file, encoding='windows-1252')
    corpus_instructions = list(df['Instruction'])
    corpus_responses = list(df['Response'])

    # Load Flan-T5 Small for speed and instruction tuning
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cuda" if torch.cuda.is_available() else "cpu")

    embedder = SentenceTransformer('all-MiniLM-L6-v2').to("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embedder.encode(corpus_instructions, convert_to_tensor=True)

    return df, corpus_instructions, corpus_responses, tokenizer, model, embedder, embeddings

# Function to save the fine-tuned model locally
def save_model(model, tokenizer, model_save_path="malizia_model"):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

# Function to load the saved model
def load_saved_model(model_save_path="malizia_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

# Function to get RAG response
def get_rag_response(user_query, corpus_instructions, corpus_responses, embedder, tokenizer, model, embeddings, top_k=3):
    query_embedding = embedder.encode([user_query], convert_to_tensor=True)

    # Ensure tensors are on the same device
    device = model.device
    query_embedding = query_embedding.to(device).float()
    embeddings = embeddings.to(device).float()

    similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    top_indices = torch.argsort(similarities, descending=True)[:top_k].cpu().numpy()

    retrieved_contexts = "\n".join([corpus_responses[idx] for idx in top_indices])
    retrieved_contexts = retrieved_contexts[:500]  # truncate to avoid long input

    full_input = f"chat with context: {retrieved_contexts}\nQuestion: {user_query}"
    input_ids = tokenizer.encode(full_input, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,         # shorter response
            do_sample=True,         # faster than beam search
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("🤖 MaliziaSign Chatbot with Clickable Suggestions")

csv_file = "malizia_sign_complete_dataset.csv"
df, corpus_instructions, corpus_responses, tokenizer, model, embedder, embeddings = load_data_and_models(csv_file)

# Device info (optional to keep)
device = model.device
st.sidebar.success(f"Using device: {device}")

# Session state
if "suggestions" not in st.session_state:
    st.session_state.suggestions = random.sample(corpus_instructions, 4)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for msg in st.session_state.chat_history:
    st.markdown(f"**You**: {msg['question']}")
    st.markdown(f"**Bot**: {msg['answer']}")

# Clickable suggestion buttons
st.markdown("### 💡 Suggested Questions:")
cols = st.columns(4)
for i, col in enumerate(cols):
    if col.button(st.session_state.suggestions[i]):
        user_question = st.session_state.suggestions[i]
        bot_response = get_rag_response(user_question, corpus_instructions, corpus_responses, embedder, tokenizer, model, embeddings)
        st.session_state.chat_history.append({"question": user_question, "answer": bot_response})
        st.rerun()

# Input handler to process text_input on Enter
def handle_input():
    user_input = st.session_state.user_input
    if user_input.strip():
        bot_response = get_rag_response(
            user_input,
            corpus_instructions,
            corpus_responses,
            embedder,
            tokenizer,
            model,
            embeddings
        )
        st.session_state.chat_history.append({"question": user_input, "answer": bot_response})
        st.session_state.user_input = ""  # Clear input after processing
        st.rerun()

# Text input field (auto-submits on Enter)
st.text_input(
    "Or ask your own question:",
    key="user_input",
    on_change=handle_input,
)

# Refresh suggestions
if st.button("🔄 Refresh Suggestions"):
    st.session_state.suggestions = random.sample(corpus_instructions, 4)
    st.rerun()

# Save the model locally after training (you can call this when done with fine-tuning)
if st.button("💾 Save Model"):
    save_model(model, tokenizer)
    st.success("Model has been saved locally!")
