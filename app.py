"""
AI Resume Screening & Recruitment Analytics
A Retrieval-Augmented Generation (RAG) Streamlit app.

Pipeline: Excel data -> Chunking (already done) -> Embeddings (MiniLM)
          -> ChromaDB (vector search) -> FLAN-T5 (answer generation)

To run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="🧑‍💼",
    layout="wide",
)

DEFAULT_DATA_FILE = "RESUME_ANALYSIS_LLM_PROJECT.xlsx"
CHUNK_SHEET_NAME = "chunked data"


# -------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# These only run ONCE per app session, not on every button click,
# which is what keeps the app fast after the first load.
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


@st.cache_resource(show_spinner="Building vector database from your data...")
def build_vector_db(chunks_df: pd.DataFrame):
    embed_model = load_embedding_model()
    client = chromadb.Client()  # in-memory, rebuilt fresh each session
    # Reset collection if it already exists (avoids duplicate-add errors on rerun)
    try:
        client.delete_collection("resume_chunks")
    except Exception:
        pass
    collection = client.create_collection("resume_chunks")

    texts = chunks_df["Chunk_Text"].tolist()
    embeddings = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    collection.add(
        ids=chunks_df["Chunk_ID"].tolist(),
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"Candidate_ID": cid} for cid in chunks_df["Candidate_ID"].tolist()],
    )
    return collection


# -------------------------------------------------------------------
# THE RAG PIPELINE (same logic you already tested in Colab)
# -------------------------------------------------------------------
def rag_pipeline(question: str, collection, tokenizer, model, n_results: int = 3):
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_chunks = results["documents"][0]
    candidate_ids = [m["Candidate_ID"] for m in results["metadatas"][0]]
    candidate_ids = list(dict.fromkeys(candidate_ids))  # remove duplicates, keep order

    context = "\n".join(retrieved_chunks)
    prompt = f"""You are a recruitment assistant. Based on the candidate information below, write 2-3 sentences explaining which candidates best match the question and why.

Candidate Information:
{context}

Question: {question}

Write a detailed answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "question": question,
        "answer": answer,
        "candidate_ids": candidate_ids,
        "retrieved_chunks": retrieved_chunks,
    }


# -------------------------------------------------------------------
# SIDEBAR: Data source + app info
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your processed resume Excel file",
        type=["xlsx"],
        help=f"Must contain a sheet named '{CHUNK_SHEET_NAME}' with columns: "
             f"Chunk_ID, Candidate_ID, Chunk_Text",
    )

    st.divider()
    st.header("ℹ️ About this app")
    st.markdown(
        """
        **AI Resume Screening & Recruitment Analytics**

        A Retrieval-Augmented Generation (RAG) assistant that helps
        recruiters find the best-matching candidates using natural
        language questions.

        **Pipeline:**
        1. Text chunks → Embeddings (`all-MiniLM-L6-v2`)
        2. Embeddings → Vector search (`ChromaDB`)
        3. Retrieved context → Answer (`FLAN-T5-base`)
        """
    )

    if st.button("🗑️ Clear conversation history"):
        st.session_state.chat_history = []
        st.rerun()


# -------------------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------------------
st.title("🧑‍💼 AI Resume Screening Assistant")
st.caption("Ask a natural-language question about your candidate pool. "
           "The assistant retrieves the most relevant resumes and generates an answer.")

# --- Load data (uploaded file takes priority, else fall back to default) ---
try:
    if uploaded_file is not None:
        chunks_df = pd.read_excel(uploaded_file, sheet_name=CHUNK_SHEET_NAME)
        data_source_label = uploaded_file.name
    else:
        chunks_df = pd.read_excel(DEFAULT_DATA_FILE, sheet_name=CHUNK_SHEET_NAME)
        data_source_label = DEFAULT_DATA_FILE

    required_cols = {"Chunk_ID", "Candidate_ID", "Chunk_Text"}
    missing = required_cols - set(chunks_df.columns)
    if missing:
        st.error(f"Your file is missing required columns: {missing}")
        st.stop()

except FileNotFoundError:
    st.error(
        f"No data file found. Please upload a processed resume Excel file "
        f"(with a '{CHUNK_SHEET_NAME}' sheet) using the sidebar."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.success(f"✅ Loaded **{len(chunks_df)}** resume chunks from `{data_source_label}`")

# --- Build/load models and vector DB (cached, so this only runs once) ---
try:
    embed_model = load_embedding_model()
    tokenizer, llm_model = load_llm()
    collection = build_vector_db(chunks_df)
except Exception as e:
    st.error(f"Error setting up the AI pipeline: {e}")
    st.stop()

# --- Chat history state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display past conversation ---
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander(f"📄 Source candidates: {', '.join(entry['candidate_ids'])}"):
            for i, chunk in enumerate(entry["retrieved_chunks"], 1):
                st.markdown(f"**Source {i}:** {chunk}")

# --- Query input ---
user_question = st.chat_input("e.g. Who are the best candidates for a Data Analyst role?")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching resumes and generating an answer..."):
            try:
                result = rag_pipeline(user_question, collection, tokenizer, llm_model)
                st.write(result["answer"])

                with st.expander(f"📄 Source candidates: {', '.join(result['candidate_ids'])}"):
                    for i, chunk in enumerate(result["retrieved_chunks"], 1):
                        st.markdown(f"**Source {i}:** {chunk}")

                # Download this single response as a text file
                download_text = (
                    f"Question: {result['question']}\n\n"
                    f"Answer: {result['answer']}\n\n"
                    f"Source Candidates: {', '.join(result['candidate_ids'])}\n\n"
                    f"Retrieved Text:\n" + "\n---\n".join(result["retrieved_chunks"])
                )
                st.download_button(
                    "⬇️ Download this response",
                    data=download_text,
                    file_name="resume_screening_response.txt",
                    mime="text/plain",
                )

                st.session_state.chat_history.append(result)

            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
