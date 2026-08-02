# AI Resume Screening & Recruitment Analytics

A Retrieval-Augmented Generation (RAG) application that lets recruiters ask natural-language questions about a candidate pool and get relevant, source-backed answers.

**Live app:** https://ai-resume-screening-rag-mny6rsppocmhr8btraquhx.streamlit.app

## Overview

Recruiters often need to sift through hundreds of resumes to find the right candidates. This project builds an AI assistant that retrieves the most relevant resumes for a given question (e.g. *"who are the best candidates for a Data Analyst role?"*) using semantic search, then generates a natural-language answer grounded in those resumes.

## Pipeline

```
Dataset Collection → Document Loader → Preprocessing → Chunking
→ Embedding Generation → Vector Database → Retriever
→ Prompt Template → LLM → Generated Response
```

## Tech Stack

| Component | Choice |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (384-dim) |
| Vector Database | ChromaDB |
| LLM | `google/flan-t5-base` |
| App Framework | Streamlit |
| Data | 550 synthetic candidate records (Excel) |

## Features

- Natural-language chat interface
- File upload for custom candidate datasets
- Source document display (shows exactly which resumes an answer came from)
- Conversation history within a session
- Downloadable responses
- Error handling for missing/malformed data

## Project Structure

```
├── app.py                              # Streamlit RAG application
├── requirements.txt                    # Python dependencies
├── RESUME_ANALYSIS_LLM_PROJECT.xlsx    # Candidate dataset (raw, cleaned, chunked)
├── Project_Documentation.docx          # Full project write-up
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), connected directly to this repository.

## Sample Questions to Try

- "Who are the best candidates for a Data Analyst role?"
- "Find candidates with more than 5 years of experience"
- "Who has AWS or cloud certifications?"
- "Compare candidates for a Machine Learning Engineer role and explain why"

## Known Limitations

- The LLM (FLAN-T5-base) retrieves a fixed top-k context window per query, so it is not reliable for exact numeric aggregation (counts, averages) across the full dataset — it's best suited for qualitative candidate matching.
- Answer detail varies with how specific the question is phrased.

## Future Improvements

- Upgrade to a larger LLM (e.g. Mistral-7B) for richer answers, given more compute.
- Add a hybrid query router for numeric/aggregate questions.
- Process real image (OCR), audio (Speech-to-Text), and video (frame extraction) resume submissions.

## Author

Submitted as part of a Large Language Models (LLMs) & RAG course project.
