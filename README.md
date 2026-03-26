# VisionRAG is a Multimodal Retrieval Augmented Generation system. In plain terms: you upload a PDF, the system reads everything in it (text, tables, images, charts), stores the meaning of each piece as a mathematical vector, and when you ask a question, it finds the most relevant pieces and sends them to an AI (Gemini) to generate an answer. The "multimodal" part means it understands both text AND images.

# What each cell does
- Cell 1 — Install Dependencies
Installs every library the project needs. This includes PyMuPDF for reading PDFs, PaddleOCR for reading text from images, SentenceTransformers and CLIP for converting content into vectors, Qdrant for storing those vectors, and the Gemini API for generating answers.
- Cell 2 — Import Libraries
Loads all the installed packages into memory and sets up a logger so you can see what is happening step by step.
- Cell 3 — Configure Gemini API
Connects to Google's Gemini 1.5 Flash model using your API key. Replace the placeholder "XXXXXXXX" with a real key from https://makersuite.google.com/app/apikey. The function call_gemini() is defined here and reused everywhere that needs an LLM response.
- Cell 4 — Project Folder Structure
Creates a clean folder tree on your local disk: visionrag_data/raw/ for PDFs, extracted/ for parsed content, ocr/ for OCR output, chunks/ for text chunks, embeddings/ for saved vectors, db/ for the database, and qdrant_storage/ for the vector database.
- Cell 5 — Relational Schema (SQLite)
Creates a local SQLite database with two tables. The documents table records every PDF you upload (id, filename, date). The chunks table records every extracted piece of content (id, doc_id, modality, text, page number, bounding box, vector id). This is the metadata layer that connects your database records to the vector store.
- Cell 6 — Load or Generate Example PDF
Automatically generates a synthetic financial report PDF (Acme Corp 2024) containing real paragraphs, a formatted table with financial metrics, and an embedded chart with a bar graph and pie chart. You can also download a real PDF (links are provided in the notebook header) and point PDF_PATH to it instead.
- Cell 7 — Document Parsing Pipeline
Opens the PDF with PyMuPDF and goes through it page by page. It extracts every text block with its position, extracts every embedded image and saves it as a PNG file, and detects tables using PyMuPDF's built-in find_tables() method. Every element gets a page number and a bounding box (coordinates on the page).
- Cell 8 — OCR Pipeline
Takes each extracted image and runs PaddleOCR on it. PaddleOCR reads text that is inside the image — things like axis labels on charts, numbers in diagrams, or text in scanned pages. The detected text is attached to the image element so it can later be searched as text.
- Cell 9 — Chunking Strategy
Long text blocks are split into smaller pieces of 150 words each, with 30 words of overlap between consecutive chunks. The overlap preserves context at the boundaries — the end of one chunk and the start of the next share the same 30 words, so sentences that span the boundary are not broken. Tables and images are kept as single atomic chunks and not split.
- Cell 10 — Text Embedding Pipeline
Uses the all-MiniLM-L6-v2 model from SentenceTransformers to convert every text and table chunk into a 384-dimensional vector. These vectors capture semantic meaning — two chunks about the same topic will have vectors that are mathematically close to each other. The vectors are saved to disk as .npy files.
- Cell 11 — Image Embedding Pipeline (CLIP)
Uses OpenAI's CLIP model (ViT-B/32) to convert each extracted image into a 512-dimensional vector. CLIP is special because it was trained on image-text pairs, so its text encoder and image encoder share the same vector space. This means you can search for an image by typing a text query — the query is encoded with CLIP's text encoder, and the result is compared against image vectors.
- Cell 12 — Setup Qdrant Vector Database
Starts a local Qdrant instance (stored as files in qdrant_storage/) and creates two collections: text_chunks for 384-dim text/table vectors, and image_chunks for 512-dim CLIP image vectors. Qdrant uses HNSW indexing for fast approximate nearest-neighbour search.
- Cell 13 — Store Embeddings
Inserts all the computed vectors into Qdrant as PointStruct objects. Each point carries a payload (metadata) containing the chunk id, document id, modality, page number, bounding box, and a text preview. This metadata is what gets returned during retrieval so you know where each result came from.
- Cell 14 — Hybrid Retrieval System
Defines two retrieval functions. retrieve_text() encodes the query with SentenceTransformer and searches the text_chunks collection. retrieve_images() encodes the query with CLIP's text encoder and searches the image_chunks collection. Both return ranked results with scores.
- Cell 15 — Cross-Modal Reranker
Combines the text results and image results into one unified ranked list. Image scores are multiplied by a weight factor (default 0.7) to account for the fact that CLIP and SentenceTransformer scores come from different distributions. The combined list is sorted by final score. The format_context_for_llm() function then assembles the top results into a readable context string for Gemini.
- Cell 16 — RAG Reasoning with Gemini
Sends the retrieved context and the user's question to Gemini with a structured prompt. If the top result includes an image, the actual PIL image is also passed to Gemini for visual reasoning. Gemini is instructed to answer only from the provided context and to cite page numbers.
- Cell 17 — End-to-End Pipeline
The query_pipeline() function assembles all stages: retrieve text → retrieve images → rerank → call Gemini → return answer with sources. This is the single function you call in production.
- Cell 18 — Evaluation Tests
Runs five representative questions through the full pipeline and prints each answer with its top source citation. This lets you verify the system is working correctly before deploying.
- Cell 19 — FastAPI Endpoint
Creates a production-ready REST API with a POST /query endpoint. It runs in a background thread so the notebook stays usable. The endpoint accepts a JSON body with question and optional tuning parameters, and returns the answer plus sources.
- Cell 20 — Performance Improvements
A discussion cell covering six optimizations: batch embedding, HNSW index tuning, Redis caching at the query level, async background workers for ingestion, int8 vector quantization in Qdrant (4x storage reduction), and better document parsers for production use.
- Cell 21 — Production Deployment Guide
Instructions for converting the notebook into a proper Python package, a Dockerfile and docker-compose.yml for containerized deployment, a Next.js frontend integration example, and a table of suggested improvements to make the project portfolio-ready.

