#!/usr/bin/env python3
"""
RAG + LLM Service - Combines FAISS-based retrieval with quantized LLM
Provides contextual responses based on technical documentation
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from flask import Flask, jsonify, request
import sqlite3
import hashlib

# FAISS for vector similarity search
try:
    import faiss
except ImportError:
    logger.warning("FAISS not available, using fallback similarity search")
    faiss = None

from nano_llm import NanoLLM, ChatHistory, ChatTemplate
from nano_llm.utils import ArgParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document in the knowledge base"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    query: str
    retrieved_docs: List[Document]
    llm_response: str
    confidence: float
    timestamp: str

class FAISSRetriever:
    """FAISS-based document retrieval system"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.doc_embeddings = []

        # Initialize FAISS index
        if faiss:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
        else:
            logger.warning("FAISS not available, using simple similarity search")

    def add_document(self, doc: Document):
        """Add a document to the retrieval system"""
        if doc.embedding is not None:
            self.documents.append(doc)
            self.doc_embeddings.append(doc.embedding)

            if self.index and faiss:
                # Normalize embedding for cosine similarity
                embedding = doc.embedding.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(embedding)
                self.index.add(embedding)

            logger.debug(f"Added document: {doc.id}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if not self.documents:
            return []

        if self.index and faiss:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))

            return results
        else:
            # Fallback: simple cosine similarity
            similarities = []
            for i, doc_emb in enumerate(self.doc_embeddings):
                similarity = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                similarities.append((self.documents[i], similarity))

            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]

    def save_index(self, filepath: str):
        """Save the FAISS index and documents"""
        if self.index and faiss:
            faiss.write_index(self.index, f"{filepath}.faiss")

        with open(f"{filepath}.docs", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_embeddings': self.doc_embeddings
            }, f)

        logger.info(f"Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load the FAISS index and documents"""
        try:
            if faiss and os.path.exists(f"{filepath}.faiss"):
                self.index = faiss.read_index(f"{filepath}.faiss")

            if os.path.exists(f"{filepath}.docs"):
                with open(f"{filepath}.docs", 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.doc_embeddings = data['doc_embeddings']

                logger.info(f"Loaded index from {filepath}")
                return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")

        return False

class NanoDBConnector:
    """Connector for NanoDB diagram database"""

    def __init__(self, db_path: str = "/data/nanodb/diagrams.db"):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Connect to NanoDB"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            logger.info(f"Connected to NanoDB: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to NanoDB: {e}")
            return False

    def search_diagrams(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant diagrams"""
        if not self.connection:
            return []

        try:
            cursor = self.connection.cursor()
            # Simple text search in diagram metadata
            cursor.execute("""
                SELECT id, title, description, file_path, metadata
                FROM diagrams
                WHERE title LIKE ? OR description LIKE ? OR metadata LIKE ?
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'file_path': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {}
                })

            return results

        except Exception as e:
            logger.error(f"Failed to search diagrams: {e}")
            return []

class RAGLLMService:
    """RAG + LLM service combining retrieval with language generation"""

    def __init__(self, llm_model: str = "microsoft/DialoGPT-medium", api: str = "mlc"):
        self.llm_model_name = llm_model
        self.api = api
        self.llm = None
        self.chat_history = None
        self.retriever = FAISSRetriever()
        self.nanodb = NanoDBConnector()
        self.query_history = []

    def initialize_llm(self):
        """Initialize the quantized LLM"""
        try:
            # Use the API specified in constructor
            if self.api == 'hf':
                self.llm = NanoLLM.from_pretrained(
                    self.llm_model_name,
                    api='hf',
                    max_context_len=2048
                )
            else:
                # Default to MLC
                self.llm = NanoLLM.from_pretrained(
                    self.llm_model_name,
                    api='mlc',
                    quantization='q4f16_ft',
                    max_context_len=2048,
                    vision_api='auto'
                )

            self.chat_history = ChatHistory(self.llm)
            logger.info(f"Loaded LLM model: {self.llm_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            return False

    def load_knowledge_base(self, kb_path: str = "/data/knowledge_base"):
        """Load the technical knowledge base"""
        if not os.path.exists(kb_path):
            logger.warning(f"Knowledge base path not found: {kb_path}")
            self._create_sample_knowledge_base()
            return

        # Try to load existing index
        if self.retriever.load_index(os.path.join(kb_path, "index")):
            logger.info("Loaded existing knowledge base index")
            return

        # Build new index from documents
        docs_path = os.path.join(kb_path, "documents")
        if os.path.exists(docs_path):
            self._build_knowledge_base(docs_path)
        else:
            self._create_sample_knowledge_base()

    def _create_sample_knowledge_base(self):
        """Create a sample knowledge base for demonstration"""
        sample_docs = [
            Document(
                id="conn_001",
                content="12-pin connector specifications: Standard automotive connector with power and signal pins. Pin 1-4: Power (12V), Pin 5-8: Ground, Pin 9-12: Signal lines. Torque specification: 5-7 Nm.",
                metadata={"type": "connector", "category": "automotive", "part_number": "CONN-12P-001"},
                embedding=np.random.rand(768)  # Placeholder embedding
            ),
            Document(
                id="warn_001",
                content="ABS warning light troubleshooting: 1. Check brake fluid level 2. Inspect wheel speed sensors 3. Scan for diagnostic codes 4. Check ABS fuse 5. Inspect wiring harness for damage",
                metadata={"type": "troubleshooting", "system": "ABS", "urgency": "high"},
                embedding=np.random.rand(768)
            ),
            Document(
                id="proc_001",
                content="Connector replacement procedure: 1. Disconnect battery 2. Remove old connector 3. Clean contact area 4. Apply dielectric grease 5. Install new connector 6. Torque to specification 7. Test connection",
                metadata={"type": "procedure", "category": "maintenance", "safety_level": "medium"},
                embedding=np.random.rand(768)
            ),
            Document(
                id="safety_001",
                content="Electrical safety procedures: Always disconnect power before working on electrical systems. Use proper PPE including insulated gloves. Verify zero energy state with multimeter. Follow lockout/tagout procedures.",
                metadata={"type": "safety", "category": "electrical", "mandatory": True},
                embedding=np.random.rand(768)
            )
        ]

        for doc in sample_docs:
            self.retriever.add_document(doc)

        logger.info("Created sample knowledge base")

    def _build_knowledge_base(self, docs_path: str):
        """Build knowledge base from document files"""
        # This would parse actual documentation files
        # For now, we'll use the sample knowledge base
        self._create_sample_knowledge_base()

    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents for a query"""
        # Create a simple query embedding (in practice, use a proper embedding model)
        query_embedding = np.random.rand(768)  # Placeholder

        results = self.retriever.search(query_embedding, k)
        return [doc for doc, score in results if score > 0.1]  # Threshold for relevance

    def generate_response(self, query: str, retrieved_docs: List[Document],
                         include_diagrams: bool = False) -> RetrievalResult:
        """Generate response using RAG + LLM"""

        # Build context from retrieved documents
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(f"Document {doc.id}: {doc.content}")

        context = "\n\n".join(context_parts)

        # Get diagrams if requested
        diagrams = []
        if include_diagrams:
            if self.nanodb.connect():
                diagrams = self.nanodb.search_diagrams(query)

        # Build enhanced prompt
        prompt = f"""Based on the following technical documentation, provide a helpful response to the query.

Context:
{context}

Query: {query}

Please provide:
1. Direct answer to the query
2. Relevant technical details
3. Safety considerations if applicable
4. Step-by-step instructions if applicable

Response:"""

        try:
            # Generate response with LLM
            response = self.llm.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )

            # Calculate confidence based on document relevance
            confidence = 0.8 if retrieved_docs else 0.3

            result = RetrievalResult(
                query=query,
                retrieved_docs=retrieved_docs,
                llm_response=response.text if hasattr(response, 'text') else str(response),
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )

            # Store in query history
            self.query_history.append(result)
            if len(self.query_history) > 100:  # Keep last 100 queries
                self.query_history.pop(0)

            return result

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return RetrievalResult(
                query=query,
                retrieved_docs=retrieved_docs,
                llm_response=f"Error generating response: {e}",
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            )

# Flask REST API
app = Flask(__name__)
rag_service = RAGLLMService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "llm_loaded": rag_service.llm is not None,
        "kb_docs": len(rag_service.retriever.documents),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/query', methods=['POST'])
def process_query():
    """Process a RAG query"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query required"}), 400

    query = data['query']
    include_diagrams = data.get('include_diagrams', False)
    k = data.get('k', 3)

    # Retrieve relevant documents
    retrieved_docs = rag_service.retrieve_documents(query, k)

    # Generate response
    result = rag_service.generate_response(query, retrieved_docs, include_diagrams)

    return jsonify({
        "query": result.query,
        "response": result.llm_response,
        "confidence": result.confidence,
        "retrieved_docs": [
            {
                "id": doc.id,
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.metadata
            }
            for doc in result.retrieved_docs
        ],
        "timestamp": result.timestamp
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Simple chat endpoint"""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt required"}), 400

    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 256)
    temperature = data.get('temperature', 0.7)

    try:
        response = rag_service.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        return jsonify({
            "response": response.text if hasattr(response, 'text') else str(response),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_query_history():
    """Get query history"""
    limit = request.args.get('limit', 10, type=int)
    history = rag_service.query_history[-limit:]

    return jsonify({
        "history": [asdict(result) for result in history],
        "count": len(history),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/kb/stats', methods=['GET'])
def get_kb_stats():
    """Get knowledge base statistics"""
    return jsonify({
        "total_documents": len(rag_service.retriever.documents),
        "model_name": rag_service.llm_model_name,
        "embedding_dimension": rag_service.retriever.embedding_dim,
        "timestamp": datetime.now().isoformat()
    })

def main():
    """Main function to initialize and run the RAG+LLM service"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG + LLM Service")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="LLM model name")
    parser.add_argument("--api", default="mlc", choices=["mlc", "hf"], help="LLM API to use")
    parser.add_argument("--kb-path", default="/data/knowledge_base", help="Knowledge base path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8555, help="Port to bind to")

    args = parser.parse_args()

    # Initialize RAG service
    rag_service.llm_model_name = args.model
    rag_service.api = args.api

    logger.info("Initializing LLM...")
    if not rag_service.initialize_llm():
        logger.error("Failed to initialize LLM")
        return 1

    logger.info("Loading knowledge base...")
    rag_service.load_knowledge_base(args.kb_path)

    # Start Flask server
    logger.info(f"Starting RAG+LLM service on {args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down RAG+LLM service")

    return 0

if __name__ == "__main__":
    exit(main())