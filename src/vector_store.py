"""
Vector Store Module
Manages FAISS vector database for document embeddings and similarity search.
"""

import os
import pickle
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStore:
    """Manage FAISS vector store for document retrieval."""
    
    def __init__(self, store_path: str = "vector_store"):
        """
        Initialize the vector store.
        
        Args:
            store_path: Directory to save/load the vector store
        """
        self.store_path = store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
    
    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed
        """
        print("\n🧮 Creating embeddings and vector store...")
        print("⏳ This may take a few minutes on first run...")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"✓ Vector store created with {len(documents)} document chunks")
        
        # Save to disk
        self.save()
    
    def save(self) -> None:
        """Save the vector store to disk."""
        if self.vectorstore is None:
            raise ValueError("No vector store to save. Create one first.")
        
        os.makedirs(self.store_path, exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(self.store_path)
        
        print(f"💾 Vector store saved to: {self.store_path}")
    
    def load(self) -> bool:
        """
        Load existing vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.store_path, "index.faiss")
        
        if not os.path.exists(index_path):
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                self.store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Vector store loaded from: {self.store_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading vector store: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents using similarity search.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create one first.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Get relevant documents without scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        results = self.search(query, k)
        return [doc for doc, score in results]
    
    def exists(self) -> bool:
        """Check if vector store exists on disk."""
        index_path = os.path.join(self.store_path, "index.faiss")
        return os.path.exists(index_path)


def create_vector_store(documents: List[Document], store_path: str = "vector_store") -> VectorStore:
    """
    Helper function to create a vector store from documents.
    
    Args:
        documents: List of Document objects
        store_path: Directory to save the vector store
        
    Returns:
        VectorStore object
    """
    vs = VectorStore(store_path)
    vs.create_from_documents(documents)
    return vs


def load_vector_store(store_path: str = "vector_store") -> VectorStore:
    """
    Helper function to load an existing vector store.
    
    Args:
        store_path: Directory containing the vector store
        
    Returns:
        VectorStore object
    """
    vs = VectorStore(store_path)
    if not vs.load():
        raise FileNotFoundError(
            f"Vector store not found at {store_path}. "
            "Run setup.py first to create it."
        )
    return vs


if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store...\n")
    
    # Create sample documents
    test_docs = [
        Document(
            page_content="The CSE department offers undergraduate and graduate programs.",
            metadata={"source": "test.txt"}
        ),
        Document(
            page_content="Dr. Maheen Islam is the department chairperson.",
            metadata={"source": "test.txt"}
        )
    ]
    
    vs = create_vector_store(test_docs, "test_vector_store")
    
    # Test search
    results = vs.search("Who is the chairperson?", k=1)
    print(f"\n✓ Search test successful!")
    print(f"  Query: 'Who is the chairperson?'")
    print(f"  Result: {results[0][0].page_content}")
