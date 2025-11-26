"""
Document Loader Module
Handles loading and chunking of university documents for RAG pipeline.
"""

import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """Load and process text documents from the documents directory."""
    
    def __init__(self, documents_dir: str = "documents"):
        """
        Initialize the document loader.
        
        Args:
            documents_dir: Path to directory containing text documents
        """
        self.documents_dir = documents_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """
        Load all text documents from the documents directory.
        
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        
        if not os.path.exists(self.documents_dir):
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_dir}\n"
                "Please create the 'documents/' folder and add your university data files."
            )
        
        txt_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.txt')]
        
        if not txt_files:
            raise ValueError(
                f"No .txt files found in {self.documents_dir}\n"
                "Please add university data text files to the documents folder."
            )
        
        for filename in txt_files:
            filepath = os.path.join(self.documents_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "file_path": filepath
                    }
                )
                documents.append(doc)
                
                print(f"✓ Loaded: {filename} ({len(content)} characters)")
                
            except Exception as e:
                print(f"✗ Error loading {filename}: {str(e)}")
                continue
        
        print(f"\n📚 Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        print("\n🔪 Chunking documents...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def load_and_chunk(self) -> List[Document]:
        """
        Convenience method to load and chunk documents in one step.
        
        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
        return chunks


def load_documents(documents_dir: str = "documents") -> List[Document]:
    """
    Helper function to load and chunk documents.
    
    Args:
        documents_dir: Path to directory containing text documents
        
    Returns:
        List of chunked Document objects
    """
    loader = DocumentLoader(documents_dir)
    return loader.load_and_chunk()


if __name__ == "__main__":
    # Test the document loader
    print("Testing Document Loader...\n")
    docs = load_documents()
    
    if docs:
        print(f"\n✓ Successfully loaded and chunked documents!")
        print(f"  First chunk preview: {docs[0].page_content[:200]}...")
        print(f"  Metadata: {docs[0].metadata}")
