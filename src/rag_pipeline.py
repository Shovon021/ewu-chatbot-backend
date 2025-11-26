"""
RAG Pipeline Module
Orchestrates the complete Retrieval-Augmented Generation pipeline.
"""

from typing import Dict, List, Generator
from langchain_core.documents import Document
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from src.query_filter import QueryFilter


class RAGPipeline:
    """Complete RAG pipeline for question answering."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_handler: LLMHandler,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Initialized VectorStore object
            llm_handler: Initialized LLMHandler object
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.query_filter = QueryFilter()
        self.top_k = top_k
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string (without source references).
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string (clean, no sources)
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            # Don't include source information - messaging app style
            context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract unique source filenames from documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of unique source filenames
        """
        sources = set()
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            sources.add(source)
        
        return sorted(list(sources))
    
    def query(self, user_query: str) -> Dict[str, any]:
        """
        Process a user query through the complete RAG pipeline (non-streaming).
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        # Step 1: Check if it's a greeting
        if self.query_filter.is_greeting(user_query):
            greeting_response = self.llm_handler.generate_greeting_response(user_query)
            return {
                'query': user_query,
                'response': greeting_response,
                'sources': [],
                'is_greeting': True,
                'retrieved_docs': 0
            }
        
        # Step 2: Filter non-academic queries
        is_academic, reason = self.query_filter.is_academic_query(user_query)
        
        if not is_academic:
            refusal_message = self.query_filter.get_refusal_message(user_query)
            return {
                'query': user_query,
                'response': refusal_message,
                'sources': [],
                'is_academic': False,
                'filter_reason': reason,
                'retrieved_docs': 0
            }
        
        # Step 3: Retrieve relevant documents
        try:
            documents = self.vector_store.get_relevant_documents(user_query, k=self.top_k)
        except Exception as e:
            return {
                'query': user_query,
                'response': f"Error retrieving information: {str(e)}",
                'sources': [],
                'error': str(e),
                'retrieved_docs': 0
            }
        
        if not documents:
            return {
                'query': user_query,
                'response': "I couldn't find any relevant information in my knowledge base to answer your question.",
                'sources': [],
                'retrieved_docs': 0
            }
        
        # Step 4: Format context
        context = self._format_context(documents)
        sources = self._extract_sources(documents)
        
        # Step 5: Generate response using LLM
        try:
            response = self.llm_handler.generate(
                prompt=user_query,
                context=context,
                temperature=0.1,
                max_tokens=512
            )
        except Exception as e:
            return {
                'query': user_query,
                'response': f"Error generating response: {str(e)}",
                'sources': sources,
                'error': str(e),
                'retrieved_docs': len(documents)
            }
        
        return {
            'query': user_query,
            'response': response,
            'sources': sources,
            'is_academic': True,
            'retrieved_docs': len(documents),
            'context_length': len(context)
        }

    def query_stream(self, user_query: str) -> Generator[str, None, None]:
        """
        Process a user query and yield chunks of the response.
        
        Args:
            user_query: User's question
            
        Yields:
            Chunks of the generated response
        """
        # Step 1: Check if it's a greeting
        if self.query_filter.is_greeting(user_query):
            yield self.llm_handler.generate_greeting_response(user_query)
            return
        
        # Step 2: Filter non-academic queries
        is_academic, reason = self.query_filter.is_academic_query(user_query)
        
        if not is_academic:
            yield self.query_filter.get_refusal_message(user_query)
            return
        
        # Step 3: Retrieve relevant documents
        try:
            documents = self.vector_store.get_relevant_documents(user_query, k=self.top_k)
        except Exception as e:
            yield f"Error retrieving information: {str(e)}"
            return
        
        if not documents:
            yield "I couldn't find any relevant information in my knowledge base to answer your question."
            return
        
        # Step 4: Format context
        context = self._format_context(documents)
        
        # Step 5: Generate streaming response using LLM
        try:
            for chunk in self.llm_handler.generate_stream(
                prompt=user_query,
                context=context,
                temperature=0.1,
                max_tokens=512
            ):
                yield chunk
        except Exception as e:
            yield f"Error generating response: {str(e)}"


def create_rag_pipeline(
    vector_store: VectorStore,
    llm_handler: LLMHandler,
    top_k: int = 5
) -> RAGPipeline:
    """
    Helper function to create a RAG pipeline.
    
    Args:
        vector_store: Initialized VectorStore object
        llm_handler: Initialized LLMHandler object
        top_k: Number of documents to retrieve
        
    Returns:
        RAGPipeline object
    """
    return RAGPipeline(vector_store, llm_handler, top_k)


if __name__ == "__main__":
    # Test the RAG pipeline
    print("Testing RAG Pipeline...\n")
    print("Note: This requires vector store and LLM to be set up first.\n")
    
    from src.vector_store import load_vector_store
    from src.llm_handler import setup_llm
    
    try:
        # Load components
        print("Loading vector store...")
        vs = load_vector_store()
        
        print("Setting up LLM...")
        llm = setup_llm()
        
        # Create pipeline
        print("Creating RAG pipeline...\n")
        rag = create_rag_pipeline(vs, llm)
        
        # Test queries
        test_queries = [
            "Hello!",
            "Who is the CSE department chairperson?",
            "What's the weather today?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            result = rag.query(query)
            print(f"\nResponse:\n{result['response']}")
            print(f"\nMetadata: {result}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to run setup.py first to initialize the system.")
