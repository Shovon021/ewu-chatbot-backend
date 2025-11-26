"""
LLM Handler Module
Manages local LLM via Ollama for text generation.
"""

import subprocess
import sys
import os
from typing import Optional, Generator
import ollama
from huggingface_hub import InferenceClient


class LLMHandler:
    """Manage local LLM via Ollama."""
    
    DEFAULT_MODEL = "llama3.2:3b"
    
    # Strict system prompt for academic assistant
    SYSTEM_PROMPT = """You are the EWU University Academic Assistant. You ONLY answer questions related to:
- East West University (EWU)
- Computer Science & Engineering (CSE) Department
- Academic programs, courses, and faculty
- Admissions, scholarships, fees, and university policies

STRICT RULES:
1. Answer ONLY from the provided context below
2. If the answer is not in the context, say "I don't have that information in my knowledge base"
3. NEVER answer general questions (coding help, math problems, weather, recipes, etc.)
4. Be concise, professional, and helpful
5. If asked a greeting, respond politely and briefly
6. NEVER mention or cite sources - just provide the information naturally

When answering:
- Give direct, conversational answers
- Use a friendly, helpful tone like a messaging app
- Keep responses clear and concise"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the LLM handler.
        
        Args:
            model: Name of the Ollama model to use
        """
        self.model = model
        self.client = None
        self.is_local = False
        
        # Check if running locally with Ollama or in cloud
        if self.check_ollama_running():
            print("✓ Local Ollama instance detected. Using local mode.")
            self.is_local = True
        else:
            print("⚠ Ollama not detected. Switching to Cloud/API mode (Hugging Face).")
            self.is_local = False
            # Use a similar model for cloud inference
            self.hf_model = "meta-llama/Llama-3.2-3B-Instruct" 
            token = os.environ.get("HF_TOKEN")
            if token:
                print("✓ HF_TOKEN found. Using authenticated client.")
            else:
                print("⚠ No HF_TOKEN found. Rate limits may apply.")
            self.client = InferenceClient(model=self.hf_model, token=token)

    def check_ollama_running(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def check_ollama_installed(self) -> bool:
        """
        Check if Ollama is installed.
        
        Returns:
            True if installed, raises error otherwise
        """
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Ollama is installed: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            self._print_ollama_install_instructions()
            raise RuntimeError("Ollama is not installed. Please install it first.")
        except Exception as e:
            print(f"⚠ Warning: Could not verify Ollama installation: {str(e)}")
        
        return False
    
    def _print_ollama_install_instructions(self):
        """Print installation instructions for Ollama."""
        print("\n" + "="*60)
        print("OLLAMA NOT FOUND")
        print("="*60)
        print("\nPlease install Ollama first:")
        print("\n1. Visit: https://ollama.ai/download")
        print("2. Download and install Ollama for Windows")
        print("3. Restart your terminal")
        print("4. Run: ollama --version  (to verify installation)")
        print("\n" + "="*60 + "\n")
    
    def check_model_exists(self) -> bool:
        """
        Check if the model is already downloaded.
        
        Returns:
            True if model exists, False otherwise
        """
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return self.model in result.stdout
        except Exception as e:
            print(f"⚠ Warning: Could not check model list: {str(e)}")
            return False
    
    def download_model(self) -> bool:
        """
        Download the LLM model via Ollama (Local only).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_local:
            print("ℹ Cloud mode: Skipping local model download.")
            return True

        if self.check_model_exists():
            print(f"✓ Model '{self.model}' already downloaded")
            return True
        
        print(f"\n📥 Downloading model '{self.model}'...")
        print("⏳ This will take a few minutes (approximately 2GB download)...")
        print("Please wait...\n")
        
        try:
            result = subprocess.run(
                ['ollama', 'pull', self.model],
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"\n✓ Model '{self.model}' downloaded successfully!")
                return True
            else:
                print(f"\n✗ Failed to download model")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n✗ Download timeout. Please check your internet connection.")
            return False
        except Exception as e:
            print(f"\n✗ Error downloading model: {str(e)}")
            return False
    
    def generate(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> str:
        """
        Generate response using the LLM (non-streaming).
        
        Args:
            prompt: User query/prompt
            context: Retrieved context from vector store
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Build full prompt with system instructions and context
        if context:
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nQuestion: {prompt}\n\nAnswer:"
        
        try:
            if self.is_local:
                response = ollama.generate(
                    model=self.model,
                    prompt=full_prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
                return response['response'].strip()
            else:
                # Cloud/API generation
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt}
                ]
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"\n✗ {error_msg}")
            return f"I apologize, but I encountered an error processing your request. Please try again."

    def generate_stream(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """
        Generate streaming response using the LLM.
        
        Args:
            prompt: User query/prompt
            context: Retrieved context from vector store
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of generated text
        """
        if context:
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"{self.SYSTEM_PROMPT}\n\nQuestion: {prompt}\n\nAnswer:"
        
        try:
            if self.is_local:
                stream = ollama.generate(
                    model=self.model,
                    prompt=full_prompt,
                    stream=True,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
                
                for chunk in stream:
                    if 'response' in chunk:
                        yield chunk['response']
            else:
                # Cloud/API streaming
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt}
                ]
                stream = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                    
        except Exception as e:
            print(f"Error in stream generation: {str(e)}")
            yield "I apologize, but I encountered an error. Please try again."
    
    def generate_greeting_response(self, greeting: str) -> str:
        """
        Generate a simple greeting response.
        
        Args:
            greeting: User's greeting message
            
        Returns:
            Greeting response
        """
        greeting_lower = greeting.lower()
        
        if any(word in greeting_lower for word in ['bye', 'goodbye', 'farewell']):
            return "Goodbye! Feel free to return if you have any questions about EWU or the CSE department."
        elif any(word in greeting_lower for word in ['thank', 'thanks']):
            return "You're welcome! Let me know if you have any other questions about EWU."
        else:
            return "Hello! I'm the EWU University Academic Assistant. How can I help you with information about East West University or the CSE department today?"


def setup_llm(model: str = LLMHandler.DEFAULT_MODEL) -> LLMHandler:
    """
    Setup and verify LLM is ready.
    
    Args:
        model: Name of the Ollama model to use
        
    Returns:
        LLMHandler object
    """
    llm = LLMHandler(model)
    llm.download_model()
    return llm


if __name__ == "__main__":
    # Test the LLM handler
    print("Testing LLM Handler...\n")
    
    llm = setup_llm()
    
    # Test generation
    test_prompt = "What is CSE department?"
    test_context = "The CSE department at EWU was founded in 1996."
    
    print(f"\nTest Query: {test_prompt}")
    print("Generating response...\n")
    
    response = llm.generate(test_prompt, test_context)
    print(f"Response: {response}")
