"""
Flask Server for EWU UniGuide Chatbot
Provides REST API for the custom web interface.
"""

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import load_vector_store
from src.llm_handler import LLMHandler
from src.rag_pipeline import create_rag_pipeline

app = Flask(__name__, static_folder='static', static_url_path='')

# Configure CORS to allow requests from any origin (for InfinityFree deployment)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize RAG pipeline
print("🔄 Initializing EWU UniGuide...")
try:
    vector_store = load_vector_store()
    print("✓ Vector store loaded")
    
    llm_handler = LLMHandler()
    print("✓ LLM handler initialized")
    
    rag_pipeline = create_rag_pipeline(vector_store, llm_handler, top_k=3)
    print("✓ RAG pipeline created")
    print("✅ System ready!\n")
except Exception as e:
    print(f"❌ Initialization failed: {str(e)}")
    print("Please run 'python setup.py' first!")
    sys.exit(1)


@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory('static', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests (non-streaming)."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Process query through RAG pipeline
        result = rag_pipeline.query(user_message)
        response = result.get('response', 'Sorry, I could not generate a response.')
        
        return jsonify({
            'response': response,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Handle chat requests with streaming response."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
            
        def generate():
            for chunk in rag_pipeline.query_stream(user_message):
                # Send chunk as SSE (Server-Sent Event)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
            
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'chatbot': 'EWU UniGuide'})


if __name__ == '__main__':
    print("=" * 70)
    print("  🎓 EWU UniGuide - Web Server")
    print("=" * 70)
    print("\n  🌐 Open your browser to: http://localhost:5000")
    print("\n" + "=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
