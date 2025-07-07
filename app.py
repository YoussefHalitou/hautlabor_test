import os
import logging
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from improved_chatbot import ImprovedChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Initialize your chatbot
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

try:
    logger.info("Initializing chatbot")
    chatbot = ImprovedChatbot(API_KEY)
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        logger.info("Received chat request")
        data = request.json
        
        if not data:
            logger.warning("No JSON data in request")
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get('message', '')
        
        if not question:
            logger.warning("Empty message in request")
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"Processing question: {question[:50]}...")
        
        # Get response from your RAG system
        response = chatbot.chat(question)
        
        logger.info("Successfully generated response")
        return jsonify({
            "message": response,
            "status": "success"
        })
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            "error": error_msg,
            "status": "error"
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        logger.info("Resetting chatbot memory")
        chatbot.reset_memory()
        logger.info("Chatbot memory reset successfully")
        return jsonify({
            "status": "success",
            "message": "Memory reset successfully"
        })
    except Exception as e:
        error_msg = f"Error resetting memory: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": error_msg
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running"""
    return jsonify({
        "status": "healthy",
        "service": "Hautlabor Chatbot API"
    })

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"Route not found: {request.path}")
    return jsonify({
        "status": "error",
        "error": "Route not found"
    }), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({
        "status": "error",
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    logger.info("Starting Hautlabor Chatbot API server on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
