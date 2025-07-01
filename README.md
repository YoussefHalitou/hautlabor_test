# Hautlabor Chatbot Project

A healthcare chatbot for aesthetic medicine treatments using RAG (Retrieval Augmented Generation) technology.

## 🚀 Features

- **Smart Healthcare Assistant**: AI-powered chatbot specialized in aesthetic medicine
- **RAG Technology**: Retrieval Augmented Generation for accurate, context-aware responses
- **Memory**: Conversation history for better context understanding
- **Web Interface**: Clean HTML/CSS frontend for easy interaction
- **Secure API**: Environment variable-based API key management

## 📁 Project Structure

```
hautlabor_test/
├── app.py                           # Flask API server
├── improved_chatbot.py              # Main chatbot implementation
├── copy_of_bestes_model_arzt.py     # Data processing and QA system
├── index.html                       # Web interface
├── enhanced_kb_chunks.csv           # Knowledge base data
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install flask flask-cors pandas langchain langchain-openai langchain-community faiss-cpu beautifulsoup4 requests
```

### 2. Set Environment Variables

Create a `.env` file or set the environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the Application

```bash
# Start the Flask API server
python app.py

# Or run the chatbot directly
python improved_chatbot.py
```

## 💻 Usage

### Web Interface
1. Open `index.html` in your browser
2. Start asking questions about aesthetic treatments
3. The chatbot will provide detailed, context-aware answers

### API Endpoints
- `POST /api/chat` - Send a message to the chatbot
- `POST /api/reset` - Reset conversation memory
- `GET /api/health` - Health check

### Example API Request
```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Welche Behandlungen bieten Sie an?"}'
```

## 🔧 Configuration

### Chatbot Settings
- **Model**: GPT-3.5-turbo (for faster responses)
- **Memory**: 5 previous conversation exchanges
- **Temperature**: 0.1 (for consistent responses)
- **Max Tokens**: 500 (optimized for speed)

### Knowledge Base
The chatbot uses filtered, high-quality chunks from the website data:
- Treatment pages (highest priority)
- Homepage content
- High-importance content (score ≥ 15)

## 🏥 Supported Treatments

The chatbot can answer questions about 30+ aesthetic treatments including:

### Face Treatments
- CO₂-Laser, Fadenlifting, Botox, Filler
- HydraFacial, Morpheus8, Ultherapy
- Sculptra, Radiesse, Skinbooster
- And more...

### Body Treatments
- Laser Hair Removal, Lipolyse
- Body Contouring, Intimate Treatments

### Men's Treatments
- Specialized treatments for male patients

## 🔒 Security

- ✅ No hardcoded API keys
- ✅ Environment variable configuration
- ✅ Input validation and error handling
- ✅ CORS protection configured

## 📊 Performance

- **Fast Responses**: GPT-3.5-turbo for 2-3x faster responses than GPT-4
- **Optimized Retrieval**: 5 relevant chunks vs 15 for better speed
- **Memory Efficient**: Limited conversation history window

## 🧠 Technical Details

### RAG Pipeline
1. **Document Processing**: Web scraping and text cleaning
2. **Chunking**: Smart text splitting with overlaps
3. **Embeddings**: OpenAI embeddings for semantic search
4. **Vector Store**: FAISS for fast similarity search
5. **Generation**: GPT-3.5-turbo for response generation

### Memory Management
- ConversationBufferWindowMemory (last 5 exchanges)
- Automatic context preservation
- Reset functionality for new conversations

## 🚀 Deployment

### Local Development
```bash
python app.py
# Server runs on http://localhost:5001
```

### Production Considerations
- Set up proper environment variable management
- Configure HTTPS/SSL
- Implement rate limiting
- Add logging and monitoring
- Consider database for conversation storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes.

## 📞 Contact

For questions about the treatments or booking appointments:
**Phone**: +49 (0) 157 834 488 90
**Website**: https://haut-labor.de

---

**Note**: This chatbot is designed to provide information about treatments. For medical advice, please consult with Dr. med. Lara Pfahl directly.
