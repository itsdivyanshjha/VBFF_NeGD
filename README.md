# Voice Form Assistant

A voice-based form filling assistant for Indian government portals. Users can fill web forms by speaking, with automatic speech-to-text, intelligent field extraction, and text-to-speech confirmations.

## Features

- **Voice Input**: Record audio via microphone with real-time waveform visualization
- **Speech-to-Text**: Transcription using OpenAI Whisper (supports English and Hindi)
- **LLM Extraction**: Intelligent field value extraction using OpenRouter-compatible LLMs
- **Text-to-Speech**: Voice responses using Microsoft Edge TTS
- **Field Validation**: Built-in validators for Aadhaar, PAN, mobile, email, PIN code
- **Session Management**: Redis-backed conversation state
- **Easy Integration**: Single script tag to add to any webpage

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Browser   │◄───►│  FastAPI Backend │◄───►│     Redis       │
│   (Widget.js)   │ WS  │                  │     │   (Sessions)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
              ┌────────────────┼────────────────┐
              │                │                │
         ┌────┴────┐     ┌─────┴─────┐    ┌────┴────┐
         │ Whisper │     │ OpenRouter │    │Edge TTS │
         │  (STT)  │     │   (LLM)    │    │  (TTS)  │
         └─────────┘     └───────────┘    └─────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (or Docker)
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))
- FFmpeg (for audio processing)

### Installation

1. **Clone and setup**
```bash
cd voice-form-assistant/backend
cp .env.example .env
```

2. **Configure environment**
Edit `.env` and add your OpenRouter API key:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Redis**
Using Docker:
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

Or using docker-compose from root:
```bash
docker-compose up -d redis
```

5. **Run the server**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Open the sample form**
Open `sample-form/index.html` in your browser, or serve it:
```bash
cd ../sample-form
python -m http.server 8080
# Then open http://localhost:8080
```

### Using Docker Compose (Full Stack)

```bash
# From root directory
docker-compose up -d

# View logs
docker-compose logs -f backend
```

## Usage

1. Open any webpage with a form (or use the sample form)
2. Click the microphone button (bottom-right corner)
3. Click "Record" and say "start" or just start speaking
4. The assistant will guide you through each field
5. Confirm each value with "Yes" or click the Yes/No buttons
6. Form fields will be automatically filled

### Integration

Add this single line to any webpage:
```html
<script src="http://localhost:8000/static/embed.js"></script>
```

## Configuration

### Whisper Models

| Model  | Size   | Speed | Accuracy | VRAM  |
|--------|--------|-------|----------|-------|
| tiny   | 39 MB  | Fast  | Good     | ~1 GB |
| base   | 74 MB  | Fast  | Better   | ~1 GB |
| small  | 244 MB | Med   | Good     | ~2 GB |
| medium | 769 MB | Slow  | Great    | ~5 GB |
| large  | 1550 MB| Slow  | Best     | ~10 GB|

### OpenRouter Models

Recommended models:
- `meta-llama/llama-3.1-8b-instruct` - Fast, good for extraction
- `anthropic/claude-3-haiku` - Fast, high quality
- `google/gemini-pro` - Good alternative

### TTS Voices

English (Indian):
- `en-IN-NeerjaNeural` (Female)
- `en-IN-PrabhatNeural` (Male)

Hindi:
- `hi-IN-SwaraNeural` (Female)
- `hi-IN-MadhurNeural` (Male)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with service status |
| `/ready` | GET | Readiness check |
| `/warmup` | POST | Pre-load models |
| `/voices` | GET | List available TTS voices |
| `/ws` | WebSocket | Main voice communication |

### WebSocket Protocol

**Client → Server:**
```json
{"type": "init", "formSchema": {...}}
{"type": "audio_chunk", "sessionId": "...", "audio": "base64..."}
{"type": "audio_end", "sessionId": "..."}
{"type": "user_confirmation", "sessionId": "...", "confirmed": true}
```

**Server → Client:**
```json
{"type": "greeting", "text": "...", "audio": "base64...", "sessionId": "..."}
{"type": "ask_field", "text": "...", "audio": "...", "fieldId": "..."}
{"type": "confirmation_request", "text": "...", "value": "..."}
{"type": "fill_field", "fieldId": "...", "value": "..."}
{"type": "completion", "text": "...", "filledFields": {...}}
```

## Project Structure

```
voice-form-assistant/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Settings from .env
│   │   ├── api/
│   │   │   ├── websocket.py     # WebSocket handler
│   │   │   └── http.py          # HTTP endpoints
│   │   ├── services/
│   │   │   ├── whisper_service.py    # STT
│   │   │   ├── openrouter_client.py  # LLM
│   │   │   ├── tts_service.py        # TTS
│   │   │   ├── audio_processor.py    # Audio utils
│   │   │   ├── session_manager.py    # Redis sessions
│   │   │   └── validators.py         # Field validation
│   │   └── static/
│   │       ├── embed.js         # Loader script
│   │       ├── widget.js        # Main widget
│   │       └── widget.css       # Widget styles
│   ├── requirements.txt
│   ├── .env.example
│   ├── Dockerfile
│   └── README.md
├── sample-form/
│   └── index.html               # Test form
└── docker-compose.yml
```

## Troubleshooting

### Common Issues

**1. WebSocket connection failed**
- Ensure the backend is running on port 8000
- Check CORS settings in `.env`
- Try opening sample form via HTTP server (not file://)

**2. Audio not recording**
- Allow microphone access in browser
- Ensure HTTPS or localhost (required for getUserMedia)

**3. Whisper model download slow**
- Models are cached in `~/.cache/whisper`
- First run will download the model

**4. Redis connection error**
- Start Redis: `docker run -d -p 6379:6379 redis:7-alpine`
- Check REDIS_HOST in `.env`

**5. OpenRouter errors**
- Verify API key is correct
- Check model name is valid
- Ensure account has credits

### Debug Mode

Enable debug logging:
```bash
DEBUG=true LOG_LEVEL=DEBUG python -m uvicorn app.main:app --reload
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
# Format
black app/
isort app/

# Lint
flake8 app/
mypy app/
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [OpenRouter](https://openrouter.ai/) - LLM API
- [Edge-TTS](https://github.com/rany2/edge-tts) - Text-to-speech
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
