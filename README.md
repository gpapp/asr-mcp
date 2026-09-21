# asr-mcp

`asr-mcp` is a Python-based Model Context Protocol (MCP) server designed for advanced Audio Speech Recognition (ASR) capabilities. It provides interfaces for performing diarization (identifying different speakers) and voiceprint recognition (speaker identification) within an MCP-compliant framework.

## Features

- **ASR with Diarization**: Automatically detect and separate multiple speakers in an audio stream.
- **Voiceprint Recognition**: Identify and track specific speakers using voice embeddings.
- **MCP Compliant**: Designed to work seamlessly with MCP-capable AI agents.
- **Secure Access**: Nginx proxy with `htpasswd` authentication for protected MCP endpoints.
- **Session Persistency**: Maintains state across interactions via session management.
- **Containerized**: Fully Dockerized deployment for consistent environments.
- **Web Interface**: Includes a dashboard/GUI for monitoring and manual interaction.

## Architecture

The project follows a modular architecture:

- **FastAPI Server**: The core application providing RESTful and MCP interfaces.
- **Nginx**: Acts as a reverse proxy, handling authentication and routing.
- **Docker Compose**: Orchestrates the app and any necessary sidecars.
- **ASR Engine**: Implemented using advanced Python libraries for audio processing and speaker embedding.

### Project Structure

```text
asr-mcp/
├── asr_mcp/             # Core application logic
│   ├── api/             # API route handlers
│   ├── config/          # Configuration management
│   ├── diarization/     # Diarization logic
│   ├── speaker/         # Speaker identification/embeddings
│   ├── static/          # Static assets for GUI
│   ├── templates/       # HTML templates
│   ├── data/           # Application data (runtime)
│   └── logs/           # Application logs
├── data/               # Persistent application data
├── logs/               # Persistent application logs
├── AGENTS.md           # Developer instructions and commands
├── docker-compose.yml  # Docker orchestration
├── requirements.txt    # Python dependencies
└── .env.example        # Template for environment variables
```

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.10+ (for local development)

### Setup

1.  **Clone the repository** (not applicable here, but for others).
2.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env and set appropriate values (e.g., SESSION_SECRET)
    ```
3.  **Install Dependencies** (Local Development):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```
4.  **Start Infrastructure**:
    ```bash
    docker-compose up -d
    ```
5.  **Run Server Locally**:
    ```bash
    python asr_mcp/server.py
    ```

### Running with Docker

To build and start the containerized application:
```bash
docker-compose up -d --build
```

## Configuration

Key configuration is handled via environment variables in the `.env` file:

| Variable | Description |
|----------|-------------|
| `HTPASSWD_PATH` | Path to the htpasswd file for Nginx authentication |
| `SESSION_SECRET` | Secret key for session management |
| `DATA_DIR` | Directory for persistent application data |
| `LOG_DIR` | Directory for application logs |

## API & MCP Interface

The server exposes several interfaces:

- **MCP Interface**: Accessible via `/mcp` (requires authentication).
- **REST API**: For programmatic access to ASR and speaker features.
- **GUI**: Web-based dashboard for monitoring and manual interaction.

---

*Built with Python, FastAPI, and MCP.*
