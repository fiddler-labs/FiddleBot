# FiddleBot

FiddleBot is an interactive CLI chatbot powered by OpenAI's GPT-4.1-mini model, specifically designed to provide information and assistance about Fiddler - the pioneer in AI Observability and Security.

## Features

- Interactive command-line interface for chatting with the AI
- Powered by OpenAI's GPT-4.1-mini model
- Maintains conversation history for context-aware responses
- Focused on providing information about Fiddler's AI Observability and Security platform
- Asynchronous operation for better performance

## Prerequisites

- Python 3.12 or higher
- Fiddler Access Details
- OpenAI API key
- UV (recommended for package management and running)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FiddleBot
```

2. Create a `.env` file in the project root and add your OpenAI API key:
```
FIDDLER_BASE_URL=<your fiddler url>
FIDDLER_ACCESS_TOKEN=<your fiddler access token>
OPENAI_API_KEY=<your_api_key_here>
```

## Installing UV

If you haven't installed UV yet, you can install it by following the instructions in the following link

[UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

## Setting up the project with UV

```bash
# Create a new virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies from pyproject.toml
uv pip install .
```

## Usage

Run the chatbot using UV:
```bash
uv run src/fdl_chat.py src/fdl_mcp_server.py
```

- Type your message and press Enter to chat with the AI
- Type `q`, `quit`, or `exit` to end the conversation
- Press Ctrl+C to exit at any time

Run the streamlit chatbot with
```bash
uv run streamlit run src/app.py
```

### Usage with OpenTelemetry 

#### Console Export of OTEL Spans to Console
```bash
uv run opentelemetry-instrument --traces_exporter console --service_name fiddlebot_local python src/fdl_chat.py src/fdl_mcp_server.py
```

#### Usage with Fiddler OTEL Collector

1. Login to dev VPN
2. Setup port forwarding to Fiddler's current OTEL Endpoint
   ```fkube fdl-newdev -n leeannotel port-forward svc/otel-to-kafka-collector 4318:4318```
3. Run Fiddlebot as below
```bash
uv run src/fdl_chat.py src/fdl_mcp_server.py
```

## About Fiddler

Fiddler is the pioneer in AI Observability and Security, enabling organizations to build trustworthy and responsible AI systems. The platform helps:

- Monitor performance of ML models and generative AI applications
- Protect LLM and GenAI applications with Guardrails
- Analyze model behavior to identify issues and opportunities
- Improve AI systems through actionable insights

## Architecture and Technical Details

FiddleBot is built with a modular architecture that combines several key components:

### Core Components

1. **Chat Interface (`fdl_chat.py`)**
   - Implements an asynchronous chatbot using OpenAI's GPT-4.1-mini model
   - Maintains conversation history for context-aware responses
   - Handles user input/output through a CLI interface
   - Integrates with the planning and execution system

2. **Planning and Execution System (`fdl_plan_n_solve.py`)**
   - Implements a multi-step planning and execution framework
   - Uses GPT-4.1-mini for task decomposition and planning
   - Manages tool execution through an MCP (Model Context Protocol) server
   - Handles conversation state and tool call results

3. **MCP Server (`fdl_mcp_server.py`)**
   - Implements a FastMCP server for tool execution
   - Provides a set of tools for interacting with the Fiddler platform
   - Handles authentication and API communication with Fiddler
   - Exposes tools for project, model, and alert management

### Key Features

1. **Asynchronous Operation**
   - Uses Python's `asyncio` for non-blocking I/O operations
   - Enables concurrent handling of multiple operations
   - Improves performance and responsiveness

2. **Tool Integration**
   - Implements a flexible tool system through MCP
   - Supports dynamic tool discovery and execution
   - Enables extensibility through custom tool definitions

3. **Conversation Management**
   - Maintains conversation history for context
   - Handles system prompts and user interactions
   - Manages tool call results and responses

4. **Error Handling**
   - Implements robust error handling for API calls
   - Manages connection issues and timeouts
   - Provides graceful cleanup of resources

### Data Flow

1. User input is received through the chat interface
2. The input is processed by the planning system
3. The planning system generates a sequence of steps
4. Each step is executed through the MCP server
5. Results are collected and formatted into a response
6. The response is returned to the user

### Security

- API keys and sensitive information are managed through environment variables
- Authentication is handled through Fiddler's token-based system
- All API communications are secured through HTTPS
