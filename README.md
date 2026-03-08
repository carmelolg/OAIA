# Generative AI Agentic Architecture

![OAIA Logo](static/logo.svg)

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/carmelolg/OAIA)

This project provides a Python library template/scaffold for organizing and generalizing the usage of Large Language Models (LLMs) and Ollama (or other providers like LangChain) in a structured, modular way.

## Overview

The architecture is designed as a library with the following key components:

- **Adapters**: Outbound adapters for interacting with LLMs.
- **Core**: Core business logic including providers, services, and integrations.
- **Commons**: Shared utilities like constants, environment variables, and math utilities.
- **Use Case**: Use case-specific logic including prompts, runners, steps, and tools.

## Structure

```
lib/
├── adapters/
│   └── outbound/
│       └── LLMExecutor.py          # Singleton executor for LLM interactions
├── commons/
│   ├── Constants.py                # Application constants
│   ├── EnvironmentVariables.py     # Environment variable management
│   └── MathUtils.py                # Mathematical utilities (e.g., cosine similarity)
├── core/
│   ├── integration/
│   │   └── http/
│   │       └── GenericHttpService.py  # Generic HTTP service with fallback
│   ├── providers/
│   │   ├── LLMProvider.py          # Abstract base class for LLM providers
│   │   ├── LLMProviderFactory.py   # Factory for provider instances
│   │   ├── OllamaProvider.py       # Ollama-specific provider implementation
│   │   ├── LiteLLMProvider.py      # LiteLLM provider (100+ backends via unified interface)
│   │   └── model/
│   │       └── LLMProviderConfiguration.py  # Configuration for providers
│   └── service/
│       └── KnowledgeService.py     # Knowledge base and similarity search
└── use_case/
    ├── integration/
    │   └── http/                   # (Empty, for future HTTP integrations)
    ├── prompts/
    │   ├── FilePromptManager.py    # Load prompts from files
    │   └── PromptManager.py        # Abstract prompt manager
    ├── runner/
    │   └── AbstractRunner.py       # Abstract runner for workflows
    ├── steps/
    │   ├── AbstractStep.py         # Abstract step in workflows
    │   └── StepResult.py           # Result of step execution
    └── tools/                      # (Empty, for future tools)
```

## Key Features

- **Modular Design**: Clean separation of concerns with adapters, core, and use case layers.
- **Provider Abstraction**: Easy to add new LLM providers by implementing the `Provider` interface.
- **Singleton Patterns**: Ensures single instances for providers and utilities.
- **Configuration Management**: Environment-based configuration with defaults.
- **Knowledge Management**: Embedding-based similarity search for knowledge bases.
- **Workflow Support**: Abstract steps and runners for building agentic workflows.
- **HTTP Integration**: Generic HTTP service with fallback to local files.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   LLM_PROVIDER=ollama
   LANGUAGE_MODEL=qwen3:1.7b
   EMBEDDING_MODEL=nomic-embed-text:latest
   THINKING_MODE=true
   # Add other variables as needed
   ```

3. For Docker usage, build and run:
   ```bash
   docker build -t OAIA .
   docker run -p 8080:8080 OAIA
   ```

## Usage

### Basic LLM Interaction

```python
from lib.adapters.outbound.LLMExecutor import LLMExecutor

executor = LLMExecutor.get_instance()
response = executor.ask("Hello, how are you?")
print(response)
```

### Knowledge Base

```python
from lib.core.service.KnowledgeService import KnowledgeService

ks = KnowledgeService()
knowledge = ks.build_knowledge(["chunk1", "chunk2", "chunk3"])
relevant = ks.get_most_relevant_chunks("query", knowledge)
```

### Custom Provider

Implement the `Provider` abstract class and register in `LLMProviderFactory`.

## LiteLLM Provider

[LiteLLM](https://www.litellm.ai/) is a Python SDK that routes requests to 100+ LLMs
(OpenAI, Anthropic, xAI, Azure OpenAI, Vertex AI, Ollama, and more) through a unified
OpenAI-compatible interface.

### Enabling LiteLLM

Set `LLM_PROVIDER=litellm` in your `.env` file and choose a model using LiteLLM's
`<provider>/<model>` format:

```
LLM_PROVIDER=litellm
LANGUAGE_MODEL=openai/gpt-4o
OPENAI_API_KEY=your-openai-key
```

### Example configurations

**OpenAI via LiteLLM**
```
LLM_PROVIDER=litellm
LANGUAGE_MODEL=openai/gpt-4o
OPENAI_API_KEY=your-openai-key
```

**Anthropic via LiteLLM**
```
LLM_PROVIDER=litellm
LANGUAGE_MODEL=anthropic/claude-3-sonnet-20240229
ANTHROPIC_API_KEY=your-anthropic-key
```

**Ollama local model via LiteLLM**
```
LLM_PROVIDER=litellm
LANGUAGE_MODEL=ollama/llama2
LITELLM_API_BASE=http://localhost:11434
```

**Azure OpenAI via LiteLLM**
```
LLM_PROVIDER=litellm
LANGUAGE_MODEL=azure/your-deployment-name
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-resource.openai.azure.com
AZURE_API_VERSION=2024-02-01
```

### Required environment variables per backend

| Backend | Environment variable(s) |
|---------|-------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| xAI | `XAI_API_KEY` |
| Azure OpenAI | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| Vertex AI | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_API_BASE` |
| HuggingFace | `HUGGINGFACE_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Ollama | `LITELLM_API_BASE` (if not `http://localhost:11434`) |

### Streaming

LiteLLM streaming works the same way as with other providers — set `chatbot_mode=True`
when calling `LLMExecutor.ask()` or `LLMExecutor.chat()`.

## Docker

The project includes a Dockerfile that sets up Ollama and pulls models automatically. The `run-docker.sh` script starts Ollama, pulls the specified models, and runs the application.

## Contributing

This is a scaffold/template project. Extend the abstract classes and add concrete implementations as needed for your use case.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.  
To view a copy of this license, visit [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).
