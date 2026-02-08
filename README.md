# Generative AI Agentic Architecture

![OAIA Logo](static/logo.svg)

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/yourusername/OAIA)

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

## Docker

The project includes a Dockerfile that sets up Ollama and pulls models automatically. The `run-docker.sh` script starts Ollama, pulls the specified models, and runs the application.

## Contributing

This is a scaffold/template project. Extend the abstract classes and add concrete implementations as needed for your use case.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.  
To view a copy of this license, visit [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).
