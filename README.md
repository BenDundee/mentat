# Mentat

An executive coaching AI assistant built with a modular architecture and powered by modern language models.

> "Every hammer has the innate capacity to strike a nail. Every human mind has the innate capacity for greatness."

## Project Overview

Mentat is an AI-powered executive coaching system designed to provide personalized guidance, feedback, and support. The system uses a modular architecture with a focus on intent detection and specialized response workflows.

## Architecture

The system follows a modular agent-based architecture:

- **Controller**: Orchestrates the conversation flow, detects user intent, and routes requests to appropriate specialized agents.
- **Intent Detector**: Analyzes user messages to determine their primary intent.
- **Conversation State**: Manages and tracks the conversation state throughout interactions.
- **Specialized Agents**: Handle specific types of interactions (currently includes SimpleResponder).

## Core Components

- `api/`: Main package containing all system logic
  - `agency/`: Contains all agent implementations
    - `_agent.py`: Base abstract class for all agents
    - `intent_detector.py`: Detects user intent from messages
    - `simple_responder.py`: Handles general conversation
  - `interfaces/`: Contains data models and schemas
  - `services/`: External service integrations
  - `controller.py`: Main workflow orchestrator
  - `api_configurator.py`: Configuration management

- `prompts/`: Contains prompt templates for the LLM
- `configs/`: Configuration files for the system
- `app.py`: FastAPI web application entry point
- `run.py`: CLI entry point for the application

## Technology Stack

- **Python 3.13**: Core programming language
- **LangChain**: Framework for LLM application development
- **Pydantic**: Data validation and settings management
- **FastAPI**: Web API framework (if applicable)

## Development

This project uses a virtual environment for dependency management. Key dependencies include:
- click
- jinja2
- kubernetes
- numpy
- pandas
- protobuf
- requests
- sqlalchemy
- langchain

## Current Status

The project is under active development with the following features implemented:
- Basic intent detection system
- Modular agent architecture
- Simple conversation handling

Upcoming features:
- Enhanced conversation history management
- Specialized coaching workflows
- Feedback and goal-setting capabilities