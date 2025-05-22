# AI Coaching Assistant

An intelligent coaching system that leverages multiple AI agents to provide personalized coaching responses and support personal development.

## Overview

This project implements a sophisticated AI coaching system that uses multiple specialized agents to process user inputs, search for relevant information, manage context, and generate appropriate coaching responses. The system includes features for persona management, journaling support, and document processing.

## Features

- 🤖 Multi-agent architecture for specialized processing
- 🎯 Intelligent preprocessing and intent detection
- 🔍 Integration with search capabilities for enhanced responses
- 💾 ChromaDB-based document storage and retrieval
- 👤 Dynamic persona management
- 📝 Journal entry processing
- 📄 Document upload handling
- ✅ Quality assurance through feedback loop

## System Architecture

The system consists of several key components:

- **Controller**: Central orchestrator that manages the flow of information between agents
- **Agents**:
  - Preprocessing Agent: Determines user intent
  - Search Agent: Formulates search queries
  - Query Agent: Manages database interactions
  - Persona Agent: Handles persona management
  - Context Manager: Aggregates information from various sources
  - Coaching Agent: Generates coaching responses
  - Feedback Agent: Ensures response quality

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the system:
- Update configuration settings in the appropriate config files
- Ensure ChromaDB is properly set up

3. Run the system:
```bash
python src/controller.py
```

## Usage

The system accepts input in the following format:

```python
{
    "role": "user",
    "content": "Your message here"
}
```

The system can handle different types of interactions:
- Coaching requests
- Journal entries
- Document uploads

## Project Structure

```
.
├── src/
│   ├── controller.py      # Main controller class
│   ├── agents/           # Agent implementations
│   ├── services/         # Service implementations (e.g., ChromaDB)
│   └── utils/           # Utility functions and helpers
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add appropriate license information]
