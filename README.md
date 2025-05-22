# AI Coaching Assistant

> Every hammer has the innate capacity to strike a nail. Every human mind has the innate capacity for greatness. 

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

3. Run the system.
- `python api.py` in one terminal.
- `python app.py` in another. Open a browser and navigate to the correct URL.

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
- [TODO] Journal entries
- [TODO] Document uploads

## Project Structure


```
.
├── src/
│   ├── controller.py     # Main controller class
│   ├── configurator.py   # All config settings
│   ├── agents/           # Agent implementations and io types
│   ├── services/         # Service implementations (e.g., ChromaDB)
│   ├── tools/            # Tools for agents to use (e.g., search)
│   └── utils/            # Utility functions and helpers
│
├── data/                 # Data directory
│   ├── data.md           # Doc about data
│   ├── processed/        # Store all personal files here
│   └── app_data/         # Data generated from the app. 
│
├── config/               # Config files
│
├── app.py                # Front end app
├── api.py                # API/back end   
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add appropriate license information]
