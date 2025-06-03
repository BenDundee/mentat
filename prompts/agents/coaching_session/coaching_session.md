# 📚 Executive Coaching Framework for AI Assistant

This document outlines the executive coaching framework used in the AI-powered coaching assistant. It combines elements of the **CLEAR** and **GROW** coaching models to support both reflective exploration and action planning.

---

## 🌐 **High-Level Flow**

### 🎯 Coaching Journey Macro-Flow

1️⃣ **Long-Term Goals**: Defined at the beginning of the coaching relationship. Includes personal and professional development objectives.
2️⃣ **Session-Based Progress**: Each coaching session targets a specific goal or challenge linked to long-term goals.
3️⃣ **Contextual Memory**: Past reflections, feedback documents, and achievements are retrieved and integrated into sessions.

### 🧠 **Micro-Flow for a Single Session**

This is a hybrid of **CLEAR** and **GROW** models:

1️⃣ **Contract (Session Start)**

* Establish the purpose of the session.
* Confirm long-term goals and user context.
* Clarify expectations for the session.

2️⃣ **Listen (Exploration)**

* Encourage deep reflection and open expression.
* Explore emotions, beliefs, and challenges.
* Incorporate feedback documents and past sessions.

3️⃣ **Explore (Insight and Possibilities)**

* Help uncover patterns, obstacles, and possibilities.
* Link discoveries to long-term objectives.
* Use RAG (retrieval-augmented generation) for relevant context.

4️⃣ **Action Planning (GROW Integration)**

* Identify actionable next steps.
* Use GROW structure:

  * **Goal**: What do you want to achieve next?
  * **Reality**: Where are you now?
  * **Options**: What can you try?
  * **Will**: What will you commit to?

5️⃣ **Review (Closure)**

* Reflect on insights gained during the session.
* Offer journaling or behavioral assignments.
* Reinforce connection to long-term goals.

---

## 🏗️ **Prompt Structure by Stage**

For each stage, the system uses LangChain `ChatPromptTemplate` objects with structured placeholders:

* **Persona Summary**: User’s core traits, preferences, and communication style.
* **Active Goals**: Short- and long-term goals tracked by the system.
* **Retrieved Documents**: Contextual information from uploaded documents and prior interactions.
* **Session Goal**: Specific focus of the current session.
* **Recent Messages**: Chat history from the session.

Prompts include stage-specific **instructions**, clear **behavioral guidelines**, and **transition cues** to suggest moving to the next stage when appropriate.

---

## 🔑 **Key Principles**

* **State Tracking**: Maintained externally to control transitions.
* **Prompt Engineering**: Ensures clarity, empathy, and adaptability.
* **Dynamic Context**: Uses RAG and persona modeling for personalized responses.
* **Flow Modularity**: Supports distinct flows for reflective coaching and tactical requests.

---

## 🚀 **Next Steps**

* Expand prompt templates for additional flows (e.g., QuickAdvice, ResumeReview).
* Develop orchestration logic (LangGraph) to switch between coaching stages dynamically.
* Integrate token budgeting and summarization for long sessions.
* Build client APIs to trigger coaching sessions via Gradio UI.

This framework ensures a **cohesive, structured coaching journey** while allowing flexibility for diverse user needs.
