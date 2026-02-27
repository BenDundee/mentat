# Agent Structure
The core aim of this project is to show that agentic chatbots provide better chat experiences. We are building an 
executive coach. Details of how the Coach will interact with the world are found in `docs/coaching-philosophy.md`.

## Meet the Agents
The back end of the chatbot comprises a set of agents that serve to construct the correct context, a Coaching agent that 
takes that context and constructs a reply, and a QA or Feedback agent that will ensure high quality responses. There 
will also be agents that manage the Coaching Plan, the understanding of the User's persona, and the management of the 
Coaching Plan.

### Orchestration Agent
The Orchestration Agent will:
  - Receive the User's response
  - Understand the User's intent
    - If the User is asking for a simple check in, the Orchestration Agent should recognize these simple intents to 
  provide speedier responses.
  - Understand the context of the conversation
    - The User Intent may be to initiate a coaching session
    - The User may be in the middle of a coaching session, if so the Orchestration Agent should understand the phase of 
the session, and the session plan
    - The User may just need to check in, or get quick advice
  - Determine which agents to engage, if any. The Orchestration agent may engage:
    - A Web Search Agent that can construct queries against the internet
    - A RAG Agent that retrieves relevant details from documents the User has uploaded, and past conversations
    - A Persona Agent that injects details about the user into the response.
    - A Plan Management Agent that ensures the conversation is flowing in a direction that will make agreed-upon 
progress.

Once the other agents (which can run in parallel) return, the Orchestration Agent passes this information to the Context 
Management Agent.

### The Search Agent
If relevant details are needed from the internet, the Search Agent determines the relevant queries to issue. It has 
access to a Search Tool, and can retrieve and summarize information from the internet.

### The RAG Agent
During the course of the relationship, the User will have many conversations, and will upload many documents. The RAG 
Agent will have access to all documents and past conversations. It will formulate semantic queries against a vector 
data store, summarize, and return relevant information.

### The Persona Agent
The goal of the Persona Agent is to develop a long-term understanding of:
  - Who the user is
  - What they want
  - What drives them
  - Their strengths and weaknesses
  - How they interact and learn (this may influence how the Coach interacts).

The Persona Agent owns the understanding of the User, and updates this understanding over time. The preliminary 
understanding of the User's Persona is established during Onboarding.

After each conversation, the Persona Agent updates its understanding of the User, based on anything it has learned from 
the current conversation.

### The Plan Management Agent
During Onboarding, a Coaching Plan is established (see `docs/coaching-philosophy.md`). This plan contains near- and 
long-term goals agreed-upon by the Coach and Client (User). It is a living document. The Plan Management Agent is 
responsible for ensuring that the conversation is driving the Plan forward, and updating the Plan when needed.

After the conversation, the Plan Management Agent is responsible for summarizing the takeaways and action items. 

### The Context Management Agent
This is one of the most important roles on the team. In building an agentic back-end, we are sacrificing a few things:
  - We are injecting additional context from the RAG Agent, the Search Agent, the Persona Agent, and the Plan Management 
Agent, which inflates the input token count.
  - But we also know that long context inputs may bury specific and important details. 

**The Context Management Agent is responsible for finding the needles in the haystack.** The Context Management Agent 
will ensure that only the *most relevant* details are passed to the Coaching Agent.  You can think of this as a first 
pass plus second pass -- the first pass should have a high recall, ensuring all relevant information is available to the 
Context Management Agent. The Context Management Agent is the second pass, designed to have high accuracy.

The Context Management Agent will rank all information it receives, and give the Coaching Agent an outline of a 
response:
  - It will truncate the conversation history, if necessary
  - It will prioritize the information found from the RAG and Search Agents
  - If a Coaching Session is ongoing, it will help the Coaching Agent understand where they are in the conversation, 
and how it may move the session forward, while accomplishing the goals set out in the plan.
  - It will use information about the User's persona to offer advice on the tone and direction of the response, and to 
call out any long term trends or tendencies that have been noted.

The Coaching Agent's success or failure depends on the Context Management Agent doing its job.

### The Coaching Agent
The Coaching Agent is responsible for taking the context and constructing the appropriate response. The response should 
be tailored to the User, personalized based on their past interactions. If a session is in progress, the Coaching Agent 
should ensure that it's moving forward in a productive direction, towards the session goals as agreed with the User.

### The Feedback Agent
Once the Coaching Agent constructs a response, the Feedback Agent reviews the relevant context and grades the response 
on a scale of 1-5. The Feedback Agent will consider the following dimensions in its rating:
  - *Sanity Check.* Does it appear that there are hallucinations?
  - *Conversation Flow.* Does the response make sense in light of the conversation?
  - *Suitability for User.* Is the response suitable for the User, given what we know about them and their situation?
  - *Plan Adherence.* Is this driving the User towards their goals? Is this advancing the Coaching Plan?
  - *Voice.* Is the response consistent with the Coach's voice and role? Is the Coach challenging the Client (User) 
enough? Is the Coach forcing the Client (User) to do the work themselves, or is the Coach doing the work?

If the response is rated 3 or less, the Feedback Agent will pass specific feedback to the Coaching Agent. This feedback 
loop continues until the Feedback Agent is happy.

### The Client Management Agent
After the conversation, the Client Management agent is responsible for:
  - Summarizing and logging the current conversation, and persisting it in the long term memory.
  - Create a "Coach's Summary" that will not be passed to the client, including next steps and things to watch out for
  - Invoking the Persona Agent to update the system's understanding of the User
  - Invoking the Plan Management Agent to update the Coaching Plan, noting any progress or new goals. The Coaching Plan 
should always include a recommendation for next steps, based on the Client's (User's) progress and near-term challenges.

## Other Design Considerations

### Agent Memory
There are three types of "memory" to which the Agents all have access:
  - *Intra-turn memory.* This is the "context". The Context Management Agent will have the most complete understanding 
of this.
  - *Inter-turn memory.* The Agents will have access to a scratchpad that will allow them to add specific notes and 
hints that persist between conversation turns. For example, if the User is seeking to initiate a coaching session, and 
the coach decides that a particular framework will be used (GROW, for example, see `docs/coaching-philosophy.md`). The 
Coach would add an outline of the expected coaching session, so that it may return to this in subsequent conversation 
turns. This will ensure that the Coaching Agent can drive the conversation forward.
  - *Long-term memory*. Past conversations and any documents uploaded by the User form the long-term memory. This is a 
crucial part of the system, as the true value in executive coaching lies in the lateral nature of the relationship. By
understanding a person, their goals, their strengths, and their weaknesses, the Coach may spot trends that expose 
themselves over time.

### Agent Configurability
All specifics about agents should be stored separately in configuration files. Each agent should have the following 
configuration options available:
  - LLM Provider (ie, `openrouter`)
  - Model name (ie, `anthropic/claude-sonnet-4.6`)
  - LLM parameters: `top_k`, `temperature`, etc.
  - LLM System Prompt, along with any necessary substitution parameters. (In the past I've solved this with a wrapper 
around (one of) the `langchain` Prompt object(s).)
  - LLM Prompt Parameters (if they're known before run-time)

Note that API keys should be stored separately in the `.env` files.

Part of the value of this project is the ability to curate which contexts are sent into which LLMs, maximizing response 
accuracy and minimizing input token spend. This will allow us to trade off between improvement in accuracy vs. token 
spend.

#### LLM Providers
There should be a registry for LLM providers that will map static strings to details of the provider. Within the code,
these LLM Providers can be initialized as an `LLMProvider` class or `llm_provider` factory function. The LLM Provider 
registry should provide a way to map LLM Provider Name to some class or object that can issue calls into the LLM. The 
LLM Provider object should contain:
  - LLM Provider Name (ie, `openrouter`, allows mapping of Agent configs to LLM Provider configs)
  - LLM Provider Endpoint (optional, could be `https://openrouter.ai/api/v1/chat/completions`, for `openrouter`, for 
example)
  - LLM Secret, and a way to map the API keys in `.env`
  - LLM Connection Details, if necessary.

Adding a new LLM provider should involve editing the LLM Provider Registry, and adding the appropriate secret in `.env`. 
It may involve writing a new class.
