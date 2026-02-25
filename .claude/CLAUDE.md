# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

You may find coding standards and guidelines are stored in `.claude/rules/code`

## What are we building
We are building a better executive coach. Our executive coach *looks* like a plain ol' chatbot, but under the hood it has an 
agentic back-end...think of the kid who wins street races in a station wagon with a nitrous kit. 

The user interface is a simple chatbot, and should have a familiar UI. The user can select and revisit past conversations. 
The user can upload documents and retrieve them later.

The agentic back-end of the bot will serve to build and manage the correct context, and ensure the response is relevant.
  - An Orchestration Agent determines the user's intent, and how to respond. The Orchestration agent may invoke any or all of:
    - A Search Agent determines if any relevant information is needed from the internet, executes searches, and summarizes results
    - A RAG Agent manage past conversations and uploaded documents, retrieving relevant details
    - A Persona Agent maintains an understanding of the user as a person, for example, current goals and challenges.
  - The Orchestration Agent collects all pieces of the context and passes to the Context Management Agent.
  - A Context Management Agent looks at the current conversation and all information provided by the Orchestration Agent and determines the most relevant details to pass to the Coaching Agent.
  - The Coaching Agent constructs a response.
  - The Quality Agent reviews the response and rates it (1-5). If the Quality Agent finds the response unacceptable (3 or less), it gives feedback to the Coaching Agent, who rewrites the response.

More details about how the coach interacts with the world are found in `docs/coaching-philosophy.md`.

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

## Auto-Update Memory (MANDATORY)

**Update memory files AS YOU GO, not at the end.** When you learn something new, update immediately. Memory
files are stored in `.claude/rules/memory/*.md`.

| Trigger | Action |
|---------|--------|
| User shares a fact about themselves | → Update `profile.md` |
| User states a preference | → Update `preferences.md` |
| A decision is made | → Update `decisions.md` with date |
| Completing substantive work | → Add to `sessions.md` |

**Skip:** Quick factual questions, trivial tasks with no new info.

**DO NOT ASK. Just update the files when you learn something.**

## Git Workflow

- Always use feature branches; do not commit directly to `main`
  - Name branches descriptively: `fix/auth-timeout`, `feat/api-pagination`, `chore/ruff-fixes`
  - Keep one logical change per branch to simplify review and rollback
- Create pull requests for all changes
  - Open a draft PR early for visibility; convert to ready when complete
  - Ensure tests pass locally before marking ready for review
  - Use PRs to trigger CI/CD and enable async reviews
- Link issues
  - Before starting, reference an existing issue or create one
  - Use commit/PR messages like `Fixes #123` for auto-linking and closure
- Commit practices
  - Make atomic commits (one logical change per commit)
  - Prefer conventional commit style: `type(scope): short description`
    - Examples: `feat(eval): group OBS logs per test`, `fix(cli): handle missing API key`
  - Squash only when merging to `main`; keep granular history on the feature branch
- Practical workflow
  1. Create or reference an issue
  2. `git checkout -b feat/issue-123-description`
  3. Commit in small, logical increments
  4. `git push` and open a draft PR early
  5. Convert to ready PR when functionally complete and tests pass
  6. Merge after reviews and checks pass

## Python Tools

- use context7 mcp to check details of libraries

## Code Formatting

1. Ruff
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check .`
   - Fix: `uv run ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: split into multiple lines

2. Type Checking
  - run `pyrefly init` to start
  - run `pyrefly check` after every change and fix resultings errors
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass


## Error Resolution

1. CI Failures
   - Fix order:
     1. Formatting
     2. Type errors
     3. Linting
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns

3. Best Practices
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly