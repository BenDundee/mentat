#!/bin/bash
# Derive the project root from this script's location (.claude/hooks/ → project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEMORY_DIR="${PROJECT_ROOT}/.claude/rules/memory"

CONTEXT=$(cat)

STRONG_PATTERNS="fixed|workaround|gotcha|that's wrong|check again|we already|should have|discovered|realized|turns out"
WEAK_PATTERNS="error|bug|issue|problem|fail"

MEMORY_REMINDER="IMPORTANT: Save any session memories to ${MEMORY_DIR}/*.md (NOT to ~/.claude). Files: profile.md, preferences.md, decisions.md, sessions.md."

if echo "$CONTEXT" | grep -qiE "$STRONG_PATTERNS"; then
    printf '{"decision":"approve","systemMessage":"This session involved fixes or discoveries. %s"}\n' "$MEMORY_REMINDER"
elif echo "$CONTEXT" | grep -qiE "$WEAK_PATTERNS"; then
    printf '{"decision":"approve","systemMessage":"If you learned something non-obvious this session, update the memory files. %s"}\n' "$MEMORY_REMINDER"
else
    printf '{"decision":"approve","systemMessage":"%s"}\n' "$MEMORY_REMINDER"
fi