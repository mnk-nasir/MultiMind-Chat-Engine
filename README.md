# MultiMind-Chat-Engine```markdown
# Multi-Agent Conversation Companion

This small Python project mirrors the "Multi-Agent Conversation" n8n workflow you provided.
It is intended as a local companion or starting point so you can run and iterate outside n8n.

What it does
- Loads a set of agent configurations (names, model identifiers, system messages).
- Accepts a chat message (CLI argument) which may include @mentions of agents.
- Extracts mentions in-order and builds a call queue for agents. If no mentions are found, all agents are called in random order.
- For each agent: composes a system message (agent systemMessage + global systemMessage + small memory summary) and calls an LLM.
- Keeps a small local session memory (a JSON file per session id) and appends user/assistant entries.
- Combines and prints all agent outputs.

Files
- main.py — main script, runnable from the CLI
- config.py — loads .env and exposes keys
- requirements.txt — dependencies
- .env.example — example environment variables

Quick start
1. Copy files into a directory.
2. Create a virtual environment and install deps:
   ```
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Create a `.env` (or use the example) if you want to use real models:
   - For OpenRouter: set OPENROUTER_API_KEY
   - For OpenAI: set OPENAI_API_KEY
   Without keys the script will run in "simulation" mode and produce placeholder responses.

4. Run:
   ```
   python main.py --message "Hello @Chad and @Gemma, please summarize the user's note" --session-id demo1 --save
   ```

Agents
- The script includes default agents (Chad, Claude, Gemma) matching your workflow.
- To override: create a JSON file mapping agent names -> { name, model, systemMessage } and pass `--agents-file path/to/agents.json`.

Session memory
- Stored under memory/<session-id>.json
- The script appends user and assistant entries for simple context between runs.

Model calling details
- The script supports OpenRouter (preferred if OPENROUTER_API_KEY present) and OpenAI (if OPENAI_API_KEY present and agent model starts with "openai/"). The OpenRouter/OpenAI HTTP usage is intentionally minimal and may require adaptation for advanced settings (or for other providers).
- If no API keys are present, the script returns safe simulated responses for testing.

Customization ideas
- Add more advanced memory summarization (only the last N tokens or a vector DB).
- Parallelize agent calls if your environment and model providers support it.
- Add richer role/message structures (assistant -> system -> user).
- Integrate back with n8n via webhook to operate as a complete companion.

If you want, I can:
- Push these files to your GitHub repo (tell me owner/repo/branch and confirm repo exists),
- Convert this into an n8n custom node,
- Add support for additional providers (Anthropic / Google Gemini) with example HTTP wiring,
- Or expand the memory handling to include summarization and pruning.

```
