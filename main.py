#!/usr/bin/env python3
"""
main.py

Lightweight companion script that mirrors the "Multi-Agent Conversation" n8n workflow:
- Defines global settings and agent configs (in-code or via JSON file)
- Accepts a chat message, extracts @mentions
- Orders agents by mention order (or randomizes all agents if none mentioned)
- For each agent: composes a system + user prompt, calls an LLM (OpenRouter/OpenAI) or simulates response
- Keeps a simple local memory buffer (per-session) and appends assistant outputs
- Combines agent outputs and prints/saves them

This is intentionally small and dependency-light. It will attempt real API calls if you provide API keys
(OPENROUTER_API_KEY or OPENAI_API_KEY). Without keys it runs a safe simulated flow.

Usage examples:
  python main.py --message "Hey @Chad and @Gemma, summarize the user note" --session-id session123 --save
  python main.py --message "Hello everyone"                # will call all agents in random order (simulation if no keys)

Files:
- config.py (loads .env)
- requirements.txt (requests, python-dotenv)
"""
from __future__ import annotations
import re
import json
import random
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from datetime import datetime
from pprint import pprint

# local config (loads .env when present)
import config

ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "memory"
MEMORY_DIR.mkdir(exist_ok=True)


# Default agents (mirrors the n8n node). You can override by supplying --agents-file JSON.
DEFAULT_AGENTS = {
    "Chad": {
        "name": "Chad",
        "model": "openai/gpt-4o",
        "systemMessage": "You are a helpful Assistant. You are eccentric and creative, and try to take discussions into unexpected territory."
    },
    "Claude": {
        "name": "Claude",
        "model": "anthropic/claude-3.7-sonnet",
        "systemMessage": "You are logical and practical."
    },
    "Gemma": {
        "name": "Gemma",
        "model": "google/gemini-2.0-flash-lite-001",
        "systemMessage": "You are super friendly and love to debate."
    }
}

# Default global settings (mirrors the n8n node)
DEFAULT_GLOBAL = {
    "user": {
        "name": "Jon",
        "location": "Melbourne, Australia",
        "notes": "Jon likes a casual, informal conversation style."
    },
    "global": {
        "systemMessage": "Don't overdo the helpful, agreeable approach."
    }
}


def load_agents_from_file(path: str | None) -> Dict[str, Dict[str, Any]]:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Agents file not found: {path}")
        return json.loads(p.read_text(encoding="utf-8"))
    return DEFAULT_AGENTS


def extract_mentions(text: str, agent_names: List[str]) -> List[str]:
    """
    Extract @mentions for agent names and return canonical agent names in appearance order.
    Case-insensitive. If none found, returns an empty list.
    """
    if not text or not agent_names:
        return []

    # Build alternation pattern escaping names and matching word boundary after name
    escaped = [re.escape(n) for n in agent_names]
    pattern = re.compile(r"\B@(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    found = []
    for m in pattern.finditer(text):
        name_mentioned = m.group(1)
        # map to canonical (case-insensitive match)
        canonical = next((n for n in agent_names if n.lower() == name_mentioned.lower()), name_mentioned)
        found.append((m.start(), canonical))
    # sort by index and return names
    found.sort(key=lambda x: x[0])
    return [name for _, name in found]


def build_agent_queue(agents: Dict[str, Any], mentions_order: List[str]) -> List[Dict[str, Any]]:
    if mentions_order:
        queue = []
        for name in mentions_order:
            if name in agents:
                queue.append(agents[name])
        return queue
    # no mentions -> randomize all agents
    all_agents = list(agents.values())
    random.shuffle(all_agents)
    return all_agents


def session_memory_path(session_id: str) -> Path:
    return MEMORY_DIR / f"{session_id}.json"


def load_memory(session_id: str) -> List[Dict[str, Any]]:
    p = session_memory_path(session_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def append_memory(session_id: str, entry: Dict[str, Any]) -> None:
    mem = load_memory(session_id)
    mem.append(entry)
    p = session_memory_path(session_id)
    p.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")


def call_model_openrouter(model: str, system_message: str, user_input: str) -> str:
    """
    Attempt to call OpenRouter's chat completions endpoint.
    This is best-effort and may need adjustment for your OpenRouter setup.
    """
    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    url = "https://api.openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # OpenRouter has choices[0].message.content for standard chat response
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)


def call_model_openai(model: str, system_message: str, user_input: str) -> str:
    """
    Call the OpenAI chat completions endpoint (v1/chat/completions).
    Requires OPENAI_API_KEY in env (loaded via config).
    """
    api_key = config.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model.replace("openai/", ""),  # support model names like openai/gpt-4o
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)


def call_agent(agent: Dict[str, Any], user_message: str, global_settings: Dict[str, Any], session_id: str) -> str:
    """
    Compose messages (system + user) and call an LLM (OpenRouter/OpenAI) if configured.
    Falls back to a simulated response if no API keys available.
    """
    system_message = "\n\n".join([
        f"Agent name: {agent.get('name')}",
        agent.get("systemMessage", ""),
        global_settings.get("global", {}).get("systemMessage", ""),
        f"User details: {global_settings.get('user', {})}",
        f"Current time (UTC): {datetime.utcnow().isoformat()}",
    ])

    # optionally include recent memory summary (very small)
    mem = load_memory(session_id)
    if mem:
        recent = mem[-3:]
        summary = "\n\n".join([f"{m['role']}: {m['text']}" for m in recent])
        system_message += "\n\nRecent conversation (most recent first):\n" + summary

    model_id = agent.get("model", "")
    try:
        # Prefer OpenRouter if key is present and model looks non-openai
        if config.OPENROUTER_API_KEY:
            # call OpenRouter
            return call_model_openrouter(model_id, system_message, user_message)
        if config.OPENAI_API_KEY and model_id.startswith("openai/"):
            return call_model_openai(model_id, system_message, user_message)
    except Exception as exc:
        # Log and continue to fallback simulation
        print(f"[warning] model call failed for agent {agent.get('name')}: {exc}")

    # Simulation fallback (safe, no external calls)
    simulated = f"(SIMULATED RESPONSE from {agent.get('name')}) System note: {agent.get('systemMessage')}\n\nUser asked: {user_message}\n\nShort reply: I understood; here's a concise answer placeholder."
    return simulated


def combine_responses(responses: List[Dict[str, Any]]) -> str:
    """
    Combine agent outputs similar to the n8n "Combine and format responses" node.
    """
    parts = []
    for r in responses:
        role = r.get("agent", "Agent")
        text = r.get("text", "")
        parts.append(f"**{role}**:\n\n{text}")
    return "\n\n---\n\n".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Agent Conversation companion for n8n workflow")
    p.add_argument("--message", "-m", required=True, help="User chat message (can contain @mentions)")
    p.add_argument("--session-id", "-s", default="default", help="Session id for memory (simple file)")
    p.add_argument("--agents-file", "-a", default=None, help="JSON file with agents dict (optional)")
    p.add_argument("--save", action="store_true", help="Save combined output to disk (outputs/)")
    p.add_argument("--no-memory", action="store_true", help="Do not write memory for this run")
    return p.parse_args()


def main():
    args = parse_args()

    # Load agents and global settings
    agents = load_agents_from_file(args.agents_file)
    global_settings = DEFAULT_GLOBAL

    # Extract agent names and mentions
    agent_names = list(agents.keys())
    mentions = extract_mentions(args.message, agent_names)
    agent_queue = build_agent_queue(agents, mentions)

    print(f"> Incoming message: {args.message}")
    if mentions:
        print("> Mentions detected (in order):", mentions)
    else:
        print("> No mentions detected — calling all agents in randomized order")

    responses = []
    for agent in agent_queue:
        name = agent.get("name", "Agent")
        print(f"\n--- Calling agent: {name} (model={agent.get('model')}) ---")
        # Determine what input the agent gets:
        # If this is first loop run for agent we may pass user message, else last assistant message — simplified here.
        user_input = args.message

        # Append a 'user' memory entry (optional small log)
        if not args.no_memory:
            append_memory(args.session_id, {"role": "user", "text": user_input, "agent": name, "ts": time.time()})

        # Call agent
        try:
            out_text = call_agent(agent, user_input, global_settings, args.session_id)
        except Exception as exc:
            out_text = f"(error calling model) {exc}"

        # Append assistant memory
        if not args.no_memory:
            append_memory(args.session_id, {"role": "assistant", "text": out_text, "agent": name, "ts": time.time()})

        responses.append({"agent": name, "text": out_text})

        print(f"Agent {name} response:\n{out_text}")

    # Combine outputs
    combined = combine_responses(responses)
    print("\n\n=== Combined Output ===\n")
    print(combined)

    # Save combined output if requested
    if args.save:
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}_multi_agent_{args.session_id}.md"
        out_path.write_text(combined, encoding="utf-8")
        print(f"\nSaved combined output to: {out_path}")

    # Print memory location for convenience
    print(f"\nSession memory: {session_memory_path(args.session_id)} (you can inspect or delete this file)")


if __name__ == "__main__":
    main()
