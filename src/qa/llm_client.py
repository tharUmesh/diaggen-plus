"""
llm_client.py
Provider-agnostic LLM client. Active provider is set in config.yaml.
Supports: Anthropic (Claude), OpenAI (GPT), Ollama (local).
API keys are read from .env — never hardcoded.
"""

from __future__ import annotations
import os
from src.utils.config_loader import get, prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROVIDER = get("llm.active_provider", "anthropic")


def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    cfg  = get("llm.anthropic")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg  = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI
    cfg    = get("llm.openai")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp   = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content


def _call_ollama(system: str, user: str) -> str:
    import ollama
    cfg  = get("llm.ollama")
    resp = ollama.chat(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options={"temperature": cfg["temperature"], "num_predict": cfg["max_tokens"]},
    )
    return resp["message"]["content"]


_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai":    _call_openai,
    "ollama":    _call_ollama,
}


def call(user_prompt: str, system_prompt: str | None = None) -> str:
    """
    Send a prompt to the configured LLM provider and return the response text.
    Falls back to the shared system prompt in prompts.yaml if none supplied.
    """
    system = system_prompt or prompts.get("system_prompt", "You are a helpful assistant.")
    fn = _DISPATCH.get(_PROVIDER)
    if fn is None:
        raise ValueError(f"Unknown LLM provider: '{_PROVIDER}'. "
                         f"Set llm.active_provider in config.yaml.")
    logger.debug(f"LLM call via provider={_PROVIDER}")
    return fn(system, user_prompt)


def format_prompt(template_key: str, **kwargs) -> str:
    """
    Retrieve a prompt template from prompts.yaml by dot-notation key
    and format it with the provided kwargs.
    Example: format_prompt("explanation.template", symptoms="...", predictions="...")
    """
    keys = template_key.split(".")
    node = prompts
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"Prompt key not found: {template_key}")
        node = node[k]
    return str(node).format(**kwargs)
