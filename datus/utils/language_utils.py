# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared helpers for the pinned response language.

Lives in ``utils`` rather than next to the agentic nodes because standalone
LLM calls outside the node stack (e.g. the visualization tool) need the same
code → name mapping without importing the node layer.
"""

from typing import Dict, Optional

LANGUAGE_NAME_MAP: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}

# The same directive restated *in the target language*. An English
# meta-instruction ("Use: Chinese") sits at the end of an English system
# prompt and then competes with dozens of turns of English schema/SQL tool
# output; smaller models drift out of it — and CJK drift lands on a
# neighbouring script (a Chinese session answered in Japanese) rather than on
# English. A sentence written in the target language survives that far better.
NATIVE_DIRECTIVE_MAP: Dict[str, str] = {
    "en": "Always answer in English, no matter what language the context or tool output is in.",
    "zh": "必须始终使用简体中文回复，无论上下文、表结构、SQL 或工具输出是什么语言。",
    "zh-cn": "必须始终使用简体中文回复，无论上下文、表结构、SQL 或工具输出是什么语言。",
    "zh-tw": "必須始終使用繁體中文回覆，無論上下文、表結構、SQL 或工具輸出是什麼語言。",
    "ja": "文脈やツールの出力が何語であっても、必ず日本語で回答してください。",
    "ko": "문맥이나 도구 출력이 어떤 언어이든 항상 한국어로 답변하세요.",
    "es": "Responde siempre en español, sea cual sea el idioma del contexto o de la salida de las herramientas.",
    "fr": "Réponds toujours en français, quelle que soit la langue du contexte ou des sorties d'outils.",
    "de": "Antworte immer auf Deutsch, unabhängig von der Sprache des Kontexts oder der Tool-Ausgaben.",
    "pt": "Responda sempre em português, seja qual for o idioma do contexto ou da saída das ferramentas.",
    "ru": "Всегда отвечай на русском языке, независимо от языка контекста и вывода инструментов.",
    "it": "Rispondi sempre in italiano, qualunque sia la lingua del contesto o dell'output degli strumenti.",
}


def resolve_language_name(code: Optional[str]) -> str:
    """Map a language code (e.g. ``"zh"``) to a human-readable name.

    Unknown codes are returned as-is so operators can plug in custom values
    without a code change.
    """
    if not code:
        return "English"
    return LANGUAGE_NAME_MAP.get(code.strip().lower(), code)


def resolve_native_directive(code: Optional[str]) -> str:
    """Return the target-language instruction for ``code``, or ``""``.

    Unknown codes have no translation to offer, so they fall back to the
    English name line alone rather than to a wrong-language sentence.
    """
    if not code:
        return ""
    return NATIVE_DIRECTIVE_MAP.get(code.strip().lower(), "")


def ensure_native_directive(section: str, code: Optional[str]) -> str:
    """Append the native-language line when ``section`` is missing it.

    ``DatusPathManager.ensure_templates`` copies the bundled templates into
    ``~/.datus/template`` with ``replace=False``, and that copy wins over the
    packaged one. Any home bootstrapped before this change therefore keeps
    rendering the old name-only section forever, so the directive is restated
    here rather than left to the template alone.
    """
    native = resolve_native_directive(code)
    if not native or not section.strip() or native in section:
        return section
    return f"{section.rstrip()}\n- {native}"


def build_fallback_directive(code: Optional[str]) -> str:
    """Minimal ``# Response Language`` section built without the template.

    Used by the callers that must not silently drop a pinned language when the
    jinja render fails; keeps the same shape (and the native sentence) as
    ``response_language_1.0.j2``.
    """
    lines = [f"# Response Language\n- Use: {resolve_language_name(code)} ({code})"]
    native = resolve_native_directive(code)
    if native:
        lines.append(f"- {native}")
    return "\n".join(lines)
