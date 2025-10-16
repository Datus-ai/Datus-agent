# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import tiktoken

_encoding = None


def get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def cal_task_size(count: int, step: int) -> int:
    return int(round(count / step + 0.5, 0))


def cal_gpt_tokens(text, encoding=None) -> int:
    if encoding is None:
        encoding = get_encoding()
    return len(encoding.encode(text))
