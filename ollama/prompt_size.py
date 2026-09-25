"""Measure a prompt before it is dispatched, and refuse one that can take the host down.

Ticket #837. On 2026-09-25 slice kernel-panicked mid-prefill:

    panic: "completeMemory() prepare count underflow" @IOGPUMemory.cpp:492
    wired ~53 GB of 64, free 62 MB, memoryPressure false
    exo, last line: Prefill progress: 47104/108753 tokens (179.6 tok/s)

A 108,753-token prompt arrived through this gateway, exo began prefilling it, and
macOS's GPU driver panicked rather than failing the allocation. The driver bug is
Apple's; reaching it from userspace with a large enough prompt is ours, and it is
reachable by any client that can POST to :8800.

So the gateway measures the prompt first and declines above a limit. Two things
this module deliberately is not:

- **It is not a context-window check.** GLM-4.7-Flash advertises 202,752 tokens
  and exo will honestly try to serve them. The limit here is a property of the
  two boxes the pool is placed on, not of the model, and it moves when the
  placement moves. Every refusal says so, because a caller told only "too long"
  will reasonably go looking for a model with a bigger window.
- **It is not a memory accounting.** It counts tokens. The mapping from tokens to
  wired bytes was measured (`config.EXO_MAX_PROMPT_TOKENS` carries the numbers)
  and is quadratic, not linear, which is exactly why a limit set by intuition
  lands in the wrong place.

Counting is exact when the resident model's `tokenizer.json` is on disk — the same
file exo tokenizes with — and falls back to a conservative chars/token estimate
when it is not. The fallback overestimates on ordinary prose; that is the safe
direction, and the refusal says which method produced the number.
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# Populated lazily, keyed by model id:
# {model_id: (tokenizer_or_None, longest_token_chars, context_length_or_None)}
_CACHE: dict = {}


class PromptTooLarge(Exception):
    """A prompt that this host cannot be asked to prefill.

    Carries the numbers a caller needs to act — what it sent, what the limit is,
    and how the count was arrived at — for the same reason
    `capacity.InsufficientCapacity` does: a refusal is an answer, and an answer a
    client cannot read is a fault by another name.
    """

    def __init__(self, tokens: int, limit: int, method: str, completion_budget: int,
                 resource_id: str, model: str, context_length: Optional[int] = None):
        self.tokens = tokens
        self.limit = limit
        self.method = method
        self.completion_budget = completion_budget
        self.resource_id = resource_id
        self.model = model
        self.context_length = context_length
        super().__init__(
            f"prompt is {tokens} tokens ({method}), over this host's limit of {limit} "
            f"(a {completion_budget}-token answer would be added to the same cache)"
        )

    def as_dict(self) -> dict:
        d = {
            "error": "prompt_too_large",
            "message": (
                f"this prompt measures {self.tokens} tokens ({self.method}), over this "
                f"host's limit of {self.limit}. The answer would add up to "
                f"{self.completion_budget} more tokens to the same KV cache; the limit "
                f"is set low enough to leave room for that. Send less, or split the "
                f"work across turns."
            ),
            "limit_tokens": self.limit,
            "measured_tokens": self.tokens,
            "completion_budget_tokens": self.completion_budget,
            "counted_with": self.method,
            "resource_id": self.resource_id,
            "model": self.model,
            # The note is part of the contract, not decoration: a caller that reads
            # only "too long" will go looking for a longer-context model, and there
            # is one — this same model, which advertises 202,752 tokens.
            "note": (
                "THIS LIMIT PROTECTS THE HOST, NOT THE MODEL'S CONTEXT WINDOW. "
                f"{self.model} advertises a "
                f"{self.context_length or 'much larger'}-token context and exo will "
                "try to serve it. The constraint is the memory the two machines "
                "behind this pool have: a prefill of this size drives Metal's wired "
                "allocation to the ceiling, and on 2026-09-25 that kernel-panicked "
                "the host rather than failing the allocation (#837). A bigger-window "
                "model would not help; a shorter prompt will."
            ),
        }
        if self.context_length:
            d["model_context_length"] = self.context_length
        return d


def _load(model_id: str, models_dir: str):
    """Return (tokenizer|None, longest_token_chars, context_length|None), cached.

    Missing `tokenizers`, a missing file and a corrupt file all land on the same
    answer — no tokenizer — because the caller's fallback is the same in all three
    and a gateway that fails to serve because it could not *count* would be a
    worse outcome than an estimate.

    The context length is read from the same directory, off the model's own
    `config.json`, rather than from `/state`: the refusal needs it to say "this
    is not your context window" and that sentence should not cost a round trip
    to the pool we are in the middle of declining to touch.
    """
    if model_id in _CACHE:
        return _CACHE[model_id]
    home = os.path.join(models_dir, model_id.replace("/", "--"))
    path = os.path.join(home, "tokenizer.json")
    tok = None
    longest = 0
    ctx = None
    try:
        with open(os.path.join(home, "config.json")) as fh:
            cfg = json.load(fh)
        ctx = cfg.get("max_position_embeddings")
    except Exception:
        pass
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415 — optional dependency
        tok = Tokenizer.from_file(path)
        # The longest single token bounds how many characters one token can stand
        # for, which is what makes the cheap pre-check below sound. Measured for
        # GLM-4.7-Flash-6bit: 512 characters (a run of spaces).
        longest = max((len(t) for t in tok.get_vocab()), default=0)
        log.info(f"prompt-size: exact counting for {model_id} "
                 f"(vocab {tok.get_vocab_size()}, longest token {longest} chars)")
    except ImportError:
        log.warning("prompt-size: `tokenizers` not installed; counting by estimate")
    except Exception as e:
        log.warning(f"prompt-size: no tokenizer for {model_id} at {path} ({e}); "
                    f"counting by estimate")
    _CACHE[model_id] = (tok, longest, ctx)
    return _CACHE[model_id]


def _text_of(messages: list) -> str:
    """Every character the model will see, flattened.

    Content may be a string or OpenAI's list-of-parts; tool calls and names ride
    along in the same prompt, so they are counted too rather than ignored.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(str(m))
            continue
        for key in ("role", "name"):
            if m.get(key):
                out.append(str(m[key]))
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    out.append(str(part.get("text") or part.get("image_url") or part))
                else:
                    out.append(str(part))
        elif content is not None:
            out.append(str(content))
        for extra in ("tool_calls", "tool_call_id", "function_call"):
            if m.get(extra):
                out.append(json.dumps(m[extra]))
    return "\n".join(out)


def count(messages: list, model_id: str, models_dir: str, chars_per_token: float,
          per_message_overhead: int = 8, fixed_overhead: int = 8) -> tuple:
    """(tokens, method) for a message list.

    The overheads cover the chat template, which wraps every message in role
    markers this function never sees. Measured against exo's own `usage` on this
    model: a 4096-token user message was reported as 4102 prompt tokens, 2048 as
    2053, 8192 as 8197 — so the true overhead is 5-6 tokens for a single message.
    8 per message plus 8 fixed is deliberately above that, since the whole point
    is to be wrong in the direction that refuses.
    """
    text = _text_of(messages)
    tok, longest, _ctx = _load(model_id, models_dir)
    n_msgs = len(messages or [])
    overhead = fixed_overhead + per_message_overhead * n_msgs
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False).ids) + overhead, "tokenizer"
    # No tokenizer: divide by the most token-dense ratio we measured rather than
    # an average one. Measured on this tokenizer, chars per token: prose 4.50,
    # python 3.86, JSON 3.38, log lines 2.76, CJK 2.00, base64-like 1.50. A
    # ratio of 1.5 therefore over-counts ordinary prose by about 3x. That is the
    # cost of not knowing, and it is paid only when the tokenizer is missing.
    return int(len(text) / chars_per_token) + overhead, f"estimate at {chars_per_token} chars/token"


def check(messages: list, model_id: str, limit: int, completion_budget: int,
          models_dir: str, chars_per_token: float, resource_id: str,
          context_length: Optional[int] = None) -> tuple:
    """Raise PromptTooLarge if this request should not be dispatched. Else (tokens, method).

    The limit is on the **prompt**, not on prompt plus answer, because the two
    costs are not the same shape. Prefill was measured to grow faster than
    linearly with prompt length (`config.EXO_MAX_PROMPT_TOKENS` carries the
    numbers); decode then adds to the same cache one token at a time, linearly.
    Bounding those with one addition would make the usable prompt shrink
    whenever a caller asks for a longer answer, which is not what the memory
    does.

    The answer is bounded separately and already: the serving path caps
    `max_tokens` at `EXO_MAX_TOKENS`, so the worst case here is
    `EXO_MAX_PROMPT_TOKENS + EXO_MAX_TOKENS` tokens of cache, and the limit was
    chosen with that total in mind. The budget is carried into the refusal so a
    caller can see it was accounted for, not to be added to the measurement.
    """
    tok, longest, declared_ctx = _load(model_id, models_dir)
    context_length = context_length or declared_ctx
    chars = sum(len(str(m)) for m in (messages or []))
    # Refuse an absurd body without tokenizing it. Sound because one token can
    # stand for at most `longest` characters, so anything longer than
    # limit*longest cannot possibly come in under the limit. Only a guard against
    # spending CPU on a body that is already hopeless — it is not the limit.
    if longest and chars > (limit * longest):
        raise PromptTooLarge(chars // max(longest, 1), limit, "unmeasured: body far over any possible limit",
                             completion_budget, resource_id, model_id, context_length)
    tokens, method = count(messages, model_id, models_dir, chars_per_token)
    if tokens > limit:
        raise PromptTooLarge(tokens, limit, method, completion_budget,
                             resource_id, model_id, context_length)
    return tokens, method
