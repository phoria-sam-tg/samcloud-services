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

Counting goes through the model's own chat template and tokenizer — the same two
files exo renders and tokenizes with — so the number means what the measurements
behind the limit mean: tokens the model is actually handed. Tools count, because
this template renders every definition inline. Missing template or tokenizer fall
back to the serialised request and then to a chars/token estimate; the refusal
always says which method produced its number.

THE FALLBACKS DO NOT ALL OVER-COUNT, which this docstring used to claim and
which is the wrong way round for the requests that matter. Measured on wafer
2026-09-29 against GLM-4.7-Flash-6bit, one user message and a varying number of
tools, `serialised` minus `chat template`:

    tools      0     1     4     8    16    32
    delta    +30   -68   -68   -68   -68   -68

A bare prompt over-counts by the serialised form's keys and quotes. The moment
tools are present it UNDER-counts by a constant 68 tokens — the template's own
wrapping, which the serialised form never sees — so the fallback admits prompts
the template would refuse. Bounded and small against a 12,288-token limit, and
so not a reason to panic, but it is a floor on the gate rather than a margin of
safety, and the direction matters if the limit ever comes down.

HOW TO TELL WHICH PATH A COUNT CAME THROUGH, without running the test:
`count()` returns it as the second element of its tuple, `PromptTooLarge`
carries it to the caller as `counted_with` in the 503 body, and `_load` and
`_template` each log once per model on first use — INFO naming the path taken,
WARNING naming what was missing. If you are looking at a `serialised` count and
expected `chat template`, the WARNING says which of the three causes it was.
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


def _template(model_id: str, models_dir: str):
    """The model's own chat template, compiled, or None.

    Counting message text alone is not counting the prompt. Measured on live
    traffic 2026-09-25: the gateway read a hermes request as 11,742 tokens while
    exo prefilled 18,118 — the model sees the template's own markup and, on this
    template, every tool definition rendered inline. 6,376 tokens of the
    difference was invisible to a counter that reads `messages`, and the limit
    was set from measurements of what exo actually prefills, so the two numbers
    have to mean the same thing.

    Rendered the way `apply_chat_template` renders it: sandboxed, `trim_blocks`
    and `lstrip_blocks` on, with the two globals templates reach for.
    """
    key = ("tmpl", model_id)
    if key in _CACHE:
        return _CACHE[key]
    tmpl = None
    path = os.path.join(models_dir, model_id.replace("/", "--"), "chat_template.jinja")
    try:
        # Imported on its own so a MISSING DEPENDENCY cannot be mistaken for a
        # missing model asset. Both used to land on the one `except` below and
        # print "no chat template for <model> at <path>", which reads as "this
        # model ships no template" — it sent #862 looking at `tokenizers`
        # versions and at the model directory for most of an investigation,
        # when the real answer was that `jinja2` was not installed at all and
        # `requirements.txt` had never declared it.
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415
        except ImportError as e:
            log.warning(
                f"prompt-size: jinja2 is not installed ({e}); every count will "
                f"use the serialised fallback, which under-counts a prompt "
                f"carrying tools. `pip install jinja2` — it is in "
                f"requirements.txt."
            )
            _CACHE[key] = None
            return None
        import datetime  # noqa: PLC0415

        def raise_exception(msg):
            raise RuntimeError(msg)

        def tojson(obj, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            # jinja2's own `tojson` takes no `ensure_ascii`, and this template
            # calls `{{ tool | tojson(ensure_ascii=False) }}` for every tool it
            # renders. Without this the render raises and counting silently
            # drops to the serialised fallback for exactly the requests the
            # template path exists to measure — the ones carrying tools.
            # Measured: 32 such falls in two days, every one of them a tool call.
            # Same signature transformers installs for apply_chat_template, and
            # deliberately not jinja2's HTML-escaping variant: exo renders
            # through transformers, and the escaping changes the token count.
            return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent,
                              separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        with open(path) as fh:
            tmpl = env.from_string(fh.read())
        log.info(f"prompt-size: counting {model_id} through its chat template")
    except Exception as e:
        log.warning(f"prompt-size: no chat template for {model_id} at {path} ({e}); "
                    f"counting the message text instead, which undercounts tools")
    _CACHE[key] = tmpl
    return tmpl


def _renderable(messages: list) -> list:
    """Messages in the shape the chat template expects.

    Two shapes that OpenAI clients send legitimately and this template cannot
    render as-is:

    - `tool_calls[].function.arguments` as a **JSON string**, which is what the
      OpenAI API specifies and what hermes sends. The template does
      `tc.arguments.items()` (line 66), so a string raises `'str object' has no
      attribute 'items'` and the whole render is lost. Measured 2026-09-27:
      identical payload, 29 tokens rendered with the arguments parsed, 126 by
      the serialised fallback — the fallback is not a small over-count on
      structured messages, so letting the render fail is expensive.
    - `content: None` on an assistant message that carries only tool_calls.

    Parsed rather than patched around: an argument string that will not parse is
    left alone, the render fails, and the fallback counts it. That is the safe
    direction and it keeps this function honest about what it can fix.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        calls = m.get("tool_calls")
        if isinstance(calls, list):
            fixed = []
            for call in calls:
                if isinstance(call, dict):
                    call = dict(call)
                    for holder in (call.get("function"), call):
                        if not isinstance(holder, dict):
                            continue
                        args = holder.get("arguments")
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                            except (ValueError, TypeError):
                                continue
                            if isinstance(parsed, dict):
                                if holder is call:
                                    call["arguments"] = parsed
                                else:
                                    holder = dict(holder)
                                    holder["arguments"] = parsed
                                    call["function"] = holder
                fixed.append(call)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def _serialised(messages: list, tools: Optional[list]) -> str:
    """Everything outbound, as JSON — the fallback when the template is missing.

    Deliberately the whole object rather than a chosen set of keys. Picking keys
    is what missed the tools: a field nobody thought of is a field nobody counts,
    and on this path the cost of being wrong is the host.
    """
    return json.dumps({"messages": messages or [], "tools": tools or []},
                      ensure_ascii=False)


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
          tools: Optional[list] = None,
          per_message_overhead: int = 8, fixed_overhead: int = 8) -> tuple:
    """(tokens, method) for what the model will actually be handed.

    Three ways down, most accurate first:

    1. **chat template** — render the model's own template over the messages and
       tools and tokenize the result. This is what exo does, so the number means
       the same thing as the measurements the limit was set from.
    2. **serialised** — tokenize the JSON of messages plus tools. Over-counts
       (keys, quotes) and misses the template's markup, but misses no field.
    3. **estimate** — characters over a conservative ratio, when there is no
       tokenizer at all.

    The overheads apply to 2 and 3, where the template's own wrapping is unseen.
    """
    tok, longest, _ctx = _load(model_id, models_dir)
    n_msgs = len(messages or [])
    overhead = fixed_overhead + per_message_overhead * n_msgs
    if tok is not None:
        tmpl = _template(model_id, models_dir)
        if tmpl is not None:
            try:
                rendered = tmpl.render(messages=_renderable(messages), tools=tools or None,
                                       add_generation_prompt=True)
                return len(tok.encode(rendered, add_special_tokens=False).ids), "chat template"
            except Exception as e:
                log.warning(f"prompt-size: chat template failed to render ({e}); "
                            f"counting the serialised request instead")
        return (len(tok.encode(_serialised(messages, tools), add_special_tokens=False).ids)
                + overhead), "serialised"
    text = _serialised(messages, tools)
    # No tokenizer: divide by the most token-dense ratio we measured rather than
    # an average one. Measured on this tokenizer, chars per token: prose 4.50,
    # python 3.86, JSON 3.38, log lines 2.76, CJK 2.00, base64-like 1.50. A
    # ratio of 1.5 therefore over-counts ordinary prose by about 3x. That is the
    # cost of not knowing, and it is paid only when the tokenizer is missing.
    return int(len(text) / chars_per_token) + overhead, f"estimate at {chars_per_token} chars/token"


def check(messages: list, model_id: str, limit: int, completion_budget: int,
          models_dir: str, chars_per_token: float, resource_id: str,
          context_length: Optional[int] = None, tools: Optional[list] = None) -> tuple:
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
    chars = sum(len(str(m)) for m in (messages or [])) + len(str(tools or ""))
    # Refuse an absurd body without tokenizing it. Sound because one token can
    # stand for at most `longest` characters, so anything longer than
    # limit*longest cannot possibly come in under the limit. Only a guard against
    # spending CPU on a body that is already hopeless — it is not the limit.
    if longest and chars > (limit * longest):
        raise PromptTooLarge(chars // max(longest, 1), limit, "unmeasured: body far over any possible limit",
                             completion_budget, resource_id, model_id, context_length)
    tokens, method = count(messages, model_id, models_dir, chars_per_token, tools=tools)
    if tokens > limit:
        raise PromptTooLarge(tokens, limit, method, completion_budget,
                             resource_id, model_id, context_length)
    return tokens, method
