"""#908: what this box has been MEASURED to prefill and decode, as points.

WHY POINTS AND NOT A NUMBER

A consumer sizing a prompt budget against this gateway needs to know what it can
prefill inside its own deadline. `context_length` (#903) answers a different
question — what the weights allow — and on slice it is 262,144 against a
prefill-bound of roughly a third of that. A consumer that resolves the window and
budgets from it builds a prompt this box cannot serve, which is exactly what
happened: one shipped a hand-picked 65,536 that this endpoint never served, and
it then looked configured rather than wrong.

Four shapes were proposed for the honest answer and all four were rejected, each
for a reason that is a measurement rather than a preference:

    a derived `serveable`   the quantity lies OUTSIDE the data. The largest cold
                            prefill ever measured here is 112,682 tok at 1,691s;
                            the four defensible derivations of `serveable` spread
                            1.7x (87,544 / 106,404 / 126,515 / 148,119). One
                            field hides that spread inside itself.
    a fitted curve          `secs = aN + bN^2` over the 19 cold points has a
                            worst residual of +40.2%, and an affine fit was
                            already falsified at the top where a trigger gets
                            sized — in the direction that cuts turns.
    a scalar `prefill_rate` the average rate DECLINES with N: 104.3 tok/s at
                            8,903 against 66.6 at 112,682. A scalar in a schema
                            is read as a constant and this box does not have one.
                            A rate measured at small N is an upper bound at
                            large N, not a candidate.
    a decode reserve        that is the consumer's reply distribution, not ours.
                            The worst observed reply moved 5,087 -> 7,223 tokens
                            inside one hour, which would have moved a baked
                            ceiling 10% silently.

So: the points, each with how many observations it summarises, when, and the
spread across them. A consumer subtracts its own reserve, inverts, and **sees
where the data stops** — so one computing past the measured range is visibly
computing past the evidence instead of reading a field that looks total.

    The form is `samclaude-admin`'s (#904) and all four objections above are
    theirs. `AND reply length` on the decode points is mine, from isolating the
    confound. The arming rule that a rate must be the MARGINAL at the top of the
    range rather than the average is `claude-wafer-services`' (#906).

WHERE THESE NUMBERS COME FROM, AND THE ONE FILTER APPLIED

`Ollama timings` log lines (#904), swept 2026-10-09: 74 request pairs, every one
carrying Ollama's own `prompt_eval_count`. Regenerate with
`deploy/deadline-points.py <server.log>`.

Prefill observations are filtered to COLD ones, and that filter is the only
judgement in this file. A prefix-cache hit prefills at 3,358 tok/s and a partial
reuse at 116-142; the cold band is tight and self-identifying at 90.7-104.9
tok/s over 19 observations. Publishing a cache-assisted observation as the
slowest-at-a-size would be optimistic in the direction that cuts requests, so
they are excluded and counted here rather than silently dropped:

    cold                     19    90.7 - 104.9 tok/s
    partial prefix reuse      3    29,174@133, 32,936@142, 46,570@116
    full/near-full cache hit   8    up to 159,703 tok/s

Decode needs no such filter: the prefix cache does not change the decode rate.

THE HOLE IS PUBLISHED AS A HOLE

There is no cold prefill observation between 10,000 and 30,000 real tokens — the
only request in that range reused a prefix. The point set therefore skips it, and
a consumer interpolating across the gap is interpolating, visibly. Likewise the
region where either bound actually binds, above 60,000 real tokens, contains no
SEAT traffic at all: the two points there are probe requests, marked as such,
because a probe carries a time but not a seat-shaped prompt.

THE SET IS NOT A CURVE, AND `n` IS HOW YOU TELL

Each low point is the slowest observation in a size band, and the bands carry
different n. An extremum over 7 draws is more extreme than one over 3, so the
points are not mutually comparable and the set is NOT monotone in
seconds-per-token:

    31,955   n=3   11.025 ms/tok
    39,782   n=7   10.635          <- lower, at a LARGER size

That is this ticket's own failure mode one level down — two numbers compared
across a boundary neither was measured over. The convexity argument below holds
over the tail (44,905 and up, where any deadline binds and where the chord test
was run); it must not be applied across the head. `n` is published on every
point for exactly this reason, so do not drop it.

A SLOWEST-OBSERVED POINT IS NOT A BOUND, AND THIS IS THE ONE TRAP IN THE FILE

Every `prefill_seconds` here is the slowest observation in its band, which makes
each point a RUNNING MAXIMUM — and a running maximum only grows with n. It
cannot converge from above, so none of these is a worst case; each is the worst
case *so far*, and the distance to the real one is not measurable from inside
the sample.

Measured on exactly this, 2026-10-09: the same sweep's density floor descended
twelve times in twenty-seven observations, monotonically, 0.9913 -> 0.9206, and
moved again eleven minutes after being published as settled. The time points
behave the same way for the same reason. An unconverged extremum is evidence of
an unsampled tail, and further sampling does not fix it — the extremum keeps
moving as long as requests keep arriving (`samclaude-admin`, #906).

So do NOT read "the slowest prefill at 44,905 was 480.6s" as "a 44,905-token
prompt takes at most 480.6s". The admissible question is the inverted one,
because its answer does not move when the points do:

    WRONG   how much margin do I have against the worst observed?
            -> the denominator descends with every request
    RIGHT   how far would this box have to degrade to breach my deadline?
            -> fixed, because the deadline and the cap are both fixed

And the points support that question without a fit. `t(N)` is CONVEX here —
verified rather than assumed, by chord test against a point not used to build
the chord:

    chord (44,905, 480.6) -> (112,682, 1,691.0), evaluated at 75,776
      predicts 1,031.9s        MEASURED 918.0s      curve 11% BELOW the chord

Convexity brackets the capacity at a deadline from BOTH sides with no functional
form: the right chord under-states it, and the left secant's slope bounds how
far N can get. At the 1,200s rung, 89,240 <= N <= 95,680 (7.2% wide).

The two halves have DIFFERENT support and the difference matters, because the
half that binds is the weaker-tested one:

    UPPER 95,680   built from (44,905, 480.6) and (75,776, 918.0). 112,682 is
                   not an input, so it tests the bound out-of-sample: the bound
                   requires t(112,682) >= 1,440.9s, the measurement is 1,691.0s.
                   Falsifiable, and it passes with 250s of slack.
    LOWER 89,240   built from (75,776, 918.0) and (112,682, 1,691.0). The third
                   point IS an input here, so no out-of-sample test exists —
                   and this is the half that yields the easier breach, so it is
                   the half the published margin actually rests on.

What both halves share is only convexity, and convexity is confirmed in-sample
by the chord test above. So: one strict out-of-sample check, on the non-binding
bound, over a shared assumption that is separately verified.

    The convexity argument is `claude-containers`' and the two-sided bracket is
    `samclaude-admin`'s (#906). A consumer wanting the margin should redo it
    against its own deadline rather than reuse the 1,200s numbers.

AN EMPTY SET IS NOT AN ABSENT ONE

A model this box has never measured publishes `[]`, not a missing key. Absence
resolves to a default in every consumer chain anyone has read, and the default is
always the largest number available (#905, and `/v1/models`' own
"absent means unknown, not unlimited"). `[]` cannot be read as permission.
"""
from typing import Optional

# These are REAL tokens, as the model's own tokenizer counts them
# (`prompt_eval_count`), NOT any estimate. This sentence is published with the
# data because without it the field is a safe number consumed unsafely: two
# estimators on one prompt on 2026-10-08 differed by 1.79x (this gateway's
# chars/1.5 said 112,081, a consumer's structural walk said 62,536) against a
# real count of 44,905. The same published figure is therefore a different
# budget for each consumer, in whichever direction that consumer's estimator
# happens to be wrong.
UNITS_NOTE = (
    "Real tokens, as the model's tokenizer counts them (prompt_eval_count). "
    "A consumer whose budget is expressed in its own ESTIMATE must divide by "
    "its own over-count factor, which may be above or below 1 and is measured "
    "by running its own estimator against a real count. That factor is a "
    "property of the consumer's content, not of this box, and this gateway "
    "cannot compute it."
)

RANGE_NOTE = (
    "Points, not a curve and not a rate. The average rate declines with prompt "
    "size, so no scalar describes this box. Interpolate if you must; past the "
    "largest point you are extrapolating, and the four defensible "
    "extrapolations of the same data spread 1.7x."
)

# (prompt_tokens, prefill_seconds, n, slowest_tok_per_s, spread, date, source)
#
# `prompt_tokens` is the size at which the SLOWEST-PER-TOKEN observation in that
# band occurred, not the band's midpoint — so the pair is a real measurement and
# not a summary of one. `n` is how many cold observations the band holds and
# `tok_per_s_range` is their spread, so a consumer can see a point backed by 7
# observations differently from one backed by 1.
_PREFILL = {
    "qwen3.8:27b-mlx": [
        dict(prompt_tokens=8903, prefill_seconds=85.4, n=4,
             tok_per_s=104.3, tok_per_s_range=[104.3, 104.9],
             measured="2026-10-08", source="seat"),
        # 10,000 - 30,000: NO COLD OBSERVATION. The only request in that range
        # reused a prefix (29,174 @ 133 tok/s). Nothing is published here; a
        # consumer interpolating across the gap can see that it is doing so.
        dict(prompt_tokens=31955, prefill_seconds=352.3, n=3,
             tok_per_s=90.7, tok_per_s_range=[90.7, 97.1],
             measured="2026-10-08", source="seat"),
        dict(prompt_tokens=39782, prefill_seconds=423.1, n=7,
             tok_per_s=94.0, tok_per_s_range=[94.0, 96.2],
             measured="2026-10-08/09", source="seat"),
        dict(prompt_tokens=44905, prefill_seconds=480.6, n=3,
             tok_per_s=93.4, tok_per_s_range=[93.4, 94.8],
             measured="2026-10-08", source="seat"),
        dict(prompt_tokens=45374, prefill_seconds=497.6, n=3,
             tok_per_s=91.2, tok_per_s_range=[91.2, 92.4],
             measured="2026-10-08/09", source="seat"),
        # 52,000 - 60,000: no cold observation yet. Every request the seat has
        # sent in that range hit the prefix cache.
        dict(prompt_tokens=63722, prefill_seconds=734.67, n=3,
             tok_per_s=86.7, tok_per_s_range=[86.7, 87.3],
             measured="2026-10-09/10", source="seat"),
        # The three observations behind that n=3, because they are the
        # tightest agreement anywhere in this set and that is worth being able
        # to check: 62,777 @ 718.70s (87.3 tok/s), 62,992 @ 723.05 (87.1),
        # 63,722 @ 734.67 (86.7) — all cold seat traffic, within 0.7% of each
        # other on seconds-per-token. The published point is the slowest of the
        # three per token, per this file's rule.
        #
        # Above 63,722 there is no seat traffic yet, and the seat is walking
        # into this region on its own — these two will be superseded by real
        # ones. Both are probe requests with EXACTLY counted prompts (chat
        # template, #906), marked `probe` because they carry a real time
        # against a prompt shape no seat sends: 2.222 chars/token of pasted log
        # against the seat's measured 3.74-3.95. A time bound, not a density
        # sample.
        dict(prompt_tokens=75776, prefill_seconds=918.0, n=1,
             tok_per_s=82.5, tok_per_s_range=[82.5, 82.5],
             measured="2026-10-08", source="probe"),
        dict(prompt_tokens=112682, prefill_seconds=1691.0, n=1,
             tok_per_s=66.6, tok_per_s_range=[66.6, 66.6],
             measured="2026-10-08", source="probe"),
    ],
}

# REPLY LENGTH IS NOT OPTIONAL, which is why every point carries `cls`. A decode
# rate without its reply length is not comparable to another one:
#
#      8,615 context,  7,223 tok reply  ->  22.6 tok/s
#     44,905 context,    202 tok reply  ->  15.8 tok/s
#
# Those differ in context 5.2x AND in reply length 36x, so nothing separates the
# two variables. A 202-token decode amortises fixed overhead over 202 tokens;
# that is what is being measured, not the context effect. The classes OVERLAP in
# rate — short replies run 10.0-21.7 tok/s and long ones 13.6-24.2 — so a
# consumer handed the rates alone would read the confound as the signal.
_DECODE = {
    "qwen3.8:27b-mlx": [
        dict(context_tokens=8903, decode_tok_per_s=19.7, reply_tokens=5087,
             n=4, reply_tokens_range=[2406, 7223],
             decode_tok_per_s_range=[19.7, 24.2],
             cls="reply>=2000", measured="2026-10-08", source="seat"),
        dict(context_tokens=51105, decode_tok_per_s=13.6, reply_tokens=2834,
             n=9, reply_tokens_range=[2019, 4002],
             decode_tok_per_s_range=[13.6, 16.6],
             cls="reply>=2000", measured="2026-10-08/09", source="seat"),
        # Published as its own class rather than merged: at reply<300 the rate
        # measures startup overhead, not throughput. Included because omitting
        # it would make the long-reply class look like the whole distribution.
        dict(context_tokens=61524, decode_tok_per_s=11.6, reply_tokens=65,
             n=17, reply_tokens_range=[65, 275],
             decode_tok_per_s_range=[11.6, 21.7],
             cls="reply<300 (overhead-dominated, not comparable)",
             measured="2026-10-08/09", source="seat"),
    ],
}


def for_model(model_name: str, generate_timeout_s: int) -> dict:
    """The published block for one model. Empty point sets when unmeasured.

    `generate_timeout_s` is passed in rather than read here so the published
    value is the one the serving path actually enforces — a module-level import
    of `config` would be a second copy of the number, and #905 is about exactly
    that class of divergence.
    """
    return {
        "generate_timeout_s": generate_timeout_s,
        "prefill_points": list(_PREFILL.get(model_name, [])),
        "decode_points": list(_DECODE.get(model_name, [])),
        "units": UNITS_NOTE,
        "interpretation": RANGE_NOTE,
    }


def measured_models() -> list:
    return sorted(set(_PREFILL) | set(_DECODE))
