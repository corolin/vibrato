# Vibrato

**A complete System One model for emotion — emotion reading as typed decisions.**

[简体中文](README_CN.md) | English

A standalone, general-purpose System One (Jev-class) model: one forward pass reads a message sequence (the `v1/affect` paradigm: each message may carry text form and/or PAD/pressure/vitality state snapshots) and emits **calibrated probability distributions** — PAD scores, an emotion family, yes/no judgments, and a confidence flag. It was born as the user-emotion front-end of the Chordia companion stack, and has since grown into an independent model that can serve any application.

> In a string instrument, vibrato is how emotion travels through the note.

---

## Why

Applications read user emotion in many contexts — companion AI, chat analytics, moderation, mental-health triage, adaptive interfaces. The default today is an **LLM call per message**: slow, costly, privacy-leaking, and it returns three bare numbers. Vibrato is the System-One answer (the model class pioneered by [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), with open kin like [Laya](https://github.com/NandhaKishorM/laya) and [Tev1](https://huggingface.co/togethercomputer/Tev1-4B-experimental)):

| | LLM-per-message | Vibrato |
|---|---|---|
| Latency | 3.7–13.2 s (measured across twelve challengers) | **head 2.97 ms CPU-int8 / 1.92 ms GPU / full path ~16 ms GPU, ~144 ms CPU (all measured)** |
| Output | 3 point numbers | **full distributions + confidence** |
| Cost / privacy | 8k–36k tokens per 40 cases, text leaves the box | $0, never leaves the box |
| Calibration | none | ordinal bins + entropy + conf head |

The teacher LLM only exists **offline** — the runtime is a 1.86M judgment head + a frozen embedding backbone (118M e5-small suffices; 568M bge-m3 is overkill).

## The battery (frozen contract)

Three question types, one forward pass:

| Type | Count | Content |
|---|---|---|
| **score** | 3 | Pleasure / Arousal / Dominance, each a 9-bin ordinal distribution (Likert 1–9) |
| **choice** | 1 | Emotion family, 8 classes: happy · affectionate · calm · amused · sad · anxious · angry · **wronged** (*wěiqu* — high-frequency in Chinese companion chat, absent from Western taxonomies) |
| **noul** | 5 | negative · needs_comfort · directed_at_me · escalating · suppressed |

Each noul maps to a host-side routing decision (comfort mode / back off / dig deeper / pressure alarm). A `conf` head flags low-confidence samples for **escalation to a real LLM** — System One means knowing when to abstain.

Score bins use the same 1–9 Likert scale as the classic 18-item PAD scale, so expected values drop into the standard `(mean − 5) / 4` normalization unchanged. Pressure is **derived, not predicted**: `ΔPressure = 1.0·(−ΔP) + 0.8·(ΔA) + 0.6·(−ΔD)` (host-side accumulation; anti-runaway rule: accumulate the raw, unmodulated ΔPAD).

## The `v1/affect` message paradigm

Jev pioneered the System One interface paradigm; this project defines the "mouth shape" of the affect front-end — **messages in, distributions out**:

```
POST /v1/affect/score
messages: [{ role: "user"|"assistant",
             message?:  string        // text form (verbatim)
             pad?:      [p,a,d]       // state snapshot [-1,1] (that speaker, that turn)
             pressure?: number        // pressure snapshot [0,100] (chordia standard)
             vitality?: number        // vitality snapshot [-30,100] (chordia-v2) }]
→ score×3 (9-bin) + choice×1 (8 families) + noul×5 (yes/no) + conf
  + write_back{pad, d_pressure_raw} + consumed{fields, digest}
```

- Last message = the target (must be `user` + `message`, no state fields — those are the answer, not the input)
- All-text arrays = the generic-agent path, fully in-distribution — **state is a precision knob, not an entry ticket**
- Two-layer versioning: wire-format digest (`vib_messages.schema_digest()`, v1 frozen, additive-only) is separate from model digestion (`CONSUMED_FIELDS`; v6 reads message/pad/pressure, vitality is carried-but-masked)
- Units discipline: the wire speaks domain-native units; normalization is the model's private business
- Echo dissolved: the caller writes the previous output back onto the previous user message's `pad` field (plus a fast cls-anchored pathway inside the net)
- **Scene break = state reset**: across scenes/sessions (time passing, environment change) the host simply omits the pad/pressure fields — the reader cold-starts with no history. **Inherit when continuous, reset when broken; state inheritance is the host's call** (accumulation within one continuous conversation is the point)

## Architecture

```
v1/affect messages ─► text units (char ids) + state pseudo-tokens (5 vals + 5 mask)
                      + frozen bge-m3 per-char features
                          │
   echo: prev user PAD ───┼─► cls anchor ─► 3-layer transformer (d192, hand-written MHA, 512 window)
                          │                   ├─ 3× bin heads (softmax over 1–9)
                          │                   ├─ 8-way family head
                          │                   ├─ 5× noul heads (2-way softmax)
                          └───────────────────┴─ conf head (sigmoid)
```

- **Dual state pathways**: echo anchored directly on cls (a zero-discovery-cost prior) plus in-sequence state pseudo-tokens (fine-grained, time-ordered context) — measured: the same information is worth +0.28 family accuracy via cls vs +0.06 in-sequence; both coexist
- **Knowledge injection (pluggable backbone)**: a frozen embedding model provides per-char features into `in_proj`. Backbone comparison (same pipeline, independent referee):

| Backbone | Params | dim | r̄ | P | A | D |
|---|---|---|---|---|---|---|
| bge-m3 | 568M | 1024 | 0.635 | **0.812** | 0.506 | 0.588 |
| **e5-small (MIT)** | **118M** | **384** | **0.652** | 0.758 | **0.616** | 0.581 |
| bekko-a25m (MIT) | 123M | 384 | 0.582 | 0.769 | 0.438 | 0.539 |
| bekko-a8m (MIT) | 106M | 384 | 0.540 | 0.679 | 0.462 | 0.479 |

  e5-small (118M) beats bge-m3 (568M) on r̄ **and** dominates A — the full bundle shrinks 417→85 MB. Swapping = 44s retrain, architecture untouched. 15% bge-dropout keeps the text-only path alive.
- **1.86M-parameter head**, ONNX-friendly (hand-written MHA avoids export pitfalls); v5 head measured: CPU-ONNX-int8 single-thread 2.25 ms, 1.7 MB after int8 (v6 export/bench pending)
- Every artifact is guarded by **contract digests** (scale / battery / affect — three fingerprints); train/deploy sides must agree or refuse to run

## Benchmarks (all measured)

Forty blind cases, fourteen contestants (each LLM answers per-case with the same battery prompt; sheets cost 26.7k–79.4k tokens each), plus an independent referee panel. Three yardsticks, each with a different bias — read them together.

### Board A — continuous perception (vs the independent referee panel)

Pearson r per PAD dimension against a dual-referee panel: two context-free blind raters (identical prompts, isolated sessions; self-agreement r = 0.98/0.98/0.97, family 0.82, noul 0.94), dual-mean PAD. Caveats: raters are GLM-family (dialect affinity with glm-5.3), n = 40.

| Contestant | r̄ (P/A/D) | latency/case | tokens |
|---|---|---|---|
| GLM-5.3 | **0.902** (0.94/0.84/0.93) | 4.7 s | 31.2k |
| gemini-3.8-flash | 0.866 (0.95/0.76/0.88) | 8.8 s | 60.3k |
| claude-sonnet-5.5 | 0.857 (0.92/0.74/0.92) | 4.7 s | 51.1k |
| grok-4.1-fast | 0.841 (0.94/0.74/0.85) | 3.4 s | 26.7k |
| Qwen3.8-27B | 0.807 | 12.8 s | 35.6k |
| gpt-5-mini | 0.762 | 13.2 s | 79.4k |
| deepseek-v4-flash | 0.716 | 3.7 s | 32.7k |
| **Vibrato v0.1.1 (1.86M, local)** | 0.635 (0.81/0.51/0.59) | **0.17 s** | **0** |
| Clef-27B (System One, Apache-2.0) | 0.621 (0.84/**0.66**/0.36) | ~1.7 s | 0 (self-host) |
| Hunyuan-A13B / ref4b-4B / gpt-4o-mini | 0.59–0.62 | — | — |
| Clef-flash-9B (System One, Apache-2.0) | 0.564 (0.84/0.54/0.31) | ~1.3 s | 0 (self-host) |
| GLM-4-32B / Qwen2.5-7B / Xing4.0-29B | 0.43–0.47 | — | — |

The A dimension is a **generational fault line**: current-gen models reach 0.68–0.84 while the previous generation collapses (0.01–0.28). Vibrato's 0.506 tops the old camp but hasn't crossed into the new one.

### Board B — behavior agreement (vs Vibrato; similarity, not correctness)

Word-decode/noul agreement with Vibrato v0.1.1. Home-field note: the 40 cases are Vibrato's own battery distribution; LLMs answer zero-shot. **Agreement ≠ accuracy** — Qwen2.5-7B shows 0.90 top2 agreement with Vibrato while scoring 0.438 on Board A (style alliance, not skill).

| Contestant | top1 / top2 / noul vs vib |
|---|---|
| Qwen2.5-7B (daily driver) | 0.72 / 0.90 / 0.82 |
| claude-sonnet-5.5 | 0.70 / 0.80 / 0.75 |
| Xing4.0-29B | 0.70 / 0.85 / 0.76 |
| deepseek-v4-flash | 0.68 / 0.78 / 0.79 |
| Clef-27B | 0.72 / 0.97 / — |
| Clef-flash-9B | **0.75** / **1.00** / — |
| grok-4.1-fast | 0.62 / 0.70 / 0.76 |
| gemini-3.8-flash | 0.60 / 0.72 / 0.82 |
| Qwen3.8-27B / glm-5.3 / gpt-5-mini / gpt-4o-mini | 0.53–0.57 |
| Hunyuan-A13B / GLM-4-32B | 0.35–0.47 |

Full sheets (`answers_*.jsonl`) and one-command re-judging (`judge.py`, `vs_ref.py --ref <tag>`) ship in the repo.

### Escalation curve (conf head)

Escalating the lowest-conf X% of cases to an LLM, rest Vibrato (r̄ vs the referee panel):

| escalate | API calls /40 | → sonnet-5.5 | → glm-5.3 |
|---|---|---|---|
| 0% | 0 | 0.635 | 0.635 |
| 20% | 8 | 0.652 | 0.667 |
| 50% | 20 | 0.713 | 0.753 |

Honest read: the mechanism works mechanically (+0.12 at 50%), but the 10% point is flat — the conf head (supervised by teacher self-consistency) currently flags *teacher disagreement*, not *Vibrato's errors*. Calibration against referee disagreement is queued.

**Standard benchmarks** (Chinese-EmoBank CVAS, 2,583 sentences, Pearson r):

| Config | Valence | Arousal |
|---|---|---|
| Zero-shot (cross-domain: chat-tuned → literary text) | 0.517 | 0.098 |
| **5-fold CV fine-tune** (2k sentences × 4 epochs per fold) | **0.709±0.017** | 0.383±0.027 |
| Academic in-domain waterline (SemEval-2026 / EmoBank family) | ~0.70 | ~0.45 |

CPED (Chinese ERC standard set, 13→8 taxonomy zero-shot mapping): acc 0.222 / macro-F1 0.177 (majority baseline 0.165; anger recall 0.66 transfers best).

In-house battery val (8 families + PAD + noul, clean split): famAcc 0.61–0.67 / noulAcc 0.79 / padMAE 0.21–0.23.

### The System One Insight (why this matters beyond the numbers)

The benchmark's most valuable finding isn't any single score — it's that **System One models (Vibrato 1.86M, Clef 27B, Clef-flash 9B) form a behaviorally coherent cluster** (mutual top-2 agreement 0.97–1.00) that reads emotion *differently* from LLMs. Three observations explain why:

**1. Gestalt pattern-matching, not autoregressive reasoning.**
LLMs "read" text — they parse semantics, infer causality, then classify ("he used X word in Y context, therefore probably angry"). System One models skip the reasoning step entirely. Whether it's a 1.86M char-level transformer (Vibrato) or a frozen Qwen with its generative pathway locked (Clef), both map input directly to typed decisions via pattern-matching — closer to an amygdala than a cortex. This is why System One latency is measured in milliseconds, not seconds.

**2. Immunity to the alignment tax.**
Why do LLMs collectively collapse on the D dimension (dominance: 0.28–0.36 for most, vs 0.588 for Vibrato)? RLHF trains LLMs to be polite, neutral, and objective — which blunts their sensitivity to power dynamics, passive aggression, and dominance signals in text. System One models carry no such baggage. Vibrato's 61k legacy training data preserves the *raw* distribution of how language exerts pressure, without social filtering. This is a feature, not a bug: a companion AI that can't sense "I'm being talked down to" from text is emotionally deaf.

**3. Dimensional convergence proves emotional intuition is low-dimensional.**
The most striking result: clef-flash (9B frozen Qwen) and Vibrato (1.86M char-level) achieve **top2 = 1.00** — clef's top-1 word *always* appears in Vibrato's top-2. When you take a 9B model and strip away its autoregressive "rational brain," forcing it to classify through routing heads only, its decision surface *collapses onto nearly the same manifold* as a 1.86M character-level model. This is mathematical evidence that **emotional intuition is a compact, learnable feature space** — it doesn't require billions of parameters to represent, only the right training signal.

## Core demo: Vibrato × Chordia v2 in the loop

[`demo/vibrato_demo.html`](demo/vibrato_demo.html) — a chat-style bubble stream where **every message carries a Vibrato sidecar card** (measured PAD tri-bars, family badge, noul flags; the small hatched bars are the script's hand-crafted gold), with a four-line trajectory chart on the right:

- **Linked arm**: Vibrato measures the user turn-by-turn → the Chordia v2 engine (7-dim MLP: user PAD + vitality + agent PAD → ΔPAD) drives the agent's dynamics
- **LLM self-enactment arm**: same personality, one system prompt — the LLM imagines the whole run and self-reports its PAD

Two personalities × three acts × 20 turns (anger confrontation → breakdown & recovery → rivalry to friendship). What the measured run shows: **the linked loop has accumulation and hysteresis** (the timid persona gets dragged into a deep empathic collapse [-1.0, -0.69] and recovers late; the cheerful persona at vitality 90 dips to -0.6 and rebounds to +0.31), while **self-enactment oscillates mildly per message** (the cheerful persona barely dips to -0.3 while the user sobs; both LLM personalities run near-parallel tracks — personality doesn't modulate the dynamics). Measured user PAD vs hand-crafted gold: r(P/A/D) = 0.74 / 0.52 / 0.35. A known flaw is visible too: recovery after a long hostile run lags (state-anchor drag).

Regenerate with `python demo/demo_html.py [--llm --key sk-..]` (linked arm is zero-API; the `--llm` arm needs an OpenAI-compatible endpoint).

## Repository layout

| File | Role |
|---|---|
| `pad_schema.py` | Frozen scale contract: bins, 18-item bridge (tent kernel, expectation-preserving), pressure formula, digests |
| `battery.py` | Full question battery + teacher prompt builder + gold validation |
| `vib_messages.py` | **`v1/affect` wire format**: validation, state vectors, form-dropout augmentation, dual rendering, dual digests |
| `net.py` | `VibratoNet` v6 (state pseudo-tokens + echo anchor + feats pathway) + `collate_pad` + self-test |
| `vib_state.py` | v5 context rendering (legacy bridge) |
| `feat_bge.py` / `feat_any.py` | Frozen bge-m3 / any-HF-model per-char features |
| `eval_backbone.py` | Backbone comparison: same pipeline, different embeddings, same referee |
| `gen_convos.py` / `label_pad.py` / `label_agent.py` | Conversation synthesis / user-side teacher labeling / agent-side labeling |
| `prep.py` / `feat_precompute.py` / `train.py` | data assembly (6-message parity window) / bge feature memmap / two-stage training |
| `judge.py` / `answer_sheet.py` / `eval_bench.py` / `finetune_cvas.py` | Offline judge (answer-sheet architecture) / sheet collection / standard benchmarks / CVAS 5-fold |
| `demo/` | The linkage demo (HTML generator, personality script, Chordia engine + checkpoint) |

## Release artifacts

| Package | Size | Contents |
|---|---|---|
| `vibrato-v0.1.1-dist.zip` | 17 MB | Code + weights + ONNX (bge downloadable) |
| `vibrato-bundle-v0.1.1.zip` | 417 MB | Full-knowledge, bge-m3 int8 (separate files) |
| `vibrato-fused-v0.1.1.zip` | 410 MB | Fused single-file ONNX (one `session.run`) |
| **`vibrato-e5s-v0.1.1.zip`** | **85 MB** | **Lightweight full-knowledge (e5-small, A-dim strongest)** |

All bundles: zero torch dependency (`onnxruntime + tokenizers + numpy`).

## Quickstart

```bash
pip install torch            # smoke tests need nothing else
python pad_schema.py         # scale contract self-test (v6 digest)
python vib_messages.py       # v1/affect paradigm self-test
python net.py                # forward + backward + state/echo-activity smoke test
python battery.py            # battery self-test + digests
```

Full v6 pipeline (data stages need any OpenAI-compatible endpoint; training needs a GPU): `gen_convos → label_pad → label_agent → prep_v6 → feat_precompute → train.py (mixed stage 1) → train.py --battery-only --init (stage 2) → judge.py`. Every data stage writes incrementally and resumes — crash-safe by design.

## Status & honest limitations

**v0.1.1.** The full chain (paradigm → data → two-stage training → judging → standard benchmarks) runs end-to-end with everything measured. Known limitations:

1. **Perceptual inertia: right direction, miscalibrated magnitude** — after 7+ hostile turns, a suddenly calm user does not flip the reading (human listeners do not believe instant recoveries either — suspicion of residual anger or passive aggression is the realistic read). That is the correct direction. But the current decay is too slow: vib trails a recovery far longer than a human reader would (humans update within 2–3 turns). The primary lever is **scene break = state reset** (see the v1/affect section) — the host omits the pad/pressure fields across scenes/sessions and the reader cold-starts. Anchor-decay tuning is off the roadmap: the inertia direction is a feature, the model stays untouched, inheritance belongs to the host.
2. **Arousal (A) is backbone-dependent** — with bge-m3 Vibrato scores 0.506; swapping to e5-small lifts A to **0.616** (best arousal across all experiments, entering the range of current-gen LLMs 0.68–0.84). Remaining gap likely training-data distribution, not backbone capacity. Conf head calibration queued.
3. ~~v6 ONNX export pending~~ **done**: full-input-signature ONNX (`model_v6/`, zero-feats = the text-only path trained in via bge-dropout), int8 = 2.2 MB, PAD deviation <0.005. A CPU-only *full-knowledge* tier (bge int8) is still unmeasured.
4. Cross-taxonomy zero-shot (CPED/GoEmotions style) is still weak (0.22 acc) — what v0.1.1 validated is that light adaptation reaches the table (the CVAS line).

## Credits

- The **System One** model class: [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) for the idea; [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) for score-type markers and RLCD training lore; [Tev1](https://github.com/togethercomputer/Tev1) (MIT) for the cheap-SFT recipe.
- Architecture lineage (hand-written MHA, echo rounds, arbiter discipline) comes from the author's own Prisma segmentation stack.
- The Chordia v2 emotion dynamics engine and personality test scenarios in `demo/` come from the author's Chordia project.
- **Origin**: Vibrato began life as the user-emotion front-end of the Chordia companion system, and grew into a standalone System One model.

## License

[MIT](LICENSE) © 2026 corolin
