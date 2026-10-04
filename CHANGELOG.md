# Changelog

## v0.1.1 (2026-10-03)

### Breaking Changes
- **noul 5→2**: removed `negative`, `needs_comfort`, `escalating` from output.
  These are derivable by the host (negative ≈ P<0; needs_comfort/escalating
  are better judged by the host's own LLM or ΔPAD history). Kept only the
  two text-only signals PAD cannot derive: `suppressed` (irony/repression)
  and `directed_at_me` (target attribution).
- **`battery_digest()` changed** (contract bump — 5→2 noul questions).

### Added
- **OCC-22 choice decode** (`occ.py`): 22-type Ortony-Clore-Collins emotion
  taxonomy with measured centroids from 2,106 self-rated cases. Pure PAD→distance
  ranking, zero additional training. `occ_rank()` returns all 22 sorted;
  `occ_top2()` includes straddling detection (d2/d1 < 1.3).
- **HTTP serve** (`serve.py` + `Dockerfile` + `docker-compose.yml`):
  FastAPI service with X-API-KEY hex auth (`sk-vibrato-` prefix).
  Deployed as Docker container on port 18973.
- **Clef benchmarks**: Cloudflare Clef-27B and Clef-flash-9B (both System One,
  Apache-2.0) added to the leaderboard. Clef-27B r̄=0.621 (A-dim 0.659,
  strongest arousal in any non-LLM). Clef-flash top2=1.00 vs Vibrato —
  System One models form a behaviorally coherent cluster.
- **System One Insight** (README): three evidence-backed observations on why
  System One reads emotion differently from LLMs (gestalt vs reasoning,
  alignment-tax immunity, dimensional convergence).
- **Embedding backbone comparison** (`eval_backbone.py`): e5-small (118M, MIT)
  beats bge-m3 (568M) on overall r̄ and dominates the A dimension.
  Full-knowledge bundle shrinks 417→85 MB.

### Fixed
- "chordia routing" mislabels in `battery.py` — renamed `route` → `host_signal`;
  Chordia is an emotion dynamics engine, not a behavior router.
- All `v6.3`/`v0.6.3` version references unified to `v0.1.1`.
- `feat_precompute.py` dim assertion now uses model's actual dim (not hardcoded 1024).
- `feat_any.py` bfloat16 → float32 conversion for numpy compatibility.

### Release Artifacts
| Package | Size | Contents |
|---|---|---|
| `vibrato-v0.1.1-dist.zip` | 17 MB | Code + weights + ONNX (backbone downloadable) |
| `vibrato-bundle-v0.1.1.zip` | 417 MB | Full-knowledge, bge-m3 int8 |
| `vibrato-fused-v0.1.1.zip` | 410 MB | Fused single-file ONNX |
| `vibrato-e5s-v0.1.1.zip` | 85 MB | Lightweight (e5-small, A-dim strongest) |

---

## v0.1.0 (2026-10-02)

Initial public release.

### Core
- **v1/affect message paradigm**: `POST /v1/affect/score` — messages in,
  distributions out. Each message may carry text form and/or state snapshots
  (pad / pressure / vitality). Scene break = state reset.
- **1.86M judgment head**: 3-layer transformer (d192, hand-written MHA, 512
  window), state pseudo-tokens, echo cls anchor, pluggable embedding backbone.
- **Battery**: score×3 (9-bin ordinal PAD) + choice (8-family trained +
  OCC-22 centroid decode) + noul×5 + conf.
- **Two-stage training**: mixed pretrain → battery-only finetune.
- **Contract digests**: pad_schema / battery / affect — three fingerprints,
  train/deploy must agree or refuse to run.

### Benchmarks (all measured)
- 16-way LLM answer-sheet evaluation (judge.py / vs_ref.py)
- Independent referee panel (dual blind raters, self-agreement r=0.98)
- CVAS 5-fold: V=0.709±0.017 (academic waterline ~0.70)
- CPED zero-shot: acc 0.222 / macro-F1 0.177
- ONNX precision: head int8 zero flips / full int8 2/40 flips
- Latency: head 2.97ms CPU-int8 / 1.92ms GPU / full path ~16ms GPU

### Demo
- HTML linkage demo: Vibrato × Chordia v2 (agent emotion dynamics engine)
  vs LLM self-enactment. Shows accumulation/hysteresis in the linked loop
  vs flat oscillation in LLM self-play.
