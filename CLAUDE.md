# CLAUDE.md — ST-GCS Project Instructions

## Agent folder

All discussion summaries, design documents, and chat records for this project live in `agent/`. When a conversation produces a meaningful design decision, architectural proposal, or research direction, save a summary markdown file there before the session ends.

Current documents:
- `agent/AGENT.md` — project overview, environment setup, file structure, key concepts
- `agent/BVC_RESERVATION.md` — design doc for BVC-based trajectory reservation (replacing ECD)
- `agent/CVT_RESERVATION.md` — design doc for CVT-based tessellation reservation (alternative to BVC)
- `agent/LAZY_BVC_RESERVATION.md` — design doc for lazy halfplane expansion (tries ECD halfplanes one at a time, BVC centroid direction first)

## Saving chat records

At the end of any session that produces design decisions or architectural changes, write a markdown summary to `agent/` using the naming convention `TOPIC_YYYYMMDD.md` (e.g., `BVC_RESERVATION.md`, `ROLLING_HORIZON_20260613.md`). The file should cover:
- What was decided and why
- Specific implementation plan (files to change, function signatures)
- Tradeoffs and open questions

## Project context

See `agent/AGENT.md` for full environment setup, file structure, and key concepts.

Active research branch: `RH` (rolling-horizon planning).

Three primary research changes under development:
1. **BVC-based reservation** (`mrmp/region_reservation/bvc.py`) — replaces ECD slicing with per-pair buffered Voronoi halfspace constraints added directly to GCS vertex sets. See `agent/BVC_RESERVATION.md`.
2. **CVT-based reservation** (`mrmp/region_reservation/cvt.py`) — replaces ECD slicing with a centroidal Voronoi tessellation that partitions each contested vertex into exactly $n$ cells (one per contesting agent). Globally consistent partition, no which-side ambiguity. See `agent/CVT_RESERVATION.md`.
3. **Lazy BVC reservation** (`mrmp/region_reservation/lazy_bvc.py`) — lazily expands ECD halfplanes one at a time (BVC centroid direction first), re-solving after each addition. Best case: 1 sub-vertex; worst case: ECD-equivalent. See `agent/LAZY_BVC_RESERVATION.md`.
4. **Rolling-horizon planning** — GCS solves only in a near horizon; graph search abstracts the far horizon.

## Code conventions

- Run all scripts from the **project root** (`cd /path/to/stgcs`).
- Drake GCS backend: `mrmp/graph.py` wraps `GraphOfConvexSets`. Do not call Drake GCS directly; go through the `Graph` and `STGCS` wrappers.
- `stgcs.copy()` is the correct way to get a fresh copy for per-agent reservation — do not mutate the shared `stgcs` object.
- `BASE_MAX_ROUNDED_PATHS = 1000`, `BASE_MAX_ROUNDING_TRIALS = 1000` in `mrmp/stgcs.py` — do not reduce these; lower values (e.g. 50/500) cause the planner to fail to find feasible solutions. Do not reintroduce log-linear scaling on top of these.
