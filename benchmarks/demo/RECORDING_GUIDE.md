# REDI SC26 Demo — Recording Guide

Total target: **90 seconds** (parts sped up in iMovie)

## Prerequisites

```bash
source ~/venv/redi/bin/activate
cd ~/drai/redi
# verify asciinema
asciinema --version
```

Set terminal: **220×50**, dark theme, large font (18–20pt), no window chrome.

---

## Scene 1 — Claude Code interaction (~12s, normal speed)

Record yourself typing a prompt in Claude Code, watching the jobs get submitted.
No asciinema needed — screen record with QuickTime.

Suggested prompt:
> "Submit the REDI output jobs for all four SC26 cases to Andes"

---

## Scene 2 — 4-pane pipeline run (~2 min real → 12s at 10×)

```bash
asciinema rec -t "REDI SC26 - Pipeline" scene2_pipeline.cast \
    --command "bash benchmarks/demo/record_pipeline.sh"
```

**What to expect:**
| Pane | Case | Real time |
|------|------|-----------|
| Top-left | ClimaX / TaiESM1 | ~2 min |
| Top-right | OC22 / HydraGNN (500 structs) | ~30s |
| Bottom-left | OpenFold (10 proteins) | ~20s |
| Bottom-right | XGC1 ITER (1 timestep) | ~1 min |

After recording, play back at 10×:
```bash
asciinema play --speed 10 scene2_pipeline.cast
```

Convert to MP4 for iMovie:
```bash
agg --speed 10 --theme monokai scene2_pipeline.cast scene2_pipeline.mp4
# install: pip install agg  OR  cargo install agg
```

---

## Scene 3 — Output walkthrough (~30s, normal speed)

```bash
asciinema rec -t "REDI SC26 - Outputs" scene3_outputs.cast \
    --command "bash benchmarks/demo/show_outputs.sh"
```

Convert:
```bash
agg --theme monokai scene3_outputs.cast scene3_outputs.mp4
```

---

## Scene 4 — Closing slide (~8s)

Static image or iMovie title card:

```
REDI — Readiness Engine for Data Integration
─────────────────────────────────────────────
  Climate      ClimaX / TaiESM1      240,898 timesteps
  Materials    OC22 / HydraGNN       8.2M structures
  Life Sci.    OpenFold / OPS        4,994 proteins
  Fusion       XGC1 ITER             163M edges/graph

  All cases → Readiness Level 5 (FULLY AI-READY)
```

---

## iMovie assembly order

1. Scene 1 — Claude Code interaction (12s, normal)
2. Scene 2 — 4-pane pipeline (sped up 10×, ~12s)
3. Scene 3 — Output walkthrough (30s, normal or 1.5×)
4. Scene 4 — Closing slide (8s)

Add a subtle fade between scenes. Total: ~62–90s.
