# The honest quality ceiling

Cineforge produces a **polished, stylized, strong-but-imperfect character-consistent AI
animated short** — **not** rigged Pixar/DreamWorks/Moonbug-grade animation, and it does
not auto-produce broadcast content. This is a limit of *local open-source AI in mid-2026*,
not of Cineforge's engineering. We surface it in the app on purpose.

**Why the ceiling exists at all.** The "Pixar-ish" AI demos people share come from
*frontier cloud* models (e.g. Google Veo). The instant you require *fully local + offline*,
you are on open weights (Wan 2.2 / LTX-2.3 / HunyuanVideo / Z-Image / Flux). Those are the
best that exist — and they are genuinely good — but the honest, sourced verdict is that
their motion, photorealism, and prompt adherence sit a real notch below the top cloud tier.
Cineforge is built to squeeze the maximum out of them and to drop in stronger models the
moment they ship.

## Per-subsystem reality

- **Video** — every top local model is fundamentally a 5–20 s single-clip generator.
  "Long-form" is always multi-shot stitching at the agent layer. Cross-shot consistency
  depends entirely on keyframe-lock / IC-LoRA discipline + the reject/retry auditor —
  expect visible face/prop **drift** over many shots and failures at **multi-character
  physical contact** (an unsolved problem even for paid frontier tools).
- **Image** — even FLUX.2 / Z-Image with LoRA + multi-reference read as *intentional
  mid-tier indie stylized 3D*, with residual identity drift and occasional hand/anatomy errors.
- **Voice** — near-commercial on short lines, but still autoregressive: budget a **5–15%
  per-line retry rate** even with best-of-N. Emotion control is a slider approximation, not
  performance direction.
- **Music** — ACE-Step approaches Suno v4.5–v5 on metrics but blind listeners rank it
  *below* v5: expect a slightly metallic timbre, flat vocal expressiveness, functional drums.
  Fine for short repetitive kids-show songs, not a session vocalist.
- **Lip-sync** — infers mouth shapes from audio+frame; non-verbal beats (a wink) won't
  appear unless separately driven. Stylized-face support is real but lightly benchmarked.
- **SFX/Foley** — plausible AI foley good for a temp mix / indie, not a human Foley artist's
  frame-accurate precision; expect the occasional wrong/hallucinated event.
- **Enhance** — a *polish* pass, not a *fix* pass: it cannot repair upstream generation
  flaws, and true depth-of-field/bokeh is immature (R&D; a gaussian-blur fallback is used).
- **Hardware** — the **8–12 GB tier is honestly short, softer, lower-res clips with
  minutes-per-clip** generation (e.g. SeedVR2 ≈ 6–8 min per 1 min of 1080p on 8 GB). Not the
  flagship demo-reel look.

## Long-form (15–20 minutes) & render time — the honest numbers

**Can Cineforge make a 15–20 minute video automatically? Architecturally, yes** — there
is no length cap. The Screenwriter writes the script in *chunks* (outline, then scene by
scene) so a ~300-shot feature stays inside a local model's context window; the pipeline
renders each shot and stitches them; and it runs **unattended with checkpoint/resume** —
a crash, OOM, or `Ctrl+C` resumes at the next un-rendered *shot*, not from zero. A
240-shot (~16 min) project drives through the whole pipeline in seconds of *orchestration*.

**But rendering it is the expensive part, and two things are true:**

1. **It is not fast.** On a mid-range laptop GPU (≈12 GB, "low" tier), Cineforge's own
   estimator puts a 16-minute film at **~70+ hours of continuous GPU time** — dominated
   by video generation and the upscale/restore pass. A 24 GB desktop card is several times
   faster; it is still hours, not minutes. This is unattended (start it, walk away, it
   checkpoints), but budget days on modest hardware.

2. **The current build plans but does not yet render pixels.** This is a **v0.1 scaffold**:
   the generation stages resolve the right model and report exactly what they *would*
   render, then stop with "install the backend(s)". Wiring each backend to its ComfyUI
   graph + downloading ~50–100 GB of weights (the v0.2→v0.5 roadmap) is what turns the plan
   into video. Until then, `run` produces the full script/shot-list/storyboard/render-plan,
   not an MP4.

And the ceiling above still holds: a 15–20 minute piece is exactly where cross-shot
character drift is most visible. Expect a *consistent-enough stylized short*, not a
rigged feature — Cineforge maximizes the local ceiling and is modular so stronger models
raise it, but it cannot make it Pixar.

## What Cineforge is genuinely good for

Storyboards and animatics; consistent-enough character shorts and social content; a full
first-pass creative package (script, shot list, voices, temp music, temp SFX) that a human
animator or editor then elevates. If your goal is broadcast-grade rigged animation, use
Cineforge for pre-production and hand the shot list + assets to a real animation pipeline.
