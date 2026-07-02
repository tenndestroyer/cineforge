# Model licenses

Cineforge's **own code is Apache-2.0**. The models it downloads carry **their own,
separate licenses** — and several are *not* freely commercial. This matters because
Cineforge is a tool other people build on.

`data/licenses.json` is the machine-checkable source of truth; the `LicenseGate` enforces
it. This page is the human-readable mirror.

## Modes

- **Safe mode (default)** — only models flagged `safe` (Apache/MIT, or a community license
  that is free for typical non-enterprise users) are selectable. This is the recommended
  default and what setup downloads.
- **Research mode** — unlocks the gated "best" picks, but each requires an **explicit
  in-app consent** before it downloads/runs, recorded per project.

## Safe-by-default (Apache / MIT / free-under-cap)

| Model | Subsystem | License |
|---|---|---|
| Wan 2.2 | video | Apache-2.0 (fully unrestricted) |
| Z-Image | image | Apache-2.0 |
| Qwen-Image-Edit | image | Apache-2.0 |
| Chatterbox | voice | MIT — embeds an **inaudible PerTh watermark** (disclosed) |
| Higgs Audio v2 | voice | Apache-2.0 |
| CosyVoice3 / Kokoro | voice | Apache-2.0 |
| ACE-Step 1.5 | music | MIT |
| HeartMuLa | music | Apache-2.0 |
| EchoMimicV3 / InfiniteTalk | lipsync | Apache-2.0 |
| LatentSync / MuseTalk | lipsync | Apache-2.0 / MIT |
| Stable Audio Open Small | sfx | Stability Community — **free under $1M annual revenue** |
| SeedVR2 / FlashVSR | enhance | Apache-2.0 |
| RIFE / GMFSS | enhance | MIT (cite hzwer for Practical-RIFE) |

## Gated (blocked in Safe mode; consent required in Research mode)

| Model | Subsystem | Catch |
|---|---|---|
| **LTX-2.3** | video | Lightricks Open Weights — **free under $10M ARR**, paid license above (2× liquidated damages). Also needs the gated `google/gemma-3-12b-it` text encoder. |
| **HunyuanVideo-1.5** | video | Tencent Community — **excludes EU / UK / South Korea**, 100M MAU cap. |
| **HunyuanVideo-Foley** | sfx | Tencent — same EU/UK/KR exclusion + 100M MAU. |
| **FLUX.2-dev** | image | **Non-commercial.** Gated repo (needs HF_TOKEN). |
| **IndexTTS-2** | voice | bilibili Index — non-commercial without written authorization. |

> ⚠️ These terms change. Verify against each model's actual `LICENSE` before commercial
> use. Mis-citing a model license in your own product is a real legal risk — Safe mode
> exists so you don't have to think about it.

## Ethics / provenance

- Voice & likeness cloning carries impersonation risk **regardless of license**. The GUI
  requires you to attest you have rights to any cloned voice/likeness.
- Some outputs are watermarked (Chatterbox PerTh). Factor that into redistribution.
