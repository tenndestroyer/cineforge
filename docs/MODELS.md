# Updating models

The 2026 open-source model landscape moves fast. Cineforge is built so adopting a new SOTA
model is **data + one small file**, never a rewrite.

## To swap or add a model

1. **Add an adapter** in `src/cineforge/backends/<subsystem>/<model>.py`:

   ```python
   from ..base import CAP_IMG2VID, ComfyBackend
   from ...models.registry import register

   @register("mymodel", subsystem="video", license_id="mymodel")
   class MyModel(ComfyBackend):
       subsystem = "video"
       default_vram = 16.0
       required_nodes = ("ComfyUI-MyNode",)
       required_weights = ("video/mymodel/",)
       def capabilities(self):
           return {CAP_IMG2VID}
   ```
   Register it in that subsystem's `__init__.py` import line.

2. **Add a matrix row** in `data/model_matrix.json` for the tier(s) it fits
   (`safe` for Apache/MIT, `best` for a gated upgrade), with `repo`, `variant`, `quant`,
   `min_vram_gb`, `workflow`.

3. **Add a license row** in `data/licenses.json` (`safe: true` for Apache/MIT; otherwise set
   `non_commercial` / `arr_cap` / `mau_cap` / `excluded_territories` / `watermark` and
   `requires_ack`).

4. **Pin the revision.** Put an exact commit/revision hash on the download so installs are
   reproducible and a silent upstream change can't break you.

5. `cineforge doctor` verifies every matrix model has an adapter; `cineforge models` shows
   what resolves for a given GPU.

## Notes

- **Repo IDs in the shipped matrix are best-effort as of mid-2026** and should be verified
  before first download. The size-verified downloader rejects a wrong/gated pull rather than
  silently "succeeding".
- **Sampler presets are non-transferable across major model versions** (e.g. LTX-2 → LTX-2.3
  moved the CFG/steps sweet spot). Ship a per-model preset in `data/presets/` rather than one
  global default.
- Keep `model_matrix.json` and `licenses.json` in sync — a model in the matrix with no license
  row is treated as permissive (`ok`), which may not be what you want for a gated model.
