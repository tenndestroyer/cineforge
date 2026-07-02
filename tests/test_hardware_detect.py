from cineforge.hardware import detect_gpus, primary_gpu, select_backend
from cineforge.hardware.backend_select import BackendPlan
from cineforge.hardware.detect import GpuInfo


def test_detect_returns_at_least_one():
    gpus = detect_gpus()
    assert gpus and all(isinstance(g, GpuInfo) for g in gpus)
    assert primary_gpu(gpus) in gpus


def test_select_backend_shape():
    plan = select_backend()
    assert isinstance(plan, BackendPlan)
    assert plan.runtime in {"cuda", "rocm", "directml", "cpu"}
    assert plan.quant_pref in {"bf16", "fp8", "gguf", "fp16"}


def test_blackwell_routes_to_nightly():
    gpu = GpuInfo(vendor="nvidia", name="RTX 5090", vram_gb=32, compute_cap="12.0", backend_hint="cuda")
    plan = select_backend([gpu])
    assert plan.torch_channel == "cu128-nightly"
    assert any("Blackwell" in w for w in plan.warnings)


def test_ada_routes_to_stable():
    gpu = GpuInfo(vendor="nvidia", name="RTX 4090", vram_gb=24, compute_cap="8.9", backend_hint="cuda")
    plan = select_backend([gpu])
    assert plan.torch_channel == "cu128-stable"


def test_blackwell_detected_by_name_without_compute_cap():
    # nvidia-smi path yields no compute_cap; must still route Blackwell to nightly.
    gpu = GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                  vram_gb=11.9, compute_cap="", backend_hint="cuda")
    plan = select_backend([gpu])
    assert plan.torch_channel == "cu128-nightly"
