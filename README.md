# antibiotic_learning

## Field Acceleration

The biological simulation now exposes optional Torch-based grid acceleration. To
enable it during training, set the following keys in your YAML config under
`environment`:

```yaml
environment:
	use_torch_fields: true        # Enable Torch/CUDA backend for grid ops
	field_device: "cuda"          # Or "cuda:1" / "cpu" depending on your setup
	enable_food_diffusion: null   # Leave null to auto-enable when acceleration works
```

When `use_torch_fields` is `true`, grid-heavy routines (quorum sensing,
biofilm EPS diffusion, optional nutrient smoothing) dispatch through Torch and
fall back to SciPy if CUDA is unavailable. Leave the flags `false` to retain the
original CPU-only behaviour.