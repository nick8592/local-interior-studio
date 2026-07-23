# Model Weights

This directory caches downloaded model weights. Files are **not** committed to git.

Models are auto-downloaded on first run via the Hugging Face `diffusers` cache:

- **InstructPix2Pix**: `timbrooks/instruct-pix2pix` (~4 GB)
- **SAM ViT-H**: `sam_vit_h_4b8939.pth` (~2.4 GB) — downloaded from Meta's hosting

To pre-download models before running the app:

```bash
python -c "from pipeline.edit import load_edit_model; load_edit_model()"
python -c "from pipeline.segment import download_sam_checkpoint; download_sam_checkpoint()"
```

On Docker, mount this directory as a volume so weights persist across rebuilds.
