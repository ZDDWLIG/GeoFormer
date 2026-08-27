# GeoFormer: Geometry-Aware Transformer and its application to 5D First-Arrival Picking

Reproduction code for *GeoFormer*, a Geometry-Aware Transformer for automatic first-break picking in 5D seismic data under strong noise.

## Data

The repository ships a sample SEG-Y file for quick testing:

- `test_data/fig4_data.sgy` — sample 5D seismic gather (576 traces × 1501 samples).

Run inference on this file to reproduce the `fig4` results. Generated outputs (picks, predicted mask SEG-Y, and images) are written to `test_results/` (git-ignored).

## Model Weights

The best checkpoint (selected on the validation set of the Lalor seismic dataset) is hosted on Hugging Face:

> **https://huggingface.co/ZDDWLIG/GeoFormer**

File: `best_model.pth` (~195 MB). Save it to the `checkpoints/` directory (git-ignored):

```bash
mkdir -p checkpoints
curl -L -o checkpoints/best_model.pth \
  https://huggingface.co/ZDDWLIG/GeoFormer/resolve/main/best_model.pth
```

Alternatively, use `huggingface-cli` (installs only the model file):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download ZDDWLIG/GeoFormer best_model.pth --local-dir checkpoints
```

Or `git lfs` (clones the whole repository):

```bash
git clone https://huggingface.co/ZDDWLIG/GeoFormer checkpoints
```

## Environment Dependencies

Python 3.11+. Install dependencies with:

```bash
pip install -r requirements.txt
```

Pinned versions are listed in `requirements.txt`. For GPU inference, first install a CUDA-enabled PyTorch build following [pytorch.org](https://pytorch.org/get-started/locally/).

## Inference

```bash
python inference.py --segy test_data/fig4_data.sgy --model checkpoints/best_model.pth --output test_results/fig4_pred
```

| Argument | Description |
|----------|-------------|
| `--segy` | Input seismic SEG-Y file |
| `--model` | Model checkpoint path |
| `--output` | Output path stem — generates `{output}_picks.txt` and `{output}_pred_mask.sgy` |
| `--batch_size` | Batch size (default: 1) |
| `--device` | Device override, e.g. `cuda:0` or `cpu` (default: auto) |
| `--gpu` | GPU index shortcut, e.g. `--gpu 0` |

Model architecture, normalization parameters, and coordinate bounds are all auto-loaded from `config.json`.

## Plotting Results

After inference completes, generate visualization images with:

```bash
python plot_results.py -d test_results --sgy test_data/fig4_data.sgy --picks test_results/fig4_pred_picks.txt
```

Images are saved into the output directory (default: `test_results/`):

| Image | Description |
|-------|-------------|
| `seismic.png` | Raw seismic data displayed in grayscale |
| `pred.png` | Predicted first-break picks overlaid as red scatter points |

## Citation

If you use this work, please cite:

```bibtex
@misc{gao2026geoformergeometryawaretransformerapplication,
      title={GeoFormer: Geometry-Aware Transformer and its application to 5D First-Arrival Picking},
      author={Tianxiang Gao and Jianwei Ma},
      year={2026},
      eprint={2608.25668},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph},
      url={https://arxiv.org/abs/2608.25668},
}
```
