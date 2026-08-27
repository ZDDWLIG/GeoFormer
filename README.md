<h1 align="center">🌍 GeoFormer</h1>

<p align="center"><em>Geometry-Aware Transformer and its application to 5D First-Arrival Picking</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.25668"><img src="https://img.shields.io/badge/arXiv-2608.25668-B31B1B" alt="arXiv 2608.25668"></a>
  <a href="https://huggingface.co/ZDDWLIG/GeoFormer"><img src="https://img.shields.io/badge/HuggingFace-ZDDWLIG%2FGeoFormer-orange" alt="Hugging Face model"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB" alt="Python 3.11">
</p>

**GeoFormer** is a Geometry-Aware Transformer architecture specifically designed for prestack seismic data. This repository provides the reproduction code for the paper — model architecture, SEG-Y inference pipeline, first-break picking algorithms, and plotting utilities — with first-arrival picking presented as a representative application.

---

## 🧠 Method

We propose **GeoFormer**, a Geometry-Aware Transformer architecture specifically designed for prestack seismic data. Unlike the Vision Transformer, whose tokens are extracted from 2D patches and primarily encode visual patterns, GeoFormer explicitly incorporates acquisition geometry. Each seismic trace is represented by a **5D unit** consisting of the waveform and four source-receiver coordinates, from which two geometric attributes are derived: **offset** and **relative elevation** (receiver elevation minus source elevation). GeoFormer therefore performs **trace-level tokenization**, where each token combines the waveform with these geometric attributes.

To exploit these geometric attributes, GeoFormer introduces **three geometry injection mechanisms** operating at different levels of the Transformer pipeline:

- At the *token* level, **GeomMLP** replaces the classification token with a per-trace geometric representation derived from offset and elevation.
- At the *normalization* level, **GeomAdaLN** replaces uniform layer normalization with geometry-conditioned feature modulation.
- At the *attention* level, **GeomAttnBias** injects a parameter-free physical prior that geometrically proximate traces should attend more to each other.

We validate GeoFormer on **first-arrival picking**, a representative seismic processing task that relies heavily on acquisition geometry. Experiments on four field datasets demonstrate that GeoFormer outperforms Vision Transformer and two task-specific baselines. Crucially, GeoFormer maintains robust picking accuracy under strong noise because its trace-level tokens encode geometric attributes that provide a physical prior independent of waveform quality. Ablation studies confirm that GeomMLP, GeomAdaLN, and GeomAttnBias each contribute to GeoFormer's picking accuracy.

<p align="center">
  <img src="assets/model.png" alt="GeoFormer architecture" width="90%">
</p>

## 📊 Results

Qualitative comparison on the **Lalor** seismic dataset:

<table align="center">
  <tr>
    <td align="center"><b>(a) Ground truth label</b></td>
    <td align="center"><b>(b) GeoFormer (ours)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/lalor_label.png" alt="Ground truth label" width="95%"></td>
    <td align="center"><img src="assets/lalor_geoformer.png" alt="GeoFormer (ours)" width="95%"></td>
  </tr>
  <tr>
    <td align="center"><b>(c) Attention U-Net</b></td>
    <td align="center"><b>(d) ViT</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/lalor_atten_unet.png" alt="Attention U-Net" width="95%"></td>
    <td align="center"><img src="assets/lalor_vit.png" alt="ViT" width="95%"></td>
  </tr>
</table>

---

## 📦 Data

The repository ships a sample SEG-Y file for quick testing:

- `test_data/fig4_data.sgy` — sample 5D seismic gather.

Run inference on this file to reproduce the `fig4` results. Generated outputs (picks, predicted mask SEG-Y, and images) are written to `test_results/`.

## 🤗 Model Weights

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

## ⚙️ Environment Dependencies

Python 3.11+. Install dependencies with:

```bash
pip install -r requirements.txt
```

Pinned versions are listed in `requirements.txt`. For GPU inference, first install a CUDA-enabled PyTorch build following [pytorch.org](https://pytorch.org/get-started/locally/).

## 🚀 Inference

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

## 🎨 Plotting Results

After inference completes, generate visualization images with:

```bash
python plot_results.py -d test_results --sgy test_data/fig4_data.sgy --picks test_results/fig4_pred_picks.txt
```

Images are saved into the output directory (default: `test_results/`):

| Image | Description |
|-------|-------------|
| `seismic.png` | Seismic data |
| `pred.png` | Predicted first-break picks overlaid as red scatter points |

## 📖 Citation

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
