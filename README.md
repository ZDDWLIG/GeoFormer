# Reproduction: GeoFormer: Geometry-Aware Transformer and its application to 5D First-Arrival Picking with Strong Noise

## Files

| File | Purpose |
|------|---------|
| `inference.py` | Main entry point — CLI, I/O orchestration, sliding-window inference |
| `model.py` | Full model architecture (block, transformer, geom_transformer merged) |
| `data_utils.py` | SEG-Y I/O, normalization, coordinate bounds |
| `picking.py` | First-break picking algorithms (NPP, offset-guided NPP) |
| `config.json` | Model architecture, preprocessing, coordinate bounds defaults |

## Dependencies

```
torch numpy scipy segyio tqdm
```

## Inference

```bash
cd code
python inference.py --segy test_data/fig4_data.sgy --model checkpoints/best_model.pth --output test_results/fig4_pred
```

| Argument | Description |
|----------|-------------|
| `--segy` | Input seismic SEG-Y file |
| `--model` | Model checkpoint (use `best_model.pth` for full ckpt, or `best_model_inference.pth` for stripped 195MB version) |
| `--output` | Output path stem — generates `{output}_picks.txt` and `{output}_pred_mask.sgy` |
| `--batch_size` | Batch size (default: 1) |
| `--device` | Device override, e.g. `cuda:0` or `cpu` (default: auto) |
| `--gpu` | GPU index shortcut, e.g. `--gpu 0` |

Model architecture, normalization parameters, and coordinate bounds are all
auto-loaded from `config.json`.


## Plotting Results

After inference completes, generate visualization images with:

```bash
python plot_results.py -d test_results --sgy test_data/fig4_data.sgy --picks test_results/fig4_pred_picks.txt
```

Images are saved directly into `test_results/`:

| Image | Description |
|-------|-------------|
| `seismic.png` | Raw seismic data displayed in grayscale |
| `pred.png` | Predicted first-break picks overlaid as red scatter points |
