# Simple Runs

These scripts are chained around the default demo-ready pipeline:

1. Prepare dataset:

```powershell
python scripts/setup_dataset.py
```

2. Prepare EgoVideo checkpoint:

```powershell
python scripts/setup_egovideo.py
```

3. Train the default strongest pipeline:

```powershell
python scripts/run_default_training.py
```

4. Predict one clip:

```powershell
python scripts/predict_single_clip.py OP03-R06-GreekSalad-331160-332380-F007944-F007981.mp4
```

5. Predict one session folder:

```powershell
python scripts/predict_clip_folder.py OP03-R06-GreekSalad
```

Outputs are stored under:

- `outputs/demo_ready/default_pipeline/`

The bundle that connects training and inference lives at:

- `outputs/demo_ready/default_pipeline/bundle.json`
