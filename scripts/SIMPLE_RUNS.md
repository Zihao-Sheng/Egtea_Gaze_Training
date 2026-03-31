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

4. Train a 1-epoch quick version that still connects to downstream testing:

```powershell
python scripts/run_default_training.py --quick-train --force-rebuild
```

5. Predict one clip:

```powershell
python scripts/predict_single_clip.py OP03-R06-GreekSalad-331160-332380-F007944-F007981.mp4
```

6. Predict one session folder:

```powershell
python scripts/predict_clip_folder.py OP03-R06-GreekSalad

python scripts/predict_clip_folder.py OP03-R06-GreekSalad --output-json outputs\demo_ready\predictions\greeksalad_top5.json

ollama pull qwen2.5:3b-instruct

python scripts/qwen_adjust_predictions.py --input-json outputs\demo_ready\predictions\greeksalad_top5.json --output-json outputs\demo_ready\predictions\greeksalad_top5_qwen.json --lag 3 --context-clips 16
```

Outputs are stored under:

- `outputs/demo_ready/default_pipeline/`

The bundle that connects training and inference lives at:

- `outputs/demo_ready/default_pipeline/bundle.json`
