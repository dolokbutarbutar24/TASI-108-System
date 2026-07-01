from pathlib import Path
from ultralytics import YOLO
BASE_DIR   = Path("/home/jovyan/work/TASI-108")
DATA_PATH  = BASE_DIR / "dataset v4/Dataset_Split_3_nococo3/data.yaml"
TRIALS_DIR = BASE_DIR / "Resultv7/tuning2_s_nococo"
OUTPUT_DIR = BASE_DIR / "Resultv7/result_tuning2"
DEVICE     = 0
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "HEALTHY"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_lines = []
for i in range(20):
    trial_name = f"trial_{i:02d}"
    model_path = TRIALS_DIR / trial_name / "weights" / "best.pt"
    if not model_path.exists():
        print(f"[SKIP] {trial_name} — weights tidak ditemukan")
        continue
    print(f"\n{'='*60}")
    print(f" Evaluasi {trial_name}")
    print(f"{'='*60}")
    trial_out = OUTPUT_DIR / trial_name
    trial_out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    # === VAL SET ===
    val_results = model.val(
        data     = str(DATA_PATH),
        split    = "val",
        imgsz    = 640,
        batch    = 16,
        device   = DEVICE,
        workers  = 0,
        project  = str(trial_out),
        name     = "val",
        exist_ok = True,
        plots    = True,
        verbose  = True,
    )
    # === TEST SET ===
    test_results = model.val(
        data     = str(DATA_PATH),
        split    = "test",
        imgsz    = 640,
        batch    = 16,
        device   = DEVICE,
        workers  = 0,
        project  = str(trial_out),
        name     = "test",
        exist_ok = True,
        plots    = True,
        verbose  = True,
    )
    val_map   = val_results.box.map
    val_map50 = val_results.box.map50
    test_map  = test_results.box.map
    test_map50= test_results.box.map50
    print(f"\n--- {trial_name} HASIL ---")
    print(f"VAL  | mAP50={val_map50:.4f} | mAP50-95={val_map:.4f}")
    print(f"TEST | mAP50={test_map50:.4f} | mAP50-95={test_map:.4f}")
    print("Per Kelas (TEST):")
    for j, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10} mAP50={test_results.box.ap50[j]:.4f}  mAP50-95={test_results.box.ap[j]:.4f}")
    summary_lines.append(
        f"{trial_name} | VAL mAP50={val_map50:.4f} mAP50-95={val_map:.4f} | "
        f"TEST mAP50={test_map50:.4f} mAP50-95={test_map:.4f}"
    )
# Simpan ringkasan
summary_path = OUTPUT_DIR / "summary.txt"
with open(summary_path, "w") as f:
    f.write("RINGKASAN SEMUA TRIAL\n")
    f.write("="*80 + "\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write("\nBaseline: TEST mAP50=0.7976 mAP50-95=0.6067 | Precision=0.8084 | Recall=0.6825\n")
print(f"\n{'='*60}")
print(f" Selesai! Ringkasan disimpan di: {summary_path}")
print(f"{'='*60}")
