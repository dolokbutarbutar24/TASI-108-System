from ultralytics import YOLO
import os

BASE_DIR   = "/home/jovyan/work/TASI-108"
MODEL_PATH = f"{BASE_DIR}/Resultv7/Baseline_1_nococo/weights/best.pt"
DATA_YAML  = f"{BASE_DIR}/dataset v4/Dataset_Split_2_nococo2/data.yaml"
OUTPUT_DIR = f"{BASE_DIR}/Resultv7/Evaluasi_Baseline_1"

model = YOLO(MODEL_PATH)
info = model.info(imgsz=640) 
print("-" * 50)
print(f"INFO MODEL:")
print(f"Model Path: {MODEL_PATH}")
print("-" * 50)

results = model.val(
    data=DATA_YAML,
    split='test',
    imgsz=640,
    batch=16,         
    device=1,
    workers=0,       
    project=OUTPUT_DIR,
    name='Evaluasi_baseline_1_nococo',
    exist_ok=True,
    plots=True 
)

print("\n" + "="*50)
print("HASIL EVALUASI FINAL:")
print("-" * 50)
# Perbaikan pemanggilan atribut box pada objek DetMetrics
print(f"mAP50 (Test)     : {results.box.map50:.4f}")
print(f"mAP50-95 (Test)  : {results.box.map:.4f}")
print(f"Precision (Test) : {results.box.mp:.4f}")
print(f"Recall (Test)    : {results.box.mr:.4f}")
print("-" * 50)
print(f"Hasil lengkap (Grafik & Gambar) ada di: {OUTPUT_DIR}/Evaluasi_baseline")
print("="*50)