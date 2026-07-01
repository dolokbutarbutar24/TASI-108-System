from ultralytics import YOLO

BASE_DIR   = "/home/jovyan/work/TASI-108"
MODEL_PATH = f"{BASE_DIR}/yolov12s.pt"


DATA_EXP1  = f"{BASE_DIR}/dataset v4/Dataset_Split_2_nococo2/data.yaml"
OUTPUT_DIR = f"{BASE_DIR}/Resultv7/"

print("="*50)
print(" MEMULAI TRAINING BASELINE YOLOv12")
print("="*50)
print(f"  • MODEL  : {MODEL_PATH}")
print(f"  • DATA   : {DATA_EXP1}")
print(f"  • OUTPUT : {OUTPUT_DIR}")
print("="*50)


model = YOLO(MODEL_PATH)

results = model.train(
    data=DATA_EXP1, 
    device=1,        
    workers=0,       
    project=OUTPUT_DIR,
    name='Baseline_1_nococo',
    exist_ok=True,
)

print("\n Baseline selesai! Hasil tersimpan di:", f"{OUTPUT_DIR}Baseline")