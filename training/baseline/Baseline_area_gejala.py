BASE_DIR   = "/home/jovyan/work/TASI-108"
MODEL_PATH = f"{BASE_DIR}/yolov12s.pt"
DATA_EXP1  = f"{BASE_DIR}/Dataset/Datasetv6-split/data.yaml"
OUTPUT_DIR = f"{BASE_DIR}/Resultv3/"

print(" Path siap")
print(f"  MODEL    : {MODEL_PATH}")
print(f"  DATA     : {DATA_EXP1}")
print(f"  OUTPUT   : {OUTPUT_DIR}")


from ultralytics import YOLO

model = YOLO(MODEL_PATH)

results = model.train(
      data=DATA_EXP1,
      epochs=100, 
      batch=16, 
      imgsz=640,
      scale=0.5,  
      mosaic=1.0,
      mixup=0.0, 
      copy_paste=0.0,  
      device=1,
      workers=0,


    project  = OUTPUT_DIR,
    name     = 'Baseline',
    exist_ok = True,
)
print("Baseline1 selesai!")

