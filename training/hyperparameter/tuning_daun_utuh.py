import optuna
import logging
from pathlib import Path
from ultralytics import YOLO

BASE_DIR    = Path("/home/jovyan/work/TASI-108")
MODEL_PATH  = BASE_DIR / "yolov12s.pt"
DATA_PATH   = BASE_DIR / "dataset v4/Dataset_Split_2_nococo2/data.yaml"
OUTPUT_DIR  = BASE_DIR / "Resultv7/tuning1_s_nococo"
LOG_DIR     = BASE_DIR / "logsv7"
DEVICE      = 0
N_TRIALS    = 20
SEED        = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_path = LOG_DIR / "tuning1_s_nococo.log"
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s %(levelname)s %(message)s",
    handlers = [
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
 
    lr0          = trial.suggest_float("lr0",          7e-3,  2e-2,  log=True) 
    lrf          = trial.suggest_float("lrf",          0.005, 0.05,  log=True)  
    momentum     = trial.suggest_float("momentum",     0.92,  0.96)             
    weight_decay = trial.suggest_float("weight_decay", 3e-4,  8e-4,  log=True)  
    box = trial.suggest_float("box", 6.5,  8.5)   
    cls = trial.suggest_float("cls", 0.35, 0.65) 
    dfl = trial.suggest_float("dfl", 1.3,  1.8)  
    hsv_h      = trial.suggest_float("hsv_h",      0.005, 0.02)
    hsv_s      = trial.suggest_float("hsv_s",      0.5,   0.9)
    hsv_v      = trial.suggest_float("hsv_v",      0.3,   0.5)
    degrees    = trial.suggest_float("degrees",    0.0,   5.0)  
    translate  = trial.suggest_float("translate",  0.05,  0.15)
    scale      = trial.suggest_float("scale",      0.4,   0.7)  
    fliplr     = trial.suggest_float("fliplr",     0.4,   0.6)   
    mosaic     = trial.suggest_float("mosaic",     0.8,   1.0)  
    mixup      = trial.suggest_float("mixup",      0.0,   0.1)  
    copy_paste = trial.suggest_float("copy_paste", 0.0,   0.15)  
    warmup_epochs = trial.suggest_float("warmup_epochs", 2.0, 4.0)

    try:
        model   = YOLO(str(MODEL_PATH))
        results = model.train(
            data          = str(DATA_PATH),
            epochs        = 100,      
            batch         = 16,
            imgsz         = 640,
            optimizer     = "SGD",     
            lr0           = lr0,
            lrf           = lrf,
            momentum      = momentum,
            weight_decay  = weight_decay,
            warmup_epochs = warmup_epochs,
            warmup_momentum = 0.8,    
            warmup_bias_lr  = 0.1,    
            box           = box,
            cls           = cls,
            dfl           = dfl,
            hsv_h         = hsv_h,
            hsv_s         = hsv_s,
            hsv_v         = hsv_v,
            degrees       = degrees,
            translate     = translate,
            scale         = scale,
            fliplr        = fliplr,
            mosaic        = mosaic,
            mixup         = mixup,
            copy_paste    = copy_paste,
            close_mosaic  = 10,        
            patience      = 50,        
            save          = True,
            plots         = False,
            verbose       = False,     
            device        = DEVICE,
            workers       = 0,
            project       = str(OUTPUT_DIR),
            name          = f"trial_{trial.number:02d}",
            exist_ok      = True,
            seed          = SEED,
        )

        map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
        map50    = results.results_dict.get("metrics/mAP50(B)", 0.0)

        logger.info(
            f"Trial {trial.number:02d} | mAP50-95={map50_95:.4f} | mAP50={map50:.4f} | "
            f"lr0={lr0:.5f} | lrf={lrf:.4f} | box={box:.3f} | cls={cls:.3f} | "
            f"mosaic={mosaic:.3f} | degrees={degrees:.1f} | mixup={mixup:.3f}"
        )
        return map50_95

    except Exception as e:
        logger.error(f"Trial {trial.number:02d} gagal: {e}")
        return 0.0


sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(
    direction      = "maximize",
    sampler        = sampler,
    study_name     = "yolov12s_tuning1_s_nococo",  
    storage        = f"sqlite:///{OUTPUT_DIR}/tuning1_nococo.db",
    load_if_exists = True,
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_trial
logger.info(f"HPO selesai | Trial terbaik: #{best.number} | mAP50-95: {best.value:.4f}")

result_path = OUTPUT_DIR / "best_params.txt"
with open(result_path, "w") as f:
    f.write(f"Best trial  : #{best.number}\n")
    f.write(f"mAP50-95    : {best.value:.4f}\n\n")
    f.write("Best parameters:\n")
    for key, val in best.params.items():
        f.write(f"  {key:<20} : {val}\n")
    f.write("\nSemua trial (diurutkan dari terbaik):\n")
    for t in sorted(study.trials, key=lambda x: x.value or 0, reverse=True):
        f.write(f"  Trial {t.number:02d} | mAP50-95={t.value:.4f} | {t.params}\n")

logger.info(f"Best params disimpan: {result_path}")