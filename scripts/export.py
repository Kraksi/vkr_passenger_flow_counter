"""Экспорт продакшен-детектора в форматы инференса.

Один параметризованный скрипт для обоих продакшен-путей:
    onnx  — PT → ONNX (desktop: DETECTOR_BACKEND=onnx, CUDA/CPU onnxruntime)
    rknn  — ONNX → RKNN FP16 (edge RK3588: DETECTOR_BACKEND=rknn, NPU)
    all   — onnx, затем rknn

Запуск (desktop, основной venv):
    python scripts/export.py --format onnx

Запуск RKNN (только x86 + изолированный venv с rknn-toolkit2):
    uv venv .venv-rknn --python 3.11
    VIRTUAL_ENV=.venv-rknn uv pip install rknn-toolkit2==2.3.2
    .venv-rknn/bin/python scripts/export.py --format rknn"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

PT_MODEL = MODELS_DIR / "yolo11n_mot20_v2.pt"
ONNX_MODEL = MODELS_DIR / "yolo11n_mot20_v2.onnx"
RKNN_MODEL = MODELS_DIR / "yolo11n_mot20_v2.rknn"

IMGSZ = 640
OPSET = 17

MEAN = [0, 0, 0]
STD = [255, 255, 255]
QUANT_DATASET_SIZE = 100


# ─────────────────────────────────────────────────────────────────────────────
# PT → ONNX (desktop)
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(pt_path: Path, onnx_path: Path) -> None:
    """Экспортирует веса Ultralytics PT в ONNX (opset 17, simplify, static)."""
    if not pt_path.exists():
        print(f"PT-модель не найдена: {pt_path}")
        print("Сначала обучи детектор (notebooks/detector_training.ipynb).")
        sys.exit(1)

    model = YOLO(str(pt_path))
    out = Path(model.export(
        format="onnx", imgsz=IMGSZ, opset=OPSET,
        simplify=True, dynamic=False, device="cpu", half=False,
    ))
    if out != onnx_path:
        shutil.move(str(out), str(onnx_path))

    size_mb = onnx_path.stat().st_size / 1024**2
    print(f"ONNX сохранён: {onnx_path}  ({size_mb:.2f} MB, imgsz={IMGSZ}, opset={OPSET})")


# ─────────────────────────────────────────────────────────────────────────────
# ONNX → RKNN (edge RK3588)
# ─────────────────────────────────────────────────────────────────────────────
def _letterbox(img, size: int = IMGSZ):
    """Letterbox до квадрата + BGR→RGB"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def _build_calib_dataset(data_dir: Path, out_dir: Path, n: int = QUANT_DATASET_SIZE):
    """Калибровочные кадры для INT8: видео из data/ + добор из MOT20. → dataset.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _save(rgb):
        p = out_dir / f"calib_{len(saved):04d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        saved.append(p)

    for video_path in sorted(data_dir.rglob("*.mp4")):
        if len(saved) >= n:
            break
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // max(1, n // 2))
        idx = 0
        while cap.isOpened() and len(saved) < n:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                _save(_letterbox(frame))
            idx += 1
        cap.release()

    mot_dir = data_dir / "MOT20" / "train"
    if len(saved) < n and mot_dir.exists():
        for seq_dir in sorted(mot_dir.iterdir()):
            imgs = sorted((seq_dir / "img1").glob("*.jpg"))
            for img_path in imgs[:: max(1, len(imgs) // 20)]:
                if len(saved) >= n:
                    break
                img = cv2.imread(str(img_path))
                if img is not None:
                    _save(_letterbox(img))
            if len(saved) >= n:
                break

    if not saved:
        return None
    dataset_txt = out_dir / "dataset.txt"
    dataset_txt.write_text("\n".join(str(p.resolve()) for p in saved) + "\n")
    print(f"Калибровочных кадров: {len(saved)} → {dataset_txt}")
    return dataset_txt


def export_rknn(onnx_path: Path, rknn_path: Path, quantize: bool = False) -> None:
    """Строит RKNN-граф под RK3588 из ONNX. По умолчанию FP16 (рекомендуется для edge)."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 не установлен (нужен x86 + изолированный venv):")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"ONNX-модель не найдена: {onnx_path}. Сначала: --format onnx")
        sys.exit(1)

    rknn = RKNN(verbose=False)
    cfg = dict(mean_values=[MEAN], std_values=[STD], target_platform="rk3588", optimization_level=3)
    if quantize:
        cfg["quantized_algorithm"] = "mmse"
        cfg["quantized_method"] = "channel"
    if rknn.config(**cfg) != 0:
        raise RuntimeError("rknn.config() failed")

    if rknn.load_onnx(model=str(onnx_path), inputs=["images"], input_size_list=[[1, 3, IMGSZ, IMGSZ]]) != 0:
        raise RuntimeError("rknn.load_onnx() failed")

    dataset_txt = None
    do_quant = quantize
    if do_quant:
        dataset_txt = _build_calib_dataset(DATA_DIR, DATA_DIR / "rknn_calib")
        if dataset_txt is None:
            print("Калибровочные кадры не найдены — переход на FP16.")
            do_quant = False

    if rknn.build(do_quantization=do_quant, dataset=str(dataset_txt) if do_quant else None) != 0:
        raise RuntimeError("rknn.build() failed")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(rknn_path)) != 0:
        raise RuntimeError("rknn.export_rknn() failed")
    rknn.release()

    size_mb = rknn_path.stat().st_size / 1024**2
    print(f"RKNN сохранён: {rknn_path}  ({size_mb:.2f} MB, "
          f"{'INT8' if do_quant else 'FP16'})")
    print("Дальше — скопировать .rknn на RK3588 (см. EDGE_DEPLOY.md).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт детектора: ONNX (desktop) / RKNN (edge)")
    parser.add_argument("--format", choices=["onnx", "rknn", "all"], default="onnx")
    parser.add_argument("--pt", default=str(PT_MODEL))
    parser.add_argument("--onnx", default=str(ONNX_MODEL))
    parser.add_argument("--rknn", default=str(RKNN_MODEL))
    parser.add_argument("--quant", action="store_true", help="INT8 для RKNN (опыт; на edge FP16)")
    args = parser.parse_args()

    if args.format in ("onnx", "all"):
        export_onnx(Path(args.pt), Path(args.onnx))
    if args.format in ("rknn", "all"):
        export_rknn(Path(args.onnx), Path(args.rknn), quantize=args.quant)


if __name__ == "__main__":
    main()
