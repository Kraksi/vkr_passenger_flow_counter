#!/usr/bin/env python3
"""
CLI-демонстрация счётчика пассажиропотока.

Запуск:
    python demo.py                               # авто-поиск тестовых видео
    python demo.py --video path/to/video.mp4    # одно видео
    python demo.py --dir data/stock_videos/     # все видео из папки
    python demo.py --video v.mp4 --gt-entries 4 --gt-exits 1  # с GT для MAPE

Выводит:
    - Клип-за-клипом: GT, Pred, MAPE, FPS, Latency
    - Итоговую сводную таблицу
"""
import argparse
import sys
import time
from pathlib import Path
import scipy.io as sio
import cv2
import numpy as np
from app.core.detector import Detector
from app.core.tracker import Tracker
import app.config as cfg

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Демо подсчёта пассажиропотока")
    p.add_argument("--video", type=Path, help="Путь к одному видеофайлу")
    p.add_argument("--dir", type=Path, help="Папка с видеофайлами")
    p.add_argument("--gt-entries", type=int, default=None, help="GT: кол-во входов")
    p.add_argument("--gt-exits",   type=int, default=None, help="GT: кол-во выходов")
    p.add_argument("--line", type=float, nargs=4,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   default=[0.0, 0.75, 1.0, 0.75],
                   help="Линия подсчёта в относительных координатах (default: горизонталь 75%%)")
    p.add_argument("--no-gpu", action="store_true", help="Использовать CPU вместо CUDA")
    return p.parse_args()


def find_test_clips(root: Path) -> list[dict]:
    """
    Возвращает список тестовых клипов с метаданными.
    Если стандартные видео найдены — использует предразмеченный GT.
    """
    clips = []

    known = [
        {
            "name": "store_entrance_UHD",
            "path": root / "data" / "stock_videos" / "uhd_30fps.mp4",
            "line": [0.293, 1.0, 0.293, 0.0],  
            "gt_entries": 4,
            "gt_exits": 1,
            "note": "2560×1440, магазин, вертикальная линия"
        },
        {
            "name": "bus_boarding_4K",
            "path": root / "data" / "stock_videos" / "13752261_2160_3840_30fps.mp4",
            "line": [0.0, 0.75, 1.0, 0.75],
            "gt_entries": None,
            "gt_exits": None,
            "note": "2160×3840, посадка в транспорт, 4K portrait"
        },
        {
            "name": "MOT20_01_test",
            "path": root / "data" / "test_mot20_01.mp4",
            "line": [0.0, 0.5, 1.0, 0.5],
            "gt_entries": None,
            "gt_exits": None,
            "note": "1920×1080, плотная сцена MOT20"
        },
    ]

    mall_gt_path = root / "data" / "mall" / "mall_dataset" / "mall_gt.mat"
    mall_video   = root / "data" / "mall" / "mall_video.mp4"
    if mall_video.exists() and mall_gt_path.exists():
        try:
            gt_mat = sio.loadmat(str(mall_gt_path))
            gt_counts = gt_mat["count"].flatten().astype(int)
            cap = cv2.VideoCapture(str(mall_video))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            seg_len = min(500, total_frames // 4)
            for i in range(4):
                start = i * seg_len
                end   = start + seg_len
                seg_crowd = int(gt_counts[start:end].mean())
                clips.append({
                    "name":        f"mall_segment_{i+1}",
                    "path":        mall_video,
                    "line":        [0.0, 0.5, 1.0, 0.5],
                    "start_frame": start,
                    "end_frame":   end,
                    "gt_entries":  None,
                    "gt_exits":    None,
                    "crowd_gt":    seg_crowd,
                    "note":        f"Mall Dataset сегм {i+1}/4 (crowd≈{seg_crowd}/кадр, {start}–{end})"
                })
        except Exception as e:
            print(f"[предупреждение] Mall Dataset не загружен: {e}")

    for clip in known:
        if clip["path"].exists():
            clips.append(clip)

    return clips


def run_clip(pipeline, clip: dict, device: str) -> dict:
    """Обработать один клип пайплайном, вернуть результат."""

    video_path = clip["path"]
    line = clip.get("line", [0.0, 0.75, 1.0, 0.75])
    start_frame = clip.get("start_frame", 0)
    end_frame   = clip.get("end_frame", None)

    pipeline.tracker.reset()
    pipeline.counter.reset()
    pipeline.counter.update_line(
        (line[0], line[1]), (line[2], line[3]), relative=True
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Не удалось открыть {video_path}"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline.counter.set_frame_size(w, h)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    total_latency = 0.0
    limit = (end_frame - start_frame) if end_frame else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame)
        frame_count += 1
        total_latency += result["latency_ms"]
        if limit and frame_count >= limit:
            break

    cap.release()

    avg_latency = total_latency / max(frame_count, 1)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "frames": frame_count,
        "avg_latency_ms": round(avg_latency, 1),
        "fps": round(fps, 1),
        "stats": pipeline.counter.stats,
    }


def compute_mape(pred_total: int, gt_entries: int | None, gt_exits: int | None) -> str:
    """Вычислить MAPE если GT доступен."""
    if gt_entries is None and gt_exits is None:
        return "—"
    gt_total = (gt_entries or 0) + (gt_exits or 0)
    if gt_total == 0:
        return "0.0%"
    mape = abs(pred_total - gt_total) / gt_total * 100
    return f"{mape:.1f}%"


def print_table(rows: list[dict]) -> None:
    col_w = [28, 6, 6, 8, 8, 10, 8]
    headers = ["Клип", "GT", "Pred", "MAPE", "FPS", "Latency", "Разм"]

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        if "error" in r:
            print(fmt.format(r["name"][:28], "ERR", "ERR", "—", "—", "—", "—"))
            continue
        stats = r["result"]["stats"]
        pred  = stats["entries"] + stats["exits"]
        gt_total = ((r["gt_entries"] or 0) + (r["gt_exits"] or 0))
        gt_str   = str(gt_total) if (r.get("gt_entries") is not None) else "—"
        mape     = compute_mape(pred, r.get("gt_entries"), r.get("gt_exits"))
        fps_str  = f"{r['result']['fps']:.0f}"
        lat_str  = f"{r['result']['avg_latency_ms']:.1f} мс"
        res_str  = r.get("resolution", "—")
        print(fmt.format(
            r["name"][:28],
            gt_str[:6],
            str(pred)[:6],
            mape[:8],
            fps_str[:8],
            lat_str[:10],
            res_str[:8],
        ))
    print(sep)


def main() -> None:
    args = parse_args()
    device = "cpu" if args.no_gpu else "cuda"

    if args.no_gpu:
        cfg.DETECTOR_DEVICE = "cpu"

    print("=" * 60)
    print("  Счётчик пассажиропотока — демонстрация MVP")
    print("=" * 60)
    print(f"  Устройство: {device.upper()}")

    print("\n[1/3] Инициализация пайплайна...")
    t0 = time.perf_counter()
    from app.core.pipeline import Pipeline
    pipeline = Pipeline()
    pipeline.initialize()
    print(f"      Готово за {(time.perf_counter()-t0)*1000:.0f} мс")
    print(f"      Модель: {pipeline.detector.model_path.name}")

    print("\n[2/3] Поиск тестовых видео...")
    if args.video:
        clips = [{
            "name":       args.video.stem[:28],
            "path":       args.video,
            "line":       args.line,
            "gt_entries": args.gt_entries,
            "gt_exits":   args.gt_exits,
            "note":       str(args.video),
        }]
    elif args.dir:
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        clips = [
            {"name": f.stem[:28], "path": f, "line": args.line,
             "gt_entries": None, "gt_exits": None, "note": str(f)}
            for f in sorted(args.dir.iterdir())
            if f.suffix.lower() in video_exts
        ]
    else:
        clips = find_test_clips(ROOT)

    if not clips:
        print("  Тестовые видео не найдены.")
        print("  Укажите --video или --dir, либо положите видео в data/stock_videos/")
        sys.exit(1)

    print(f"  Найдено клипов: {len(clips)}")

    print("\n[3/3] Обработка клипов...\n")
    results = []
    for i, clip in enumerate(clips, 1):
        name = clip["name"]
        path = clip["path"]
        print(f"  [{i}/{len(clips)}] {name} ...", end=" ", flush=True)

        if not path.exists():
            print("ПРОПУЩЕН (файл не найден)")
            results.append({"name": name, "error": "not found"})
            continue

        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        resolution = f"{w}×{h}"

        t_clip = time.perf_counter()
        result = run_clip(pipeline, clip, device)
        elapsed = time.perf_counter() - t_clip

        if "error" in result:
            print(f"ОШИБКА: {result['error']}")
            results.append({"name": name, "error": result["error"]})
            continue

        stats = result["stats"]
        pred  = stats["entries"] + stats["exits"]
        mape  = compute_mape(pred, clip.get("gt_entries"), clip.get("gt_exits"))
        print(f"готово ({result['frames']} кадров, {elapsed:.1f} с) — "
              f"IN:{stats['entries']} OUT:{stats['exits']} MAPE:{mape} FPS:{result['fps']}")

        results.append({
            "name":       name,
            "result":     result,
            "gt_entries": clip.get("gt_entries"),
            "gt_exits":   clip.get("gt_exits"),
            "resolution": resolution,
            "note":       clip.get("note", ""),
        })

    print("\n" + "=" * 60)
    print("  Итоговые результаты")
    print("=" * 60)
    print_table([r for r in results if "result" in r or "error" in r])

    valid = [r for r in results if "result" in r]
    if valid:
        fps_list = [r["result"]["fps"] for r in valid]
        lat_list = [r["result"]["avg_latency_ms"] for r in valid]
        print(f"\n  Среднее FPS:     {np.mean(fps_list):.1f}")
        print(f"  Средняя Latency: {np.mean(lat_list):.1f} мс")

        mape_vals = []
        for r in valid:
            gt_total = (r.get("gt_entries") or 0) + (r.get("gt_exits") or 0)
            if gt_total > 0:
                pred_total = r["result"]["stats"]["entries"] + r["result"]["stats"]["exits"]
                mape_vals.append(abs(pred_total - gt_total) / gt_total * 100)
        if mape_vals:
            print(f"  Средний MAPE:    {np.mean(mape_vals):.1f}% (по {len(mape_vals)} клипам с GT)")

    pipeline.shutdown()
    print("\n  Готово. Подробные метрики — в notebooks/04_metrics_acceptance.ipynb")


if __name__ == "__main__":
    main()
