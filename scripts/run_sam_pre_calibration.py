from __future__ import annotations

import argparse
import io
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EUROSAT_LABELS = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway or road",
    "Industrial": "industrial area",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential area",
    "River": "river",
    "SeaLake": "sea or lake",
}


class PowerSampler:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    def __enter__(self) -> "PowerSampler":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._handle = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            import pynvml
        except Exception:
            return
        while not self._stop.is_set():
            if self._handle is not None:
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    self.samples.append((time.perf_counter(), float(power_w)))
                except Exception:
                    pass
            time.sleep(self.interval_s)

    def energy_j(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        total = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            total += 0.5 * (p0 + p1) * max(0.0, t1 - t0)
        return total

    def mean_power_w(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([p for _, p in self.samples]))

    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, self.samples[-1][0] - self.samples[0][0])


def measure_idle_power(seconds: float = 1.0) -> float:
    with PowerSampler(interval_s=0.05) as sampler:
        time.sleep(seconds)
    return sampler.mean_power_w()


def eurosat_items(root: Path, max_images: int, seed: int) -> list[tuple[Path, str]]:
    base = root / "2750"
    if not base.exists():
        raise FileNotFoundError(f"EuroSAT folder not found: {base}")
    items: list[tuple[Path, str]] = []
    for cls in sorted(EUROSAT_LABELS):
        cls_dir = base / cls
        for path in sorted(cls_dir.glob("*.jpg")):
            items.append((path, cls))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:max_images]


def load_mask_generator(args: argparse.Namespace):
    if args.backend == "mobile_sam":
        if args.mobile_sam_path:
            sys.path.insert(0, str(Path(args.mobile_sam_path).resolve()))
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

        model_type = "vit_t"
        checkpoint = args.mobile_checkpoint
    else:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model_type = "vit_b"
        checkpoint = args.sam_checkpoint

    if checkpoint is None:
        raise ValueError(f"checkpoint required for backend {args.backend}")
    import torch

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=args.device)
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=0,
        min_mask_region_area=args.min_region_area,
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    return generator


def filter_masks(masks: list[dict], image_area: int, min_ratio: float, max_ratio: float) -> list[dict]:
    filtered = []
    for mask in masks:
        area = float(mask.get("area", 0.0))
        ratio = area / max(image_area, 1)
        if min_ratio <= ratio <= max_ratio:
            filtered.append(mask)
    if not filtered and masks:
        filtered = [max(masks, key=lambda m: float(m.get("predicted_iou", 0.0)) + float(m.get("stability_score", 0.0)))]

    def score(m: dict) -> float:
        area_ratio = float(m.get("area", 0.0)) / max(image_area, 1)
        return float(m.get("predicted_iou", 0.0)) + float(m.get("stability_score", 0.0)) - 0.2 * math.sqrt(max(area_ratio, 0.0))

    return sorted(filtered, key=score, reverse=True)


def union_mask(masks: list[dict], limit: int | None) -> np.ndarray | None:
    selected = masks if limit is None else masks[:limit]
    if not selected:
        return None
    out = np.zeros_like(selected[0]["segmentation"], dtype=bool)
    for mask in selected:
        out |= mask["segmentation"].astype(bool)
    return out


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255)).resize(size, Image.Resampling.NEAREST)


def apply_mask(image: Image.Image, mask: np.ndarray | None) -> tuple[Image.Image, float, int]:
    if mask is None:
        return image.copy(), 1.0, 1
    small = resize_mask(mask, image.size)
    mask_arr = np.array(small) > 0
    img = np.array(image.convert("RGB"))
    out = np.zeros_like(img)
    out[mask_arr] = img[mask_arr]
    rho = float(mask_arr.mean())
    return Image.fromarray(out), rho, 0


@dataclass
class ClipClassifier:
    model: object
    processor: object
    text_features: object
    labels: list[str]
    device: str


def load_clip(model_path: str, labels: list[str], device: str) -> ClipClassifier:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    prompts = [f"a satellite photo of {EUROSAT_LABELS[label]}" for label in labels]
    text = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = _feature_tensor(model.get_text_features(**text))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return ClipClassifier(model=model, processor=processor, text_features=text_features, labels=labels, device=device)


def _feature_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def classify_images_with_scores(
    classifier: ClipClassifier,
    images: list[Image.Image],
    target_labels: list[str],
    batch_size: int,
) -> tuple[list[str], list[float]]:
    import torch

    preds: list[str] = []
    target_probs: list[float] = []
    label_to_index = {label: index for index, label in enumerate(classifier.labels)}
    classifier.model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            batch_targets = target_labels[start : start + batch_size]
            inputs = classifier.processor(images=batch, return_tensors="pt").to(classifier.device)
            image_features = _feature_tensor(classifier.model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = classifier.model.logit_scale.exp().clamp(max=100.0)
            logits = logit_scale * (image_features @ classifier.text_features.T)
            probs = logits.softmax(dim=1)
            idx = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            preds.extend([classifier.labels[i] for i in idx])
            target_idx = torch.tensor(
                [label_to_index[label] for label in batch_targets],
                device=probs.device,
                dtype=torch.long,
            )
            target_probs.extend(
                probs.gather(1, target_idx[:, None]).squeeze(1).detach().cpu().numpy().tolist()
            )
    return preds, [float(value) for value in target_probs]


def jpeg_size_bytes(image: Image.Image, quality: int) -> int:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        subsampling=0,
    )
    return len(buffer.getvalue())


def plot_curve(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    rows = summary[summary["selector"] != "full_image"].sort_values("rho_mean")
    ax.plot(rows["rho_mean"], rows["eta"], marker="o", lw=1.8, color="#1f5a9d", label=str(rows["backend"].iloc[0]) if not rows.empty else "SAM")
    ax.axhline(1.0, color="#777777", lw=0.8, ls="--")
    for _, row in rows.iterrows():
        ax.annotate(row["selector"].replace("sam_", ""), (row["rho_mean"], row["eta"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"Retained pixel ratio $\rho$")
    ax.set_ylabel(r"Utility retention $\eta$")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate PRE with SAM masks and CLIP/EuroSAT utility retention.")
    parser.add_argument("--backend", choices=["mobile_sam", "sam_vit_b"], default="sam_vit_b")
    parser.add_argument("--data-root", default="data/eurosat")
    parser.add_argument("--clip-model", default="/root/autodl-tmp/models/clip-vit-base-patch32")
    parser.add_argument("--sam-checkpoint", default="/root/autodl-tmp/asymsam/checkpoints/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--mobile-sam-path", default="/root/autodl-tmp/models/MobileSAM/MobileSAM-master")
    parser.add_argument("--mobile-checkpoint", default="/root/autodl-tmp/models/mobile_sam/mobile_sam.pt")
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--points-per-side", type=int, default=12)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.80)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--min-region-area", type=int, default=32)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--summary-output", default="results/sam_pre_calibration.csv")
    parser.add_argument("--per-image-output", default="results/sam_pre_per_image.csv")
    parser.add_argument("--profile-output", default="results/sam_pre_profile.csv")
    parser.add_argument("--figure-output", default="figures/sam_pre_eta_curve.png")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    labels = sorted(EUROSAT_LABELS)
    items = eurosat_items(Path(args.data_root), args.max_images, args.seed)
    classifier = load_clip(args.clip_model, labels, args.device)
    generator = load_mask_generator(args)
    idle_w = measure_idle_power(1.0) if args.device.startswith("cuda") else 0.0

    selectors = {
        "sam_top1": 1,
        "sam_top3": 3,
        "sam_top5": 5,
        "sam_all_filtered": None,
    }
    full_images: list[Image.Image] = []
    full_labels: list[str] = []
    masked_images: dict[str, list[Image.Image]] = {name: [] for name in selectors}
    per_image_rows: list[dict] = []
    sam_latencies: list[float] = []

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with PowerSampler(interval_s=0.02) as sampler:
        for idx, (path, label) in enumerate(items):
            image = Image.open(path).convert("RGB")
            source_jpg_bytes = int(path.stat().st_size)
            full_encoded_bytes = jpeg_size_bytes(image, args.jpeg_quality)
            sam_image = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            sam_np = np.array(sam_image)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            masks = generator.generate(sam_np)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            sam_latencies.append(latency_ms)
            filtered = filter_masks(masks, args.image_size * args.image_size, args.min_area_ratio, args.max_area_ratio)

            full_images.append(image)
            full_labels.append(label)
            for selector, limit in selectors.items():
                mask = union_mask(filtered, limit)
                masked, rho, fallback_full = apply_mask(image, mask)
                pre_encoded_bytes = jpeg_size_bytes(masked, args.jpeg_quality)
                masked_images[selector].append(masked)
                per_image_rows.append(
                    {
                        "backend": args.backend,
                        "image": str(path),
                        "label": label,
                        "selector": selector,
                        "rho": rho,
                        "source_jpg_bytes": source_jpg_bytes,
                        "full_encoded_bytes": full_encoded_bytes,
                        "pre_encoded_bytes": pre_encoded_bytes,
                        "rho_bytes_vs_source": pre_encoded_bytes / max(source_jpg_bytes, 1),
                        "rho_bytes_reencoded": pre_encoded_bytes / max(full_encoded_bytes, 1),
                        "jpeg_quality": args.jpeg_quality,
                        "mask_count_raw": len(masks),
                        "mask_count_filtered": len(filtered),
                        "fallback_full": fallback_full,
                        "sam_latency_ms": latency_ms,
                    }
                )
            if (idx + 1) % 25 == 0:
                print(f"processed {idx + 1}/{len(items)} images", flush=True)

    wall_energy_j = sampler.energy_j()
    duration_s = sampler.duration_s()
    active_energy_j = max(0.0, wall_energy_j - idle_w * duration_s)
    peak_mem_gb = float(torch.cuda.max_memory_allocated() / 1e9) if args.device.startswith("cuda") else 0.0

    full_preds, full_true_probs = classify_images_with_scores(
        classifier, full_images, full_labels, args.batch_size
    )
    full_acc = float(np.mean([p == y for p, y in zip(full_preds, full_labels)]))

    detail = pd.DataFrame(per_image_rows)
    detail["full_pred"] = ""
    detail["full_correct"] = 0
    detail["full_true_prob"] = 0.0
    summary_rows = [
        {
            "backend": args.backend,
            "selector": "full_image",
            "images": len(items),
            "rho_mean": 1.0,
            "rho_std": 0.0,
            "rho_bytes_vs_source_mean": 1.0,
            "rho_bytes_reencoded_mean": 1.0,
            "accuracy": full_acc,
            "eta": 1.0,
            "eta_prob_mean": 1.0,
            "eta_prob_median": 1.0,
            "sam_latency_ms_per_img": float(np.mean(sam_latencies)),
            "sam_wall_j_per_img": wall_energy_j / max(len(items), 1),
            "sam_active_j_per_img": active_energy_j / max(len(items), 1),
            "idle_w": idle_w,
            "peak_mem_gb": peak_mem_gb,
            "valid_mask_rate": 1.0,
            "seed": args.seed,
        }
    ]
    for selector, images in masked_images.items():
        preds, true_probs = classify_images_with_scores(
            classifier, images, full_labels, args.batch_size
        )
        acc = float(np.mean([p == y for p, y in zip(preds, full_labels)]))
        selector_index = detail.index[detail["selector"] == selector]
        detail.loc[selector_index, "full_pred"] = full_preds
        detail.loc[selector_index, "full_correct"] = [
            int(pred == label) for pred, label in zip(full_preds, full_labels)
        ]
        detail.loc[selector_index, "full_true_prob"] = full_true_probs
        detail.loc[selector_index, "pre_pred"] = preds
        detail.loc[selector_index, "pre_correct"] = [
            int(pred == label) for pred, label in zip(preds, full_labels)
        ]
        detail.loc[selector_index, "pre_true_prob"] = true_probs
        eta_probs = [
            min(float(pre_prob) / max(float(full_prob), 1e-9), 1.0)
            for pre_prob, full_prob in zip(true_probs, full_true_probs)
        ]
        detail.loc[selector_index, "eta_prob"] = eta_probs
        rows = detail.loc[selector_index]
        valid_rate = float(np.mean(rows["fallback_full"] == 0)) if not rows.empty else 0.0
        summary_rows.append(
            {
                "backend": args.backend,
                "selector": selector,
                "images": len(items),
                "rho_mean": float(rows["rho"].mean()),
                "rho_std": float(rows["rho"].std(ddof=1)) if len(rows) > 1 else 0.0,
                "rho_bytes_vs_source_mean": float(rows["rho_bytes_vs_source"].mean()),
                "rho_bytes_reencoded_mean": float(rows["rho_bytes_reencoded"].mean()),
                "accuracy": acc,
                "eta": min(acc / max(full_acc, 1e-9), 1.0),
                "eta_prob_mean": float(np.mean(eta_probs)),
                "eta_prob_median": float(np.median(eta_probs)),
                "sam_latency_ms_per_img": float(np.mean(sam_latencies)),
                "sam_wall_j_per_img": wall_energy_j / max(len(items), 1),
                "sam_active_j_per_img": active_energy_j / max(len(items), 1),
                "idle_w": idle_w,
                "peak_mem_gb": peak_mem_gb,
                "valid_mask_rate": valid_rate,
                "seed": args.seed,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    detail.to_csv(args.per_image_output, index=False)
    profile = summary[summary["selector"] == "sam_top3"].copy()
    if profile.empty:
        profile = summary[summary["selector"] != "full_image"].head(1).copy()
    profile = profile.rename(
        columns={
            "sam_active_j_per_img": "pre_energy_j_per_image",
            "sam_latency_ms_per_img": "pre_latency_ms_per_image",
            "rho_mean": "rho",
        }
    )
    profile.to_csv(args.profile_output, index=False)
    plot_curve(summary, Path(args.figure_output))
    print(f"wrote {summary_path}, {args.per_image_output}, {args.profile_output}, {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
