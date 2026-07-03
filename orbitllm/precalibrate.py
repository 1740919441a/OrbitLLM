from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _build_model(num_classes: int):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model, weights.transforms()

def _apply_patch_keep(images, keep_idx, grid: int):
    import torch

    b, c, h, w = images.shape
    ph = h // grid
    pw = w // grid
    out = torch.zeros_like(images)
    patch_idx = 0
    for gy in range(grid):
        for gx in range(grid):
            mask = (keep_idx == patch_idx).any(dim=1).view(b, 1, 1, 1)
            out[:, :, gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw] = torch.where(
                mask,
                images[:, :, gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw],
                out[:, :, gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw],
            )
            patch_idx += 1
    return out


def _mask_random(images, rho: float, grid: int, generator):
    import torch

    if rho >= 0.999:
        return images
    b = images.shape[0]
    patch_count = grid * grid
    keep = max(1, min(patch_count, int(round(rho * patch_count))))
    scores = torch.rand((b, patch_count), device=images.device, generator=generator)
    keep_idx = scores.topk(keep, dim=1).indices
    return _apply_patch_keep(images, keep_idx, grid)


def _mask_topk_variance(images, rho: float, grid: int = 4):
    import torch

    if rho >= 0.999:
        return images
    b, c, h, w = images.shape
    ph = h // grid
    pw = w // grid
    scores = []
    for gy in range(grid):
        for gx in range(grid):
            patch = images[:, :, gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw]
            scores.append(patch.var(dim=(1, 2, 3)))
    score_tensor = torch.stack(scores, dim=1)
    keep = max(1, min(grid * grid, int(round(rho * grid * grid))))
    keep_idx = score_tensor.topk(keep, dim=1).indices
    return _apply_patch_keep(images, keep_idx, grid)


def _mask_occlusion_saliency(model, preprocess, images, rho: float, grid: int):
    import torch

    if rho >= 0.999:
        return images
    b, c, h, w = images.shape
    ph = h // grid
    pw = w // grid
    with torch.no_grad():
        base_logits = model(preprocess(images))
        target = base_logits.argmax(dim=1)
        base_score = base_logits.gather(1, target.view(-1, 1)).squeeze(1)
        drops = []
        for gy in range(grid):
            for gx in range(grid):
                occluded = images.clone()
                occluded[:, :, gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw] = 0
                logits = model(preprocess(occluded))
                score = logits.gather(1, target.view(-1, 1)).squeeze(1)
                drops.append(base_score - score)
        score_tensor = torch.stack(drops, dim=1)
    keep = max(1, min(grid * grid, int(round(rho * grid * grid))))
    keep_idx = score_tensor.topk(keep, dim=1).indices
    return _apply_patch_keep(images, keep_idx, grid)


def _select_images(model, preprocess, images, rho: float, selector: str, grid: int, generator):
    if selector == "full" or rho >= 0.999:
        return images
    if selector == "random":
        return _mask_random(images, rho, grid, generator)
    if selector == "variance":
        return _mask_topk_variance(images, rho, grid)
    if selector == "occlusion":
        return _mask_occlusion_saliency(model, preprocess, images, rho, grid)
    raise ValueError(f"unknown selector: {selector}")


def _accuracy(model, loader, preprocess, device: str, rho: float = 1.0, selector: str = "variance", grid: int = 4, seed: int = 7) -> tuple[float, float]:
    import torch

    model.eval()
    correct = 0
    total = 0
    selector_times = []
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(seed)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            t0 = time.perf_counter()
            masked = _select_images(model, preprocess, images, rho, selector, grid, generator)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            selector_times.append((time.perf_counter() - t0) / max(images.shape[0], 1))
            x = preprocess(masked)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    acc = correct / max(total, 1)
    selector_ms = float(np.mean(selector_times) * 1000.0) if selector_times else 0.0
    return acc, selector_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate PRE retained-utility on a public remote-sensing image dataset.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="results/pre_real_calibration_metrics.csv")
    parser.add_argument("--max-train", type=int, default=2500)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--selectors", nargs="*", default=["random", "variance", "occlusion"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision.datasets import EuroSAT
    from torchvision.transforms import ToTensor

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    rng = torch.Generator().manual_seed(args.seed)
    dataset = EuroSAT(root=args.data_root, download=True, transform=ToTensor())
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_all, test_all = random_split(dataset, [train_len, test_len], generator=rng)
    train_idx = list(range(min(args.max_train, len(train_all))))
    test_idx = list(range(min(args.max_test, len(test_all))))
    train_ds = Subset(train_all, train_idx)
    test_ds = Subset(test_all, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model, preprocess = _build_model(num_classes=len(dataset.classes))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(preprocess(images))
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
        print(f"epoch={epoch + 1} loss={total_loss / max(len(train_ds), 1):.4f}")

    full_acc, full_selector_ms = _accuracy(model, test_loader, preprocess, args.device, rho=1.0, selector="full", grid=args.grid, seed=args.seed)
    rows = []
    for selector in args.selectors:
        for rho in [0.25, 0.50, 0.75, 1.00]:
            acc, selector_ms = _accuracy(model, test_loader, preprocess, args.device, rho=rho, selector=selector, grid=args.grid, seed=args.seed)
            rows.append(
                {
                    "dataset": "EuroSAT",
                    "model": "frozen_resnet18_linear_head",
                    "selector": selector,
                    "grid": args.grid,
                    "train_n": len(train_ds),
                    "test_n": len(test_ds),
                    "rho": rho,
                    "retained_patches": int(round(rho * args.grid * args.grid)),
                    "full_accuracy": full_acc,
                    "retained_accuracy": acc,
                    "eta_accuracy_ratio": min(acc / max(full_acc, 1e-9), 1.0),
                    "selector_ms_per_image": selector_ms if rho < 0.999 else full_selector_ms,
                    "seed": args.seed,
                }
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
