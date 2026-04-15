"""Dataset implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Generic image dataset with class-subdir layout."""

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls, idx in self.class_to_idx.items():
            for ext in extensions:
                self.samples.extend(
                    (p, idx) for p in (self.root / cls).glob(f"**/*{ext}")
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
