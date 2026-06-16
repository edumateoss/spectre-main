from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class RFUAVDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Any] = None,
        loader: Callable[[Union[str, Path]], Any] = default_loader,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(self.root)

        # Define classes based on top-level folders
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples: List[Path] = []
        self.targets: List[int] = []

        self.snr: List[Optional[int]] = []
        self.cmap: List[Optional[str]] = []
        self.stftp: List[Optional[int]] = []

        self.loader = loader

        self._load_dataset()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image_path = self.samples[index]
        target = self.targets[index]

        image = self.loader(image_path)
        image_np = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image_np)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.samples)

    def _load_dataset(self):
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = self.root / class_name

            images = [
                f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS
            ]

            if images:
                # Flat structure: class -> images
                for img_path in images:
                    self.samples.append(img_path)
                    self.targets.append(class_idx)
                    self.snr.append(None)  # No SNR metadata
                    self.cmap.append(None)  # No CMAP metadata
                    self.stftp.append(None)  # No STFTP metadata
            else:
                # Nested structure: class -> snr -> cmap -> images
                for snr_dir in class_dir.iterdir():
                    if not snr_dir.is_dir():
                        continue

                    try:
                        snr_value = int(snr_dir.name.replace("dB", ""))
                    except ValueError:
                        print(
                            f"Expected SNR directory names to be in format 'XXdB', got '{snr_dir.name}'"
                        )
                        continue

                    for cmap_dir in snr_dir.iterdir():
                        if not cmap_dir.is_dir():
                            continue

                        for stftp_dir in cmap_dir.iterdir():
                            if not stftp_dir.is_dir():
                                continue

                            for img_path in stftp_dir.rglob("*"):
                                if img_path.suffix.lower() in IMG_EXTENSIONS:
                                    self.samples.append(img_path)
                                    self.targets.append(class_idx)
                                    self.snr.append(snr_value)
                                    self.cmap.append(cmap_dir.name)
                                    self.stftp.append(int(stftp_dir.name))
