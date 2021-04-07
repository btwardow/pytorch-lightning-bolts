from typing import Any, Optional, Sequence, Union

from torchvision.datasets.cifar import CIFAR100

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets.cifar10_dataset import TrialCIFAR10
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import CIFAR100
else:
    warn_missing_pkg('torchvision')  # pragma: no-cover
    CIFAR100 = None


class CIFAR100DataModule(VisionDataModule):
    name = "cifar100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        return 100

    def default_transforms(self):
        if self.normalize:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar10_normalization()])
        else:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return cf10_transforms