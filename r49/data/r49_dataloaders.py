import torch
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    ImageDataLoaders,
    RandomSplitter,
    delegates,
)


class R49DataLoaders(ImageDataLoaders):
    @classmethod
    @delegates(ImageDataLoaders.from_dblock)
    def from_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        valid_pct,
        seed: int = 42,
        **kwargs,
    ) -> ImageDataLoaders:
        """Create R49DataLoaders from a torch.Dataset."""

        items = list(range(len(dataset)))

        def get_x(idx):
            data, _ = dataset[idx]
            return data

        def get_y(idx):
            _, label = dataset[idx]
            return label

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
            get_items=lambda _source: items,
            get_x=get_x,
            get_y=get_y,
        )
        return cls.from_dblock(dblock, source=dataset, **kwargs)
