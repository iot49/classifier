import torch
from fastai.data.all import (
    CategoryBlock,
    DataBlock,
    RandomSplitter,
)
from fastai.vision.all import (
    CropPad,
    ImageBlock,
    ImageDataLoaders,
    Rotate,
)


class R49DataLoaders(ImageDataLoaders):
    @classmethod
    def from_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        valid_pct: float=0.2,
        seed: int = 42,
        crop_size: int = 64,
        vocab: list[str] | None = None,
        **kwargs,  
    ) -> ImageDataLoaders:
        """Create R49DataLoaders from a torch.Dataset."""

        items = list(range(len(dataset))) 
        def get_x(idx: int): return dataset[idx][0] 
        def get_y(idx: int) -> str: return dataset[idx][1]  
        

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock(vocab=vocab, sort=False)),  
            splitter=RandomSplitter(valid_pct=valid_pct, seed=seed), 
            get_items=lambda _source: items,  
            get_x=get_x,
            get_y=get_y,
            item_tfms=CropPad(crop_size),    
            batch_tfms=[Rotate(max_deg=180, p=0.5)],
        )
        return cls.from_dblock(dblock, source=dataset, **kwargs)