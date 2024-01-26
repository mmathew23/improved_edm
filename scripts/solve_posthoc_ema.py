import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ema_model import EMAModel
from model import UNet2DModel
from pipeline import KarrasPipeline
from diffusers import DDIMScheduler
import fire
from typing import List, Optional


def main(
    ema_checkpoint_path: str,
    target_sigma_rels: List[float] = [0.05, 0.75, 0.1],
    snapshot_t: Optional[int] = None,
):
    ema_model = EMAModel.from_pretrained(ema_checkpoint_path, model_cls=UNet2DModel)
    unet = ema_model.model_cls.from_config(ema_model.model_config)
    save_base_path = os.path.join(os.path.dirname(ema_checkpoint_path), 'posthoc_ema_checkpoints')
    os.makedirs(save_base_path, exist_ok=True)

    if snapshot_t is None:
        snapshot_t = ema_model.snapshot_t[-1]

    for target_sigma in target_sigma_rels:
        ema_model.copy_ema_profile(unet.parameters(), target_sigma, snapshot_t, ema_checkpoint_path)
        pipeline = KarrasPipeline(unet, DDIMScheduler())
        pipeline.save_pretrained(os.path.join(save_base_path, f'srel_{int(target_sigma*10000)}'))


if __name__ == '__main__':
    fire.Fire(main)
