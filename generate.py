import os
import fire
from pipeline import KarrasPipeline
import torch
from torchvision.utils import save_image
from model import UNet2DModel


def main(
        checkpoint: str,
        num_samples: int = 64,
        num_inference_steps: int = 35,
        device: str = 'cuda:0',
        save_dir: str = 'samples',
        tile_size: int = 4,
        seed: int = 24357234501,
        eta: float = 0.0,
        ema_unet_path: str = None,
        class_labels: int = None,
):
    print(f'checkpoint {checkpoint}')
    print(f'num samples {num_samples}')
    print(f'device {device}')
    if class_labels:
        print(f'class labels {class_labels}')
        class_labels = torch.randint(0, class_labels, (num_samples,))

    # get checkpoint last folder
    checkpoint = checkpoint[:-1] if checkpoint[-1] == '/' else checkpoint
    checkpoint_base = os.path.dirname(checkpoint)
    checkpoint_last_folder = checkpoint.split('/')[-1]
    save_dir = f'{checkpoint_base}/{save_dir}/{checkpoint_last_folder}'
    print(f'Saving in {save_dir}')

    pipeline = KarrasPipeline.from_pretrained(checkpoint)
    if ema_unet_path is not None:
        ema_unet = UNet2DModel.from_pretrained(ema_unet_path)
        pipeline.unet = ema_unet
    pipeline.to(device)
    images = pipeline(
        batch_size=num_samples,
        num_inference_steps=num_inference_steps,
        return_dict=True,
        output_type='tensor',
        to_device='cpu',
        generator=torch.manual_seed(seed),
        eta=eta,
        class_labels=class_labels,
    ).images

    os.makedirs(f'{save_dir}', exist_ok=True)
    prefix = 'sample_'
    if ema_unet_path is not None:
        prefix = 'ema_sample_'
    for i, start in enumerate(range(0, len(images), tile_size*tile_size)):
        save_image(images[start:start+tile_size*tile_size], f'{save_dir}/{prefix}{i}.png', nrow=tile_size)


if __name__ == '__main__':
    fire.Fire(main)
