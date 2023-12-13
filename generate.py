import os
import fire
from pipeline import KarrasPipeline
import torch
from torchvision.utils import save_image


def main(
        checkpoint: str,
        num_samples: int = 64,
        num_inference_steps: int = 35,
        device: str = 'cuda:0',
        save_dir: str = 'samples',
        tile_size: int = 4,
        seed: int = 24357234501,
):
    print(f'checkpoint {checkpoint}')
    print(f'num samples {num_samples}')
    print(f'device {device}')

    # get checkpoint last folder
    checkpoint = checkpoint[:-1] if checkpoint[-1] == '/' else checkpoint
    checkpoint_base = os.path.dirname(checkpoint)
    checkpoint_last_folder = checkpoint.split('/')[-1]
    save_dir = f'{checkpoint_base}/{save_dir}/{checkpoint_last_folder}'
    print(f'Saving in {save_dir}')

    pipeline = KarrasPipeline.from_pretrained(checkpoint)
    pipeline.to(device)
    images = pipeline(
        batch_size=num_samples,
        num_inference_steps=num_inference_steps,
        return_dict=True,
        output_type='tensor',
        to_device='cpu',
        generator=torch.manual_seed(seed),
    ).images

    os.makedirs(f'{save_dir}', exist_ok=True)
    for i, start in enumerate(range(0, len(images), tile_size*tile_size)):
        save_image(images[start:start+tile_size*tile_size], f'{save_dir}/sample_{i}.png', nrow=tile_size)


if __name__ == '__main__':
    fire.Fire(main)
