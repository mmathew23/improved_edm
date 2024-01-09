import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from diffusers.utils import make_image_grid
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from diffusers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from diffusers import EMAModel
from pipeline import KarrasPipeline
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import LoggerType
import os
from tqdm import tqdm
import shutil
import math
from datasets import load_dataset
from model import UNet2DModel
import numpy as np
import re


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset_length, seed=31129347):
        self.dataset_length = dataset_length
        self.seed = seed

    def __iter__(self):
        rnd = np.random.RandomState(self.seed)
        order = np.arange(self.dataset_length)
        rnd.shuffle(order)
        window = int(np.rint(order.size * 0.5))
        if window < 2:
            window = 3
        idx = 0
        while True:
            idx = idx % len(order)
            yield order[idx]
            j = (idx - rnd.randint(window)) % order.size
            order[idx], order[j] = order[j], order[idx]
            idx += 1


def get_total_steps(config):
    # round up, round since casting may round down due to fp precision
    total_steps = int(round(config.num_train_kimg * 1000 / (config.train_batch_size) + 0.5))
    return total_steps


def map_wrapper(func, from_key, to_key):
    def wrapper(example):
        example[to_key] = func(example[from_key])
        return example
    return wrapper


def get_inverse_sqrt_schedule(optimizer, num_warmup_steps, t_ref):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0 / math.sqrt(max(1.0, (current_step - num_warmup_steps) / t_ref))
    return LambdaLR(optimizer, lr_lambda)


def evaluate(config, step, pipeline):
    if 'num_class_embeds' in config.unet:
        labels = torch.arange(config.unet.num_class_embeds, device='cuda:0')[:config.val_batch_size]
        if labels.shape[0] < config.val_batch_size:
            labels = labels.repeat(config.val_batch_size//labels.shape[0] + 1)
            labels = labels[:config.val_batch_size]
    else:
        labels = None
    for i in range(4):
        images = pipeline(
            batch_size=config.val_batch_size,
            class_labels=labels,
            generator=torch.manual_seed(config.seed+i),
        ).images

        image_grid = make_image_grid(images, rows=4, cols=len(images)//4 + (1 if len(images) % 4 != 0 else 0))

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{step:04d}_{i:03d}.png")


def get_sigma(batch_size, P_mean, P_std, device):
    sigma = torch.randn([batch_size, 1, 1, 1], device=device)
    sigma = (sigma*P_std + P_mean).exp()
    return sigma


def get_sigma_weight(sigma, sigma_data):
    w = (sigma**2 + sigma_data**2) / (sigma*sigma_data)**2
    return w


def add_noise(sample, noise, sigma):
    noise *= sigma
    return sample+noise


def replace_grad_nans(model):
    # Iterate through all parameters
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Replace nan, inf, -inf in gradients with 0
            torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    assert 'gradient_accumulation_steps' in config and config.gradient_accumulation_steps >= 1
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=["wandb", LoggerType.TENSORBOARD],
        project_dir=os.path.join(config.output_dir, "logs"),
        kwargs_handlers=[ddp_kwargs],
        split_batches=True
    )
    is_distributed = accelerator.num_processes > 1

    if config.use_ema:
        ema = EMAModel(model.parameters(), 0.999, model_cls=UNet2DModel, model_config=model.config)

    start_step = 0
    latest_checkpoint = -1
    if 'resume' in config and config.resume is not None:
        print(f"Resuming from {config.resume}")
        torch_dict = torch.load(config.resume, map_location='cpu')
        optimizer.load_state_dict(torch_dict['optimizer'])
        lr_scheduler.load_state_dict(torch_dict['lr_scheduler'])
        if "scaler" in torch_dict:
            accelerator.scaler.load_state_dict(torch_dict["scaler"])
        model.load_state_dict(torch_dict['model'])
        latest_checkpoint = torch_dict['latest_checkpoint']
        start_step = torch_dict['step']
        if config.use_ema:
            parent_dir = os.path.dirname(config.resume)
            parent_dir_children = os.listdir(parent_dir)
            numbers = [int(re.match(r"ema_checkpoints_(\d+)", f).group(1)) for f in parent_dir_children if re.match(r"ema_checkpoints_(\d+)", f)]
            num = max(numbers)
            print(f"Using EMA checkpoint {num}")
            ema.from_pretrained(os.path.join(parent_dir, f'ema_checkpoints_{num}'), model_cls=UNet2DModel)

    if accelerator.is_main_process:
        # Create output directory if needed, and asserted for not None in train
        os.makedirs(config.output_dir, exist_ok=True)
        hydra_dir = os.path.join(HydraConfig.get().runtime.output_dir, '.hydra')
        print(f'copying from hydra dir {hydra_dir}')
        f_name = 'config.yaml'
        shutil.copy2(os.path.join(hydra_dir, f_name), os.path.join(config.output_dir, f_name))
        f_name = 'hydra.yaml'
        shutil.copy2(os.path.join(hydra_dir, f_name), os.path.join(config.output_dir, f_name))
        f_name = 'overrides.yaml'
        shutil.copy2(os.path.join(hydra_dir, f_name), os.path.join(config.output_dir, f_name))

        accelerator.init_trackers(
            config.model_name,
            config={
                'resume': config.resume if 'resume' in config else '',
                'batch_size': config.train_batch_size,
                'num_train_kimg': config.num_train_kimg,
                'lr': config.learning_rate,
                'dataset': config.data.dataset.path,
            },
        )


    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if config.use_ema:
        ema.to(accelerator.device)


    P_mean, P_std, sigma_data = config.training.P_mean, config.training.P_std, config.training.sigma_data
    total_steps = get_total_steps(config)
    progress_bar = tqdm(total=total_steps-start_step, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Train")
    train_iter = iter(train_dataloader)
    for step in range(start_step, total_steps):
        batch = next(train_iter)
        images = batch["images"]
        label = None
        if "label" in batch:
            label = batch["label"]

        noise = torch.randn_like(images)
        # Add noise to the clean images according to the noise magnitude at each timestep
        sigma = get_sigma(images.shape[0], P_mean, P_std, images.device)
        noisy_images = add_noise(images, noise, sigma)
        loss_w = get_sigma_weight(sigma, sigma_data)

        with accelerator.accumulate(model):
            # Predict the noise
            pred, u_sigma = model(noisy_images, sigma[:, 0, 0, 0], class_labels=label, return_dict=False, return_loss_mlp=True)

            loss = F.mse_loss(pred[0], images, reduction="none")
            loss = loss.mean(dim=(1,2,3))
            scaled_loss = loss_w[:, 0, 0, 0] * loss
            u_sigma = u_sigma[:, 0]
            scaled_loss_mlp = (scaled_loss / u_sigma.exp() + u_sigma).mean()
            accelerator.backward(scaled_loss_mlp)
            replace_grad_nans(model)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            if config.use_ema:
                ema.step(model.parameters())
            progress_bar.update(config.gradient_accumulation_steps)
            logs = {
                "loss": loss.detach().mean().item(),
                "scaled_loss": scaled_loss.detach().mean().item(),
                "mlp_loss": scaled_loss_mlp.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": step+1
            }
            p_logs = {
                "loss": f'{logs["loss"]:7.5f}',
                "scaled_loss": f'{logs["scaled_loss"]:6.4f}',
                "mlp_loss": f'{logs["mlp_loss"]:7.4f}',
                "lr": f'{logs["lr"]:.6f}',
                "step": step+1
            }
            progress_bar.set_postfix(**p_logs)
            accelerator.log(logs, step=step+1)

        if accelerator.is_main_process:
            save_image = (step + 1) % (config.save_image_steps*config.gradient_accumulation_steps) == 0 or step == total_steps - 1
            save_model = (step + 1) % (config.save_model_steps*config.gradient_accumulation_steps) == 0 or step == total_steps - 1
            if save_image or save_model:
                if is_distributed:
                    pipeline = KarrasPipeline(unet=accelerator.unwrap_model(model.module), scheduler=noise_scheduler)
                else:
                    pipeline = KarrasPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                if save_image:
                    if config.use_ema:
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    evaluate(config, step+1, pipeline)
                    if config.use_ema:
                        ema.restore(model.parameters())

                if save_model:
                    latest_checkpoint = step+1
                    pipeline.save_pretrained(os.path.join(config.output_dir, f"checkpoints_{step+1}"))
                    save_dict = dict(
                                    optimizer=optimizer.state_dict(),
                                    lr_scheduler=lr_scheduler.state_dict(),
                                    model=model.module.state_dict() if is_distributed else model.state_dict(),
                                    step=step+1,
                                    latest_checkpoint=latest_checkpoint,
                                    scaler=accelerator.scaler.state_dict() if config.mixed_precision else None,
                                )
                    torch.save(
                        save_dict,
                        os.path.join(config.output_dir, f"latest_training_state.pt")
                    )
                    torch.save(
                        save_dict,
                        os.path.join(config.output_dir, f"training_state_{step+1}.pt")
                    )
                    if config.use_ema:
                        ema_save_dir = os.path.join(config.output_dir, f"ema_checkpoints_{step+1}")
                        ema.save_pretrained(ema_save_dir)

    accelerator.end_training()


@hydra.main(version_base=None, config_path="", config_name='pixel_diffusion')
def train(config: DictConfig) -> None:
    assert config.output_dir is not None, "You need to specify an output directory"

    # Dataloader
    train_dataset = load_dataset(config.data.dataset.path, split=config.data.dataset.split)
    if 'map' in config.data.dataset:
        assert 'obj' in config.data.dataset.map, 'map object not specified'
        assert 'from_key' in config.data.dataset.map, 'map from_key not specified'
        assert 'to_key' in config.data.dataset.map, 'map to_key not specified'
        map_transform = hydra.utils.instantiate(config.data.dataset.map.obj)
        map_func = map_wrapper(map_transform, config.data.dataset.map.from_key, config.data.dataset.map.to_key)
        train_dataset = train_dataset.map(map_func)
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    def transform(examples):
        images = [transforms(image.convert("RGB")) for image in examples["image"]]
        if 'label' in examples:
            return {"images": images, "label": examples["label"]}
        return {"images": images}

    train_dataset.set_transform(transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=Sampler(len(train_dataset)), num_workers=config.data.dataloader.num_workers, pin_memory=True)

    # Model
    # _convert_ partial to save listconfigs as lists in unet so that it can be saved
    unet = hydra.utils.instantiate(config.unet, _convert_="partial")
    print(f'Parameter count: {sum([torch.numel(p) for p in unet.parameters()])}')
    noise_scheduler = hydra.utils.instantiate(config.noise_scheduler)

    # Optimizer and Scheduler
    optimizer = hydra.utils.instantiate(config.optimizer, params=unet.parameters())
    num_warmup_steps = config.lr_scheduler.num_warmup_steps if hasattr(config.lr_scheduler, 'num_warmup_steps') else 0
    if 'name' not in config.lr_scheduler or config.lr_scheduler.name == 'cosine':
        num_cycles = config.lr_scheduler.num_cycles if hasattr(config.lr_scheduler, 'num_cycles') else 0.5
        total_steps = get_total_steps(config)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, num_cycles=num_cycles)

    elif config.lr_scheduler.name == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    elif config.lr_scheduler.name == 'inverse_sqrt':
        t_ref = config.lr_scheduler.t_ref if hasattr(config.lr_scheduler, 't_ref') else 1
        lr_scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=num_warmup_steps, t_ref=t_ref)

    else:
        raise NotImplementedError("Only cosine, constant, and inverse_sqrt lr schedulers are supported")





    train_loop(config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)


if __name__ == '__main__':
    train()
