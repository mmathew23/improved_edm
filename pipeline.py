from diffusers import DiffusionPipeline
from diffusers import DDIMScheduler
from diffusers import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
import math
import torch
from typing import Optional, Union, Tuple, List
from tqdm.auto import trange


class KarrasPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, method='euler'):
        super().__init__()

        # we ignore this, just having a scheduler for HF compatibility
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)
        self.trained_image_size = unet.config.sample_size
        self.method = method
        # Adjust noise levels based on what's supported by the network.
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7

    def step(self, x, t, num_inference_steps=50):
        if self.method == 'euler':
            return self.step_euler(x, t, num_inference_steps=num_inference_steps)
        elif self.method == 'rk':
            return self.step_rk(x, t, num_inference_steps=num_inference_steps)
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        to_device: Optional[torch.device] = None,
        second_order: bool = True,
        class_labels: Optional[torch.Tensor] = None,
        augmentation_labels: Optional[torch.Tensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        # image += 0.1 * torch.randn(
        #                 (image.shape[0], image.shape[1], 1, 1), device=image.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        # Time step discretization.
        step_indices = torch.arange(num_inference_steps, dtype=torch.float64, device=image.device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (num_inference_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]).to(dtype=torch.float16)
        t_steps[-1] = 1e-6

        image = image * t_steps[0]
        for t in self.progress_bar(range(num_inference_steps)):
            t_cur = t_steps[t]
            t_next = t_steps[t + 1]
            gamma = min(S_churn / num_inference_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = image + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(image)

            denoised = self.unet(x_hat, t_hat, class_labels=class_labels, augmentation_labels=augmentation_labels).sample
            d_cur = (x_hat - denoised) / t_hat
            image = x_hat + (t_next - t_hat) * d_cur

            if second_order and t < num_inference_steps - 1:
                denoised = self.unet(image, t_next, class_labels=class_labels, augmentation_labels=augmentation_labels).sample
                d_prime = (image - denoised) / t_next
                image = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "pil":
            image = image.cpu()
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            image = image.cpu()
            image = image.permute(0, 2, 3, 1).numpy()
        else:
            if to_device is not None:
                image = image.to(to_device)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @torch.no_grad()
    def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
        """DPM-Solver++(2M)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
        return x


    @torch.no_grad()
    def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
        """DPM-Solver++(2M) SDE."""

        if solver_type not in {'heun', 'midpoint'}:
            raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])

        old_denoised = None
        h_last = None

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

                if eta:
                    x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h
        return x