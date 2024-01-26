from typing import Iterable, Union, Dict, Any, Optional
import torch
import contextlib
import copy
import numpy as np
import os


class PostHocEMASolver:
    def __init__():
        pass


# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        sigma_rels: Iterable[float] = [0.05],
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            sigma_rels (Iterable[float]): The sigma_rels for each parameter.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.

        """
        self.sigma_rels = sigma_rels
        self.get_gammas()
        parameters = list(parameters)
        self.shadow_params = [[p.clone().detach() for p in parameters] for _ in range(len(self.sigma_rels))]

        self.temp_stored_params = None

        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.optimization_step = 0
        self.snapshot_t = []
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config

    def get_gammas(self):
        self.gammas = [self.gamma_from_sigma(sigma_rel) for sigma_rel in self.sigma_rels]

    # from Karras Paper
    def gamma_from_sigma(self, sigma_rel):
        t = sigma_rel ** -2
        return np.roots([1, 7, 16-t, 12-t]).real.max()

    # from Karras Paper
    def p_dot_p(self, t_a, gamma_a, t_b, gamma_b):
        t_ratio = t_a / t_b
        t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
        t_max = np.maximum(t_a, t_b)
        num = (gamma_a+1)*(gamma_b+1)*t_ratio**t_exp
        den = (gamma_a+gamma_b+1)*t_max
        return num/den

    # from Karras Paper
    def solve_weights(self, t_i, gamma_i, t_r, gamma_r):
        def rv(x):
            return np.float64(x).reshape(-1, 1)

        def cv(x):
            return np.float64(x).reshape(1, -1)

        A = self.p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
        B = self.p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
        X = np.linalg.solve(A, B)
        return X

    @classmethod
    def from_pretrained(cls, path, model_cls, snapshot_t: Optional[int] = None) -> "EMAModel":
        if snapshot_t is None or snapshot_t <= 0:
            snapshot_t = str(max(map(int, os.listdir(path)))) # should be a list of ints as strings
            print(f'using snapshot_t {snapshot_t}')

        path = os.path.join(path, str(snapshot_t))
        model = model_cls.from_pretrained(path + '/sigma_rel_0')
        _, ema_kwargs, _h = model_cls.extract_init_dict(model.config, return_unused_kwargs=True)
        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config)
        ema_model.load_state_dict(ema_kwargs)
        for i in range(1, len(ema_model.sigma_rels)):
            model = model_cls.from_pretrained(path + f'/sigma_rel_{i}')
            ema_model.shadow_params.append([])
            ema_model.shadow_params[i] = [p.clone().detach() for p in model.parameters()]
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        self.snapshot_t.append(self.optimization_step)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        if path[-1] == "/":
            path = path[:-1]

        for i in range(len(self.sigma_rels)):
            print("Saving EMA model to", path + f"/sigma_rel_{i}")

            self.copy_to(model.parameters(), i)
            sigma_rel_path = os.path.join(path, f"{self.optimization_step}/sigma_rel_{i}")
            model.save_pretrained(sigma_rel_path)

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step)

        if step <= 0:
            return [0.0] * len(self.gammas)

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)
        cur_decay_values = [min(cur_decay_value, (1-1/step) ** g) for g in self.gammas]

        # make sure decay is not smaller than min_decay
        cur_decay_values = [max(decay_value, self.min_decay) for decay_value in cur_decay_values]
        return cur_decay_values

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = [1 - d for d in decay]

        context_manager = contextlib.nullcontext
        for i, shadow_params in enumerate(self.shadow_params):
            for s_param, param in zip(shadow_params, parameters):
                with context_manager():
                    if param.requires_grad:
                        s_param.sub_(one_minus_decay[i] * (s_param - param))
                    else:
                        s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter], rel_idx: int) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
            rel_idx: The index of sigma_rel parameters to use
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params[rel_idx], parameters):
            param.data.copy_(s_param.to(param.device).data)

    def copy_ema_profile(self, parameters: Iterable[torch.nn.Parameter], target_sigma_rel: float, target_t: float, ema_checkpoint_path: str) -> None:
        """
        Calculate a specific ema profile and copy paramters

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
            target_sigma_rel: float; the sigma_rel to use for the ema profile at corresponding target_t
            target_t: `float`; the target_t to use for the ema profile at corresponding sigma_rel
        """

        assert self.model_cls is not None
        assert self.model_config is not None
        parameters = list(parameters)
        for p in parameters:
            p.data.zero_()
        total_snapshots = len(self.snapshot_t)
        snapshot_t = np.array(self.snapshot_t*len(self.sigma_rels))
        sigma_rels = np.array([item for s in self.sigma_rels for item in [s] * total_snapshots])

        # weights stored as [len(snapshot_t), len(target_t)]
        weights = self.solve_weights(snapshot_t, sigma_rels, np.array([target_t]), np.array([target_sigma_rel]))
        for i in range(len(self.sigma_rels)):
            for snapshot_idx, snapshot in enumerate(self.snapshot_t):
                # load the ema snapshot
                model = self.model_cls.from_pretrained(f"{ema_checkpoint_path}/{snapshot}/sigma_rel_{i}")
                for param, s_param in zip(parameters, model.parameters()):
                    # param.data.copy_(s_param.to(param.device).data)
                    new_param_weighted = (s_param*weights[snapshot_idx + i*total_snapshots, 0]).to(param.device)
                    param.data.add_(new_param_weighted.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [[
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in shadow_params
        ] for shadow_params in self.shadow_params]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "sigma_rels": self.sigma_rels,
            "gammas": self.gammas,
            "shadow_params": self.shadow_params,
            "snapshot_t": self.snapshot_t,
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.sigma_rels = state_dict.get("sigma_rels", self.sigma_rels)
        if not isinstance(self.sigma_rels, (list, tuple)):
            raise ValueError("Invalid sigma_rels")

        self.gammas = state_dict.get("gammas", self.gammas)
        if not isinstance(self.gammas, (list, tuple)):
            raise ValueError("Invalid gammas")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")

        snapshot_t = state_dict.get("snapshot_t", None)
        if snapshot_t is not None:
            self.snapshot_t = snapshot_t

