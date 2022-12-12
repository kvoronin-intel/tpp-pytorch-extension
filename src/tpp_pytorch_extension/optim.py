import math
from typing import Callable, Iterable, Tuple, List, Optional
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import required
from ._C import _optim as optim_cpp


class SGD(Optimizer):
    r"""Implements low precision stochastic gradient descent with extra state."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if not is_available():
            raise ValueError("Module function 'bf16_update' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum != 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay != 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SplitSGD, self).__init__(params, defaults)
        print("Using SplitSGD")

    def __setstate__(self, state):
        super(SplitSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dtype == torch.bfloat16:
                    param_state = self.state[p]
                    if "low_bits" not in param_state:
                        buf = param_state["low_bits"] = torch.zeros_like(
                            p.data, dtype=torch.short
                        )
                    else:
                        buf = param_state["low_bits"]
                # if weight_decay != 0:
                #     d_p = d_p.add(weight_decay, p.data)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf

                # p.data.add_(-group['lr'], d_p)
                if p.dtype == torch.bfloat16:
                    optim_cpp.bf16_split_add_(p.data, buf, d_p, -group["lr"])
                else:
                    if d_p.is_sparse:
                        optim_cpp.dense_sparse_add(p.data, d_p, -group["lr"])
                    else:
                        p.data.add_(d_p, alpha=-group["lr"])

        return loss


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                data = p.data
                # if grad.is_sparse:
                #     raise RuntimeError(
                #         "Adam does not support sparse gradients, please consider SparseAdam instead"
                #     )
                if hasattr(torch, "bfloat8") and p.data.dtype == torch.bfloat8:
                    data = data.to(torch.float)
                    grad = grad.to(torch.float)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(data)
                    # Lower bits for bf16 params
                    if p.data.dtype == torch.bfloat16:
                        state["low_bits"] = torch.zeros_like(p.data)
                    elif hasattr(torch, "bfloat8") and p.data.dtype == torch.bfloat8:
                        state["master_copy"] = data

                if hasattr(torch, "bfloat8") and p.data.dtype == torch.bfloat8:
                    data = state["master_copy"]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if data.dtype == torch.bfloat16:
                    low_bits = state["low_bits"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # # Decay the first and second moment running average coefficient
                # # In-place operations to update the averages at the same time
                # exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # # Just adding the square of the weights to the loss function is *not*
                # # the correct way of using L2 regularization/weight decay with Adam,
                # # since that will interact with the m and v parameters in strange ways.
                # #
                # # Instead we want to decay the weights in a manner that doesn't interact
                # # with the m/v parameters. This is equivalent to adding the square
                # # of the weights to the loss with plain (non-momentum) SGD.
                # # Add weight decay at the end (fixed version)
                # if group["weight_decay"] > 0.0:
                #     p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                if data.dtype == torch.bfloat16:
                    optim_cpp.fused_split_adamw(
                        data,
                        low_bits,
                        grad,  # contiguous() is called inside cpp code if not sparse
                        exp_avg,
                        exp_avg_sq,
                        beta1,
                        beta2,
                        step_size,
                        group["lr"],
                        group["weight_decay"],
                        group["eps"],
                    )
                else:
                    optim_cpp.fused_adamw(
                        data,
                        grad,  # contiguous() is called inside cpp code if not sparse
                        exp_avg,
                        exp_avg_sq,
                        beta1,
                        beta2,
                        step_size,
                        group["lr"],
                        group["weight_decay"],
                        group["eps"],
                    )
                    if hasattr(torch, "bfloat8") and p.data.dtype == torch.bfloat8:
                        p.data.copy_(state["master_copy"].to(torch.bfloat8))

        return loss


def clip_grad_norm_(parameters, max_norm, norm_type=2, grad_list=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    from torch._six import inf

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if not grad_list:
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        grads = [p.grad.detach() for p in parameters]
    else:
        grads = parameters
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == 2:
        return torch.tensor(optim_cpp.clip_grad_norm(grads, max_norm))

    if norm_type == inf:
        total_norm = max(grad.detach().abs().max().to(device) for grad in grads)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(grad.detach(), norm_type).to(device) for grad in grads]
            ),
            norm_type,
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.detach().mul_(clip_coef.to(grad.device))
    return total_norm


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        adam: bool = False,
        correct_bias: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.bf16 = p.grad.dtype == torch.bfloat16
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Lower bits for bf16 params
                    if p.data.dtype == torch.bfloat16:
                        state["low_bits"] = torch.zeros_like(p.data)
                    state["weight_norm"] = -1.0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if p.data.dtype == torch.bfloat16:
                    low_bits = state["low_bits"]
                beta1, beta2 = group["betas"]
                weight_norm = state["weight_norm"]

                state["step"] += 1

                if p.data.dtype == torch.bfloat16:
                    assert False, "BF16 LAMB optimizer not implemented yet!"
                    state["weight_norm"] = optim_cpp.fused_split_lamb(
                        p.data,
                        low_bits,
                        grad.contiguous(),
                        exp_avg,
                        exp_avg_sq,
                        beta1,
                        beta2,
                        weight_norm,
                        group["lr"],
                        group["weight_decay"],
                        group["eps"],
                    )
                else:
                    state["weight_norm"] = optim_cpp.fused_lamb(
                        p.data,
                        grad.contiguous(),
                        exp_avg,
                        exp_avg_sq,
                        beta1,
                        beta2,
                        weight_norm,
                        group["lr"],
                        group["weight_decay"],
                        group["eps"],
                    )

                #  # Decay the first and second moment running average coefficient
                #  if self.bf16:
                #      # m_t
                #      exp_avg.mul_(beta1).add_(1 - beta1, grad_fp32)
                #      # v_t
                #      exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad_fp32, grad_fp32)
                #  else:
                #      # m_t
                #      exp_avg.mul_(beta1).add_(1 - beta1, grad)
                #      # v_t
                #      exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                #  # Paper v3 does not use debiasing.
                #  # bias_correction1 = 1 - beta1 ** state['step']
                #  # bias_correction2 = 1 - beta2 ** state['step']
                #  # Apply bias to lr to avoid broadcast.
                #  step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                #  weight_norm = data_fp32.pow(2).sum().sqrt().clamp(0, 10) if self.bf16 \
                #      else p.data.pow(2).sum().sqrt().clamp(0, 10)

                #  adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                #  if group['weight_decay'] != 0:
                #      if self.bf16:
                #          adam_step.add_(group['weight_decay'], data_fp32)
                #      else:
                #          adam_step.add_(group['weight_decay'], p.data)

                #  adam_norm = adam_step.pow(2).sum().sqrt()
                #  if weight_norm == 0 or adam_norm == 0:
                #      trust_ratio = 1
                #  else:
                #      trust_ratio = weight_norm / adam_norm
                #  state['weight_norm'] = weight_norm
                #  state['adam_norm'] = adam_norm
                #  state['trust_ratio'] = trust_ratio
                #  if self.adam:
                #      trust_ratio = 1

                #  if self.bf16:
                #      data_fp32.add_(-step_size * trust_ratio, adam_step)
                #      p.data = data_fp32.to(torch.bfloat16)
                #  else:
                #      p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


class DistLamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        adam=False,
        bias_correction=True,
        block_size=1024,
        perform_allreduce=False,
        fused_param_norm=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        self.block_size = block_size
        self.bias_correction = bias_correction
        self.perform_allreduce = perform_allreduce
        self.distributed = (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        self.fused_param_norm = fused_param_norm
        self._acc_steps = 0
        self._one_time_setup_done = False
        super(DistLamb, self).__init__(params, defaults)

    class FlatBuffer:
        def __init__(self, param_list, group, dtype, block_size):
            self.param_list = param_list
            self.group = group
            self.dtype = dtype
            self.block_size = block_size
            p_i = 0
            total_size = 0
            size_array = []
            padded_size_array = []
            offset_array = [0]
            cur_offset = 0
            model_params = []
            block2param = []
            block_sizes = []
            for p in self.param_list:
                sz = p.data.numel()
                aligned_blocks = (sz + self.block_size - 1) // self.block_size
                aligned_sz = aligned_blocks * self.block_size
                block2param += [p_i] * aligned_blocks
                block_sizes += [self.block_size] * aligned_blocks
                block_sizes[-1] = (
                    self.block_size
                    if sz % self.block_size == 0
                    else sz % self.block_size
                )
                size_array.append(sz)
                padded_size_array.append(aligned_sz)
                cur_offset += aligned_sz
                offset_array.append(cur_offset)
                total_size += aligned_sz
                p_i += 1

            self._flat_w = torch.zeros([total_size], dtype=dtype)
            self._flat_g = torch.zeros([total_size], dtype=dtype)
            self._flat_m = torch.zeros([total_size], dtype=dtype)
            self._flat_v = torch.zeros([total_size], dtype=dtype)
            self._flat_u = torch.zeros([total_size], dtype=dtype)
            self._flat_ag = torch.zeros([total_size], dtype=dtype)
            if dtype == torch.bfloat16:
                self._flat_wl = torch.zeros([total_size], dtype=dtype)
            else:
                self._flat_wl = torch.zeros([0])
            self._param_sizes = torch.tensor(size_array, dtype=torch.long)
            self._param_sizes_padded = torch.tensor(padded_size_array, dtype=torch.long)
            self._offsets = torch.tensor(offset_array, dtype=torch.long)
            self._block2param = torch.tensor(block2param, dtype=torch.int)
            self._block_sizes = torch.tensor(block_sizes, dtype=torch.int)
            self._weight_norms = torch.zeros(
                [len(self.param_list) + 1], dtype=torch.double
            )
            self._update_norms = torch.zeros_like(self._weight_norms)
            for i, p in enumerate(self.param_list):
                s = offset_array[i]
                e = offset_array[i] + size_array[i]
                p.data = self._flat_w[s:e].view_as(p.data).copy_(p.data)
                if p.grad is None:
                    p.grad = self._flat_g[s:e].view_as(p.data)
                else:
                    p.grad = self._flat_g[s:e].view_as(p.data).copy_(p.grad.data)

    def _one_time_setup(self):
        if self._one_time_setup_done == True:
            return
        from collections import defaultdict

        self.flat_params = []
        for group in self.param_groups:
            model_params = defaultdict(list)
            for p in group["params"]:
                # torch.distributed.broadcast(p, 0)
                if not p.requires_grad:
                    continue
                dt = p.dtype
                model_params[dt].append(p)
            for dt, param_list in model_params.items():
                flat_buf = self.FlatBuffer(param_list, group, dt, self.block_size)
                self.flat_params.append(flat_buf)

        self._step = 0
        self._acc_steps = 0
        self._one_time_setup_done = True

    def clip_grad_norm_(self, max_norm, norm_type=2):
        if hasattr(self, "flat_params"):
            grads = [fp._flat_g for fp in self.flat_params]
        else:
            grads = [p.grad for group in self.param_groups for p in group["params"]]
        return clip_grad_norm_(grads, max_norm, norm_type, grad_list=True)

    def sync_params(self):
        if not self.distributed:
            return
        if hasattr(self, "flat_params"):
            for fp in self.flat_params:
                torch.distributed.broadcase(fp._flat_w.data, 0)
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    torch.distributed.broadcase(p.data, 0)

    def sync_grads(self):
        if not self.distributed:
            return
        acc_steps = self.merge_acc_grad(avg=False)
        world_size = torch.distributed.get_world_size() * acc_steps
        if hasattr(self, "flat_params"):
            for fp in self.flat_params:
                fp._flat_g.div_(world_size)
                # if torch.distributed.get_rank() == 0: print(f"{fp._flat_g.dtype} - {fp._flat_g.shape}")
                torch.distributed.all_reduce(fp._flat_g)
                # splts = fp._flat_g.split(2*1024*1024)
                # for s in splts:
                #    torch.distributed.all_reduce(s)
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad.data.div_(world_size)
                    torch.distributed.all_reduce(p.grad.data)

    def acc_and_zero_grad(self):
        self._one_time_setup()
        if hasattr(self, "flat_params"):
            for fp in self.flat_params:
                fp._flat_ag.add_(fp._flat_g)
                fp._flat_g.zero_()
            self._acc_steps += 1
        else:
            raise NotImplemented

    def merge_acc_grad(self, avg=True):
        if self._acc_steps == 0:
            return 1
        total_acc_steps = self._acc_steps + 1
        if hasattr(self, "flat_params"):
            for fp in self.flat_params:
                fp._flat_g.add_(fp._flat_ag)
                if avg:
                    fp._flat_g.div_(total_acc_steps)
                fp._flat_ag.zero_()
            self._acc_steps = 0
            return 1 if avg else total_acc_steps
        else:
            raise NotImplemented

    def zero_grad(self):
        if hasattr(self, "flat_params"):
            for fp in self.flat_params:
                fp._flat_g.zero_()
        else:
            super(DistLamb, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self._one_time_setup()
        self._step += 1
        self.state[self.param_groups[0]["params"][0]]["step"] = self._step
        self.merge_acc_grad()
        if self.perform_allreduce:
            self.sync_grads()

        for ii, fp in enumerate(self.flat_params):
            group = fp.group
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            optim_cpp.fused_lamb_v2(
                fp._flat_w,
                fp._flat_g,
                fp._flat_m,
                fp._flat_v,
                fp._flat_u,
                fp._flat_wl,
                fp._offsets,
                fp._block_sizes,
                fp._block2param,
                fp._weight_norms,
                fp._update_norms,
                weight_decay,
                beta1,
                beta2,
                lr,
                eps,
                fp.block_size,
                self._step,
                self.fused_param_norm,
            )
            # if weight_decay > 0.0 and torch.distributed.get_rank() < 2: print(f"wn: {fp._weight_norms[:5].sqrt()}  un: {fp._update_norms[:5].sqrt()}")
            # if weight_decay > 0.0: print(f"XXX {self._step:3d} NORM {ii}: wn: {fp._weight_norms[0].sqrt().item():.10f}  un: {fp._update_norms[0].sqrt().item():.10f}")

        return loss


class FlatBufferSGD:
    def __init__(self, param_list, group, dtype, block_size, with_f32_weight_copy = False, with_bf16_momentum = False, with_f32_momentum = False, with_split_weight = False): #, with_momentum = False):
        self.param_list = param_list
        self.group = group
        self.dtype = dtype
        self.block_size = block_size
        self.with_f32_weight_copy = with_f32_weight_copy
        self.with_bf16_momentum   = with_bf16_momentum
        self.with_f32_momentum    = with_f32_momentum
        self.with_split_weight    = with_split_weight

        # FIXME: Re-enable after finishing the split implementation
        #if with_f32_weight_copy and with_split_weight:
        #    print("Error: at most one of the flags with_f32_weight_copy/with_split_weight can be enabled")
        #    exit()

        #self.with_momentum = with_momentum
        p_i = 0
        total_size = 0
        size_array = []
        padded_size_array = []
        offset_array = [0]
        cur_offset = 0
        model_params = []
        block2param = []
        block_sizes = []
        for p in self.param_list:
            sz = p.data.numel()
            aligned_blocks = ((sz + self.block_size - 1) // self.block_size)
            aligned_sz = aligned_blocks * self.block_size
            block2param += [p_i] * aligned_blocks
            block_sizes += [self.block_size] * aligned_blocks
            block_sizes[-1] = self.block_size if sz % self.block_size == 0 else sz % self.block_size
            size_array.append(sz)
            padded_size_array.append(aligned_sz)
            cur_offset += aligned_sz
            offset_array.append(cur_offset)
            total_size += aligned_sz
            p_i += 1

        self._flat_w = torch.zeros([total_size], dtype=dtype)
        self._flat_g = torch.zeros([total_size], dtype=dtype)

        if dtype == torch.bfloat16 and self.with_f32_weight_copy:
            self._flat_f32_w = torch.zeros([total_size], dtype=torch.float32)
        else:
            self._flat_f32_w = None

        if self.with_bf16_momentum:
            self._flat_m = torch.zeros([total_size], dtype=torch.bfloat16)
        elif self.with_f32_momentum:
            self._flat_m = torch.zeros([total_size], dtype=torch.float32)
        else:
            self._flat_m = torch.zeros([0], dtype=dtype)

        #self._flat_m = torch.zeros([total_size], dtype=dtype)
        #self._flat_v = torch.zeros([total_size], dtype=dtype)
        #self._flat_u = torch.zeros([total_size], dtype=dtype)
        #self._flat_ag = torch.zeros([total_size], dtype=dtype)
        if dtype == torch.bfloat16 and self.with_split_weight:
            self._flat_wl = torch.zeros([total_size], dtype=dtype)
        else:
            self._flat_wl = torch.zeros([0], dtype=dtype)

        self._param_sizes = torch.tensor(size_array, dtype=torch.long)
        self._param_sizes_padded = torch.tensor(padded_size_array, dtype=torch.long)
        self._offsets = torch.tensor(offset_array, dtype=torch.long)
        self._block2param = torch.tensor(block2param, dtype=torch.int)
        self._block_sizes = torch.tensor(block_sizes, dtype=torch.int)
        #self._weight_norms = torch.zeros([len(self.param_list)+1], dtype=torch.double)
        #self._update_norms = torch.zeros_like(self._weight_norms)
        
        for i, p in enumerate(self.param_list):
            s = offset_array[i]
            e = offset_array[i] + size_array[i]
            p.data = self._flat_w[s : e].view_as(p.data).copy_(p.data)
            #print("_flat_w, dtype = ", self._flat_w, self._flat_w.dtype)
            if p.grad is None:
                p.grad = self._flat_g[s : e].view_as(p.data)
            else:
                p.grad = self._flat_g[s : e].view_as(p.data).copy_(p.grad.data)
            #print("_flat_g, dtype = ", self._flat_g, self._flat_g.dtype)
            if self._flat_f32_w is not None:
                self._flat_f32_w[s : e].view_as(p.data).copy_(p.to(torch.float32).data)
                #print("_flat_f32_w, dtype = ", self._flat_f32_w, self._flat_f32_w.dtype)
            # decided against doing it here
            #if self._flat_f32_m is not None:
            #    self._flat_f32_m[s : e].view_as(p.data).copy_(p.grad.to(torch.float32).data).add(self._flat_f32_w[s : e].view_as(p.data), alpha=weight_decay)

    def remap_onto(self, new_param_list):
        # Recalculating stats
        p_i = 0
        total_size = 0
        size_array = []
        padded_size_array = []
        offset_array = [0]
        cur_offset = 0
        model_params = []
        block2param = []
        block_sizes = []
        for p in new_param_list:
            if p.dtype != self.dtype:
                print("Error: type mismatch between new_param_list entry and the existing FlatBufferSGD instance in remap_onto()")
                exit()
            sz = p.data.numel()
            aligned_blocks = ((sz + self.block_size - 1) // self.block_size)
            aligned_sz = aligned_blocks * self.block_size
            block2param += [p_i] * aligned_blocks
            block_sizes += [self.block_size] * aligned_blocks
            block_sizes[-1] = self.block_size if sz % self.block_size == 0 else sz % self.block_size
            size_array.append(sz)
            padded_size_array.append(aligned_sz)
            cur_offset += aligned_sz
            offset_array.append(cur_offset)
            total_size += aligned_sz
            p_i += 1

        #dummy = torch.tensor(size_array,dtype=torch.long)
        #equal = torch.equal(dummy, self._param_sizes)
        #if equal is not True:
            #print("equal, dummy, self._param_sizes, diff = ", equal, dummy, self._param_sizes, dummy - self._param_sizes)

        # TODO: Could be more checks for rest of the arrays to be safer that flat buffer structure is not compromised
        if torch.equal(torch.tensor(size_array,dtype=torch.long), self._param_sizes) is not True:
            print("Error: size array for new param list is different from the existing one in remap_onto()!")
            #print("dbg: size_array, as tensor, type = ", size_array, torch.tensor(size_array,dtype=torch.long), type(size_array))
            #print("dbg: self._param_sizes, type = ", self._param_sizes, type(self._param_sizes), self._param_sizes.dtype)
            exit()

        #self._param_sizes = torch.tensor(size_array, dtype=torch.long)
        #self._param_sizes_padded = torch.tensor(padded_size_array, dtype=torch.long)
        #self._offsets = torch.tensor(offset_array, dtype=torch.long)
        #self._block2param = torch.tensor(block2param, dtype=torch.int)
        #self._block_sizes = torch.tensor(block_sizes, dtype=torch.int)

        #print("inside remap, self, self._flat_w data_ptr = ", self, self._flat_w.data_ptr())

        for i, p in enumerate(new_param_list):
            s = offset_array[i]
            e = offset_array[i] + size_array[i]
            #print("dbg: p.data before remap = ", p.data, p.data_ptr())
            # No need to copy here as p data should not have changed between
            # load_state_dict() [where flat_w was loaded] and step() [where remap is called]
            p.data = self._flat_w[s : e].view_as(p.data)
            #print("dbg: p.data after remap = ", p.data, p.data_ptr())
            #print("_flat_w, dtype = ", self._flat_w, self._flat_w.dtype)
            #p.grad = self._flat_g[s : e].view_as(p.data)
            # Need to copy data from p.grad into flat_g as it could be modified by the backward() call between
            # load_state_dict() [where flat_g was loaded] and step() [where remap is called]
            if p.grad is None:
                p.grad = self._flat_g[s : e].view_as(p.data)
            else:
                p.grad = self._flat_g[s : e].view_as(p.data).copy_(p.grad.data)

        #print("inside remap after, self._flat_w data_ptr = ", self._flat_w.data_ptr())

class SGD_fb_enhanced(torch.optim.SGD): #Optimizer):
    r"""Implements an SGD enhanced with a flat buffer (perf optimization)."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
    ):
        #if not is_available():
        #    raise ValueError("Module function 'bf16_update' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        #if momentum != 0.0:
        #    raise ValueError("Invalid momentum value: {}".format(momentum))
        #if weight_decay != 0.0:
        #    raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov:
            raise ValueError("Invalid nesterov value: {}".format(nesterov))
        if maximize:
            raise ValueError("Invalid maximize value: {}".format(maximize))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        #print("debug: lr = ", lr)
        #print("debug: defaults = ", defaults)
        #print("debug: params = ", params)
        #super(SplitSGD, self).__init__(params, defaults)
        super(torch.optim.SGD, self).__init__(params, defaults)
        #super(SGD_fb_enhanced, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        ##print("debug: type(self.param_groups) = ", type(self.param_groups))
        #print("debug: self.param_groups = ", self.param_groups)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            #print("debug: keys in group in init = ", group.keys())
        #def __init__(self, params, lr=required, momentum=0, dampening=0,
        #         weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None):

        print("Using torch.optim.SGD underneath SGD_fb_enhanced (with FlatBufferSGD)")

        self._one_time_setup_done = False
        print("Caution: setting block_size to 64 in SGD_fb_enhanced constructor")
        self.block_size = 64

    def __setstate__(self, state):
        #super(SplitSGD, self).__setstate__(state)
        #print("debug: calling setstate function of SGD_fb_enhanced")
        super(SGD_fb_enhanced, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    # Copied from https://gitlab.devtools.intel.com/pcl/pt/pcl_bert/-/blob/master/src/pcl_bert_ext/optim.py#L503
    class FlatBuffer:
        def __init__(self, param_list, group, dtype, block_size):
            self.param_list = param_list
            self.group = group
            self.dtype = dtype
            self.block_size = block_size
            p_i = 0
            total_size = 0
            size_array = []
            padded_size_array = []
            offset_array = [0]
            cur_offset = 0
            model_params = []
            block2param = []
            block_sizes = []
            for p in self.param_list:
                sz = p.data.numel()
                aligned_blocks = ((sz + self.block_size - 1) // self.block_size)
                aligned_sz = aligned_blocks * self.block_size
                block2param += [p_i] * aligned_blocks
                block_sizes += [self.block_size] * aligned_blocks
                block_sizes[-1] = self.block_size if sz % self.block_size == 0 else sz % self.block_size
                size_array.append(sz)
                padded_size_array.append(aligned_sz)
                cur_offset += aligned_sz
                offset_array.append(cur_offset)
                total_size += aligned_sz
                p_i += 1

            self._flat_w = torch.zeros([total_size], dtype=dtype)
            self._flat_g = torch.zeros([total_size], dtype=dtype)
            self._flat_m = torch.zeros([total_size], dtype=dtype)
            self._flat_v = torch.zeros([total_size], dtype=dtype)
            self._flat_u = torch.zeros([total_size], dtype=dtype)
            self._flat_ag = torch.zeros([total_size], dtype=dtype)
            if dtype == torch.bfloat16:
                self._flat_wl = torch.zeros([total_size], dtype=dtype)
            else:
                self._flat_wl = torch.zeros([0])
            self._param_sizes = torch.tensor(size_array, dtype=torch.long)
            self._param_sizes_padded = torch.tensor(padded_size_array, dtype=torch.long)
            self._offsets = torch.tensor(offset_array, dtype=torch.long)
            self._block2param = torch.tensor(block2param, dtype=torch.int)
            self._block_sizes = torch.tensor(block_sizes, dtype=torch.int)
            self._weight_norms = torch.zeros([len(self.param_list)+1], dtype=torch.double)
            self._update_norms = torch.zeros_like(self._weight_norms)
            for i, p in enumerate(self.param_list):
                s = offset_array[i]
                e = offset_array[i] + size_array[i]
                p.data = self._flat_w[s : e].view_as(p.data).copy_(p.data)
                if p.grad is None:
                    p.grad = self._flat_g[s : e].view_as(p.data)
                else:
                    p.grad = self._flat_g[s : e].view_as(p.data).copy_(p.grad.data)

    def _one_time_setup(self):
        if self._one_time_setup_done == True:
            return
        from collections import defaultdict
        self.flat_params = []
        for group in self.param_groups:
            #print("debug: group = ", group)
            model_params = defaultdict(list)
            for p in group['params']:
                #print("debug: in group params p  = ", p)
                #torch.distributed.broadcast(p, 0)
                if not p.requires_grad:
                    continue
                dt = p.dtype
                model_params[dt].append(p)
                #print("debug: appended to model_params (p, dt) = ", p, dt)
            for dt, param_list in model_params.items():
                #print("debug: (dt, param_list) = ", dt, param_list)
                #flat_buf = self.FlatBuffer(param_list, group, dt, self.block_size)
                flat_buf = FlatBufferSGD(param_list, group, dt, self.block_size)
                self.flat_params.append(flat_buf)

        self._step = 0
        self._acc_steps = 0
        self._one_time_setup_done = True

    def sync_params(self):
        if not self.distributed:
            return
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                torch.distributed.broadcase(fp._flat_w.data, 0)
        else:
            for group in self.param_groups:
                for p in group['params']:
                        torch.distributed.broadcase(p.data, 0)

    def sync_grads(self):
        if not self.distributed:
            return
        acc_steps = self.merge_acc_grad(avg=False)
        world_size = torch.distributed.get_world_size() * acc_steps
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                fp._flat_g.div_(world_size)
                #if torch.distributed.get_rank() == 0: print(f"{fp._flat_g.dtype} - {fp._flat_g.shape}")
                torch.distributed.all_reduce(fp._flat_g)
                #splts = fp._flat_g.split(2*1024*1024)
                #for s in splts:
                #    torch.distributed.all_reduce(s)
        else:
            for group in self.param_groups:
                for p in group['params']:
                        p.grad.data.div_(world_size)
                        torch.distributed.all_reduce(p.grad.data)

    def acc_and_zero_grad(self):
        self._one_time_setup()
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                fp._flat_ag.add_(fp._flat_g)
                fp._flat_g.zero_()
            self._acc_steps += 1
        else:
            raise NotImplemented

    def merge_acc_grad(self, avg=True):
        if self._acc_steps == 0:
            return 1
        total_acc_steps = self._acc_steps+1
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                fp._flat_g.add_(fp._flat_ag)
                if avg: fp._flat_g.div_(total_acc_steps)
                fp._flat_ag.zero_()
            self._acc_steps = 0
            return 1 if avg else total_acc_steps
        else:
            raise NotImplemented


    def zero_grad(self):
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                fp._flat_g.zero_()
        else:
            super(SGD_fb_enhanced, self).zero_grad()

    def sgd(self,
            params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
            # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
            has_sparse_grad: bool = None,
            foreach: bool = None,
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            maximize: bool):

        r"""Functional API that performs SGD algorithm computation.

        See :class:`~torch.optim.SGD` for details.
        """

        if foreach is None:
            # Placeholder for more complex foreach logic to be added when value is not set
            foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if foreach and not torch.jit.is_scripting():
            func = self._multi_tensor_sgd
        else:
            func = self._single_tensor_sgd

        func(params,
             d_p_list,
             momentum_buffer_list,
             weight_decay=weight_decay,
             momentum=momentum,
             lr=lr,
             dampening=dampening,
             nesterov=nesterov,
             has_sparse_grad=has_sparse_grad,
             maximize=maximize)


    def _single_tensor_sgd(self,
                           params: List[Tensor],
                           d_p_list: List[Tensor],
                           momentum_buffer_list: List[Optional[Tensor]],
                           *,
                           weight_decay: float,
                           momentum: float,
                           lr: float,
                           dampening: float,
                           nesterov: bool,
                           maximize: bool,
                           has_sparse_grad: bool):

        #print("debug: params in single_tensor_sgd = ", params)
        #print("debug: momentum in single_tensor_sgd = ", momentum)
        for i, param in enumerate(params):

            d_p = d_p_list[i]
            #print("debug: i, param, d_p = ", i, param, d_p)
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            alpha = lr if maximize else -lr
            param.add_(d_p, alpha=alpha)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self._one_time_setup()
        self._step += 1
        self.state[self.param_groups[0]['params'][0]]['step'] = self._step
        self.merge_acc_grad()
        #print("Fixme here (is this needed?)")
        #if self.perform_allreduce:
        #    self.sync_grads()


        for ii, fp in enumerate(self.flat_params):
            #print("debug: ii, fp = ", ii ,fp)
            group = fp.group
            #beta1, beta2 = group['betas']
            lr = group['lr']
            #eps = group['eps']
            weight_decay = group['weight_decay']
            #optim_cpp.fused_lamb_v2(fp._flat_w, fp._flat_g, fp._flat_m, fp._flat_v, fp._flat_u, fp._flat_wl, fp._offsets, fp._block_sizes, fp._block2param, fp._weight_norms, fp._update_norms, weight_decay, beta1, beta2, lr, eps, fp.block_size, self._step, self.fused_param_norm)
            #if weight_decay > 0.0 and torch.distributed.get_rank() < 2: print(f"wn: {fp._weight_norms[:5].sqrt()}  un: {fp._update_norms[:5].sqrt()}")
            #if weight_decay > 0.0: print(f"XXX {self._step:3d} NORM {ii}: wn: {fp._weight_norms[0].sqrt().item():.10f}  un: {fp._update_norms[0].sqrt().item():.10f}")

        #print("debug: param_groups = ", self.param_groups)

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            #print("debug: type of group = ", type(group))
            #print("debug: keys of group = ", group.keys())
            #print("debug: d_p_list      = ", d_p_list)
            #print("debug: momentum_buffer_list = ", momentum_buffer_list)

            #print("debug: group weight_decay = ", group['weight_decay'])
            #print("debug: group momentum = ", group['momentum'])
            #print("debug: group dampening = ", group['dampening'])
            #print("debug: group nesterov = ", group['nesterov'])
            #print("debug: group maximize = ", group['maximize'])
            #print("debug: group foreach  = ", group['foreach'])
            #print("debug: group = ", group)

            self.sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

# inspired by https://github.com/intel-sandbox/pcl-pytorch-extension/blob/loop_opt/examples/bert/pretrain_mlperf/lamb.py#L127
class SGD_bf16_enhanced(torch.optim.SGD):
    r"""Implements an SGD tailored for bf16 training (with an extra fp32 copy)."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        #if not is_available():
        #    raise ValueError("Module function 'bf16_update' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        #if momentum != 0.0:
        #    raise ValueError("Invalid momentum value: {}".format(momentum))
        #if weight_decay != 0.0:
        #    raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov:
            raise ValueError("Invalid nesterov value: {}".format(nesterov))
        #if maximize:
        #    raise ValueError("Invalid maximize value: {}".format(maximize))

        #print("type(lr) = " , type(lr))

        # If inheriting from the Optimizer
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        #print("debug: lr = ", lr)
        #print("debug: defaults = ", defaults)
        #print("debug: params = ", params)
        #super(SplitSGD, self).__init__(params, defaults)

        super(torch.optim.SGD, self).__init__(params, defaults)
        """
        # If inheriting from the SGD
        # for PT SGD:
        #def __init__(self, params, lr=required, momentum=0, dampening=0,
        #         weight_decay=0, nesterov=False):
        super(SGD_bf16_enhanced, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        ##print("debug: type(self.param_groups) = ", type(self.param_groups))
        #print("debug: self.param_groups = ", self.param_groups)
        """
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
        """
        #print("debug: keys in group in init = ", group.keys())
        #def __init__(self, params, lr=required, momentum=0, dampening=0,
        #         weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None):

        print("Using torch.optim.SGD underneath SGD_bf16_enhanced")

    # Not really well tested (affects saving/loading from checkpoints)
    def __setstate__(self, state):
        #super(SplitSGD, self).__setstate__(state)
        super(SGD_bf16_enhanced, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
        state_values = list(self.state.values())
        data_fp32_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['data_fp32'])
        #print("dbg: SGD_bf16_enhanced: state_values        = ", state_values)
        #print("dbg: SGD_bf16_enhanced: data_fp32_is_tensor = ", data_fp32_is_tensor)
        if not data_fp32_is_tensor:
            for s in state_values:
                #print("dbg: SGD_bf16_enhanced: s loop, s = ", s)
                s['data_fp32'] = torch.tensor(float(s['data_fp32']))
    

    """
    def sync_params(self):
        if not self.distributed:
            return
        for group in self.param_groups:
            for p in group["params"]:
                torch.distributed.broadcase(p.data, 0)

    def sync_grads(self):
        if not self.distributed:
            return
        world_size = torch.distributed.get_world_size()
        for group in self.param_groups:
            for p in group["params"]:
                p.grad.data.div_(world_size)
                torch.distributed.all_reduce(p.grad.data)

    def zero_grad(self):
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                fp._flat_g.zero_()
        else:
            super(SGD_bf16_enhanced, self).zero_grad()
    """

    def state_dict_dbg(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0
        start_index_dbg = 0

        def pack_group(group):
            nonlocal start_index
            nonlocal start_index_dbg
            packed = {k: v for k, v in group.items() if k != 'params'}
            print("debug: group params = ", group['params'])
            for i, p in enumerate(group['params'], start_index_dbg):
                print("i p id(p) is_in = ", i, p, id(p), 1 if id(p) in param_mappings else 0)
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            print("debug: packed, param_mappings = ", packed, param_mappings)
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        print("debug: self.param_groups = ", self.param_groups)
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        print("state keys = ", self.state.keys())
        print("param_mappings = ", param_mappings)
        for k, v in self.state.items():
            print("type(k) type(v) id(k) = ", type(k), type(v), id(k))
            if isinstance(k, torch.Tensor):
                print("id(k) param_mappings[id(k)] = ", id(k), param_mappings[id(k)])
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def sgd(self,
            params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool):
        r"""Functional API that performs SGD algorithm computation.

        See :class:`~torch.optim.SGD` for details.
        """

        for i, param in enumerate(params):

            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            #print("at the start param.shape, d_p shape = ", param.shape, d_p.shape)
            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                    #print("variant 1")
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    #print("variant 2")

                #print("buf shape = ", buf.shape)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
            #print("before last add param.shape, d_p shape = ", param.shape, d_p.shape)
            #print("dbg optimizer sgd_bf16: i, momentum, len(momentum_buffer_list) param shape, param dtype = ", i, momentum, len(momentum_buffer_list), param.shape, param.dtype)
            param.add_(d_p, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            params_with_grad_true = []

            for p in group['params']:

                state = self.state[p]

                local_params_with_grad = []
                local_d_p_list = []
                local_momentum_buffer_list = []

                if len(state) == 0:
                    state["step"] = 0

                    if p.grad is not None and p.data.dtype == torch.bfloat16:
                        # additional fp32 version of master weights
                        state["data_fp32"] = p.data.to(torch.float32)
                    else:
                        state["data_fp32"] = torch.zeros_like(p.data).to(torch.float32)

                if p.grad is not None:

                    #print("dbg: p, p.grad before = ", p, p.grad)

                    bf16_param = p.data.dtype == torch.bfloat16

                    if bf16_param:
                        #grad = grad.to(torch.float32)
                        grad = p.grad.data.to(torch.float32)
                        data = state["data_fp32"]
                    else:
                        grad = p.grad.data
                        data = p.data

                    #print("dbg: p.shape, p.numel, bf16_param = ", p.shape, p.numel(), 1 if bf16_param else 0)
                    #print("dbg: id(p) id(p.grad) id(grad) id(data) = ", id(p), id(p.grad), id(grad), id(data))

                    if grad.is_sparse:
                        raise RuntimeError("This optimizer does not support sparse gradients.")


                    #params_with_grad.append(data) #(p)
                    #d_p_list.append(grad) #(p.grad)

                    
                    temp_p_param      = torch.nn.parameter.Parameter(data)
                    temp_p_grad_param = torch.nn.parameter.Parameter(grad)

                    #print("data shape =  grad shape = ", data.shape, grad.shape)
                    #print("dbg at the start : type(p) type(p.data) type(temp_p_param) type(temp_p_grad_param) = ", type(p), type(p.data), type(temp_p_param), type(temp_p_grad_param))

                    params_with_grad.append(temp_p_param) #(p)
                    d_p_list.append(temp_p_grad_param) #(p.grad)
                    
                    params_with_grad_true.append(p)
                    
                    #params_with_grad.append(p)
                    #d_p_list.append(p.grad)
                    

                    local_params_with_grad.append(temp_p_param)
                    local_d_p_list.append(temp_p_grad_param)

                    #local_params_with_grad.append(p)
                    #local_d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        #momentum_buffer_list.append(None)
                        local_momentum_buffer_list.append(None)
                    else:
                        #momentum_buffer_list.append(state['momentum_buffer'])
                        local_momentum_buffer_list.append(state['momentum_buffer'])

                    state["step"] += 1

                    #print("local_params_with_grad shape = ", temp_p_param.data.shape)
                    #print("dbg: optimizer sgd_bf16 local_params_with_grad = ", local_params_with_grad)
                    #print("dbg: optimizer sgd_bf16 momentum weight_decay, lr, dampening, nesterov = ", momentum, weight_decay, lr, dampening, nesterov)
                    #print("local_d_p_list shape = ", temp_p_grad_param.data.shape)
                    #print("local_d_p_list = ", local_d_p_list)
                    #print("momentum_buffer_list = ", momentum_buffer_list)

                    self.sgd(local_params_with_grad,
                          local_d_p_list,
                          local_momentum_buffer_list,
                          weight_decay=weight_decay,
                          momentum=momentum,
                          lr=lr,
                          dampening=dampening,
                          nesterov=nesterov)

                    state['momentum_buffer'] = local_momentum_buffer_list[0]

                    #print("dbg before the final assigment: id(p) id(p.grad) id(grad) id(data) = ", id(p), id(p.grad), id(grad), id(data))
                    #p.data      = temp_p_param[0].data.to(p.dtype)
                    #p.grad.data = temp_p_grad_param[0].data.to(p.grad.dtype)
                    #print("dbg finally: id(p) id(p.grad) id(grad) id(data) = ", id(p), id(p.grad), id(grad), id(data))

                    #print("temp_p_param  ", temp_p_param)
                    #print("temp_p_grad_param ", temp_p_grad_param)

                    
                    if bf16_param:
                        #p.data = data.to(torch.bfloat16)
                        p.data      = temp_p_param.data.to(torch.bfloat16)
                        p.grad.data = temp_p_grad_param.data.to(p.grad.dtype)
                    else:
                        p.data      = temp_p_param.data.to(p.dtype)
                        p.grad.data = temp_p_grad_param.data.to(p.grad.dtype)
                    
                    #print("p  ", p)
                    #print("p.grad ", p.grad)
                    

            #print("params_with_grad      before = ", params_with_grad)
            #print("params_with_grad_true before = ", params_with_grad_true)

            """
            self.sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)
            """
            #print("params_with_grad      after = ", params_with_grad)
            #print("params_with_grad_true after = ", params_with_grad_true)

            """
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad_true, momentum_buffer_list):
            #for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                #print("id(p) = ", id(p))
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
            """
            """
            for p in group['params']:
                if p.grad is not None:

                    bf16_param = p.data.dtype == torch.bfloat16

                    state = self.state[p]

                    if bf16_param:
                        #p.data = data.to(torch.bfloat16)
                        p.data = state["data_fp32"].to(torch.bfloat16)

                        #print("dbg: id(p) id(p.grad) id(p.data) id(state[data_fp32]) = ", id(p), id(p.grad), id(p.data), id(state["data_fp32"]))
                    #else:
                    #    print("dbg: id(p) id(p.grad) id(p.data)                      = ", id(p), id(p.grad), id(p.data))
            """
        return loss

class SGD_bf16fb_enhanced(torch.optim.SGD):
    r"""Implements an SGD tailored for bf16 training (with an extra fp32 copy and flat buffers)."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        #if not is_available():
        #    raise ValueError("Module function 'bf16_update' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if nesterov:
            raise ValueError("Invalid nesterov value: {}".format(nesterov))
        if dampening != 0:
            raise ValueError("Invalid dampening value: {}".format(dampening))

        # If inheriting from the Optimizer
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        #print("debug: lr = ", lr)
        #print("debug: defaults = ", defaults)
        #print("debug: params = ", params)
        #super(SplitSGD, self).__init__(params, defaults)

        super(torch.optim.SGD, self).__init__(params, defaults)
        """
        # If inheriting from the SGD
        # for PT SGD:
        #def __init__(self, params, lr=required, momentum=0, dampening=0,
        #         weight_decay=0, nesterov=False):
        super(SGD_bf16fb_enhanced, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        
        self._one_time_setup_done = False
        self.block_size = 64
        print("Caution: setting block_size in SplitSGD_bf16fb_enhanced constructor to ", self.block_size)
        #print("Caution: setting with_weight_f32_copy to True in SGD_bf16fb_enhanced constructor")
        self.with_f32_weight_copy = True

    
    # Not really well tested (affects saving/loading from checkpoints)
    def __setstate__(self, state):
        print("Error: __setstate__ has not been implemented for SGD_bf16fb_enhanced")
        """
        super(SGD_bf16fb_enhanced, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
        state_values = list(self.state.values())
        data_fp32_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['data_fp32'])
        if not data_fp32_is_tensor:
            for s in state_values:
                s['data_fp32'] = torch.tensor(float(s['data_fp32']))
        """
    
    def __getstate__(self):
        print("Error: __getstate__ has not been implemented for SGD_bf16fb_enhanced")
        exit()
    
    def _one_time_setup(self):
        if self._one_time_setup_done == True:
            return
        from collections import defaultdict
        self.flat_params = [] #defaultdict(list) #[]
        for i, group in enumerate(self.param_groups):
            #print("debug: i ~ group index = ", i)
            #print("debug: group = ", group)
            model_params = defaultdict(list)
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                #print("debug: in group params p  = ", p)
                #torch.distributed.broadcast(p, 0)
                if not p.requires_grad:
                    continue
                dt = p.dtype
                model_params[dt].append(p)
                #print("debug: appended to model_params (p.shape, dt) = ", p.shape, dt)
            for dt, param_list in model_params.items():
                #print("debug: (dt, param_list) = ", dt, param_list)
                #print("debug: dt in model_params = ", dt)
                #flat_buf = self.FlatBuffer(param_list, group, dt, self.block_size)
                flat_buf = FlatBufferSGD(param_list, group, dt, self.block_size, self.with_f32_weight_copy) #, True if momentum != 0 else False )
                #print("dbg: type of group, group = ", type(group), group)
                self.flat_params.append(flat_buf)

            #value_list = self.flat_params #[0]
            #print("debug: value_list, type = ", value_list, type(value_list))
            #exit()

        #self._step = 0
        self._one_time_setup_done = True

    
    def zero_grad(self):
        if hasattr(self, 'flat_params'):
            for fp in self.flat_params:
                fp._flat_g.zero_()
            #for i, fp in self.flat_params.items():
            #    fp._flat_g.zero_()
        else:
            super(SGD_bf16fb_enhanced, self).zero_grad()
    

    def sgd(self,
            params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool):
        r"""Functional API that performs SGD algorithm computation.

        See :class:`~torch.optim.SGD` for details.
        """

        for i, param in enumerate(params):

            d_p = d_p_list[i]
            #print("dbg: d_p before decaying-add, param = ", d_p, param)
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)
            #print("dbg: after decay d_p = ", d_p)

            #print("at the start param.shape, d_p shape = ", param.shape, d_p.shape)
            if momentum != 0:
                buf = momentum_buffer_list[i]

                #print("dbg:  momentum , dampening, list before = ", momentum, dampening, momentum_buffer_list)
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                    #print("Error: this should not happen since momentum has been added to the flat buffers")
                    #exit()
                    #print("dbg: variant 1")
                else:
                    #print("dbg: before momentum buf = ", buf)
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    #print("dbg: variant 2")

                #print("dbg: after momentum buf = ", buf)
                #print("buf shape = ", buf.shape)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
                #print("dbg: after momentum d_p = ", d_p)
            #print("before last add param.shape, d_p shape = ", param.shape, d_p.shape)
            #print("dbg optimizer sgd_bf16: i, momentum, len(momentum_buffer_list) param shape, param dtype = ", i, momentum, len(momentum_buffer_list), param.shape, param.dtype)
            param.add_(d_p, alpha=-lr)
            #print("dbg: after lr param = ", param)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._one_time_setup()
        #self.state[self.param_groups[0]['params'][0]]['step'] = self._step
        #self.merge_acc_grad()

        for ii, fp in enumerate(self.flat_params):
            group = fp.group
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for i, p in enumerate(fp.param_list):

                state = self.state[p]

                local_params_with_grad = []
                local_d_p_list = []
                local_momentum_buffer_list = []

                s = fp._offsets[i]
                e = fp._offsets[i] + fp._param_sizes[i]

                bf16_param = p.data.dtype == torch.bfloat16

                if len(state) == 0:
                    state["step"] = 0
                    if bf16_param:
                        state["data_fp32"] = fp._flat_f32_w[s : e].view_as(p.data)

                #if 'momentum_buffer' in state:
                #    print("dbg: momentum_buffer = ", state["momentum_buffer"])

                if p.grad is not None:

                    #print("dbg: i = ", i)
                    #print("dbg: p, p.grad before = ", p, p.grad)
                    
                    if bf16_param:
                        #grad = grad.to(torch.float32)
                        grad = p.grad.data.to(torch.float32)
                        data = state["data_fp32"]
                        #print("dbg: grad, data (state[fp_32]), dtype = ", grad, data, data.dtype)
                    else:
                        grad = p.grad.data
                        data = p.data

                    if grad.is_sparse:
                        raise RuntimeError("This optimizer does not support sparse gradients.")

                    #print("dbg: p.shape, p.numel, bf16_param = ", p.shape, p.numel(), 1 if bf16_param else 0)
                    #print("dbg: id(p) id(p.grad) id(grad) id(data) = ", id(p), id(p.grad), id(grad), id(data))

                    #params_with_grad.append(data) #(p)
                    #d_p_list.append(grad) #(p.grad)

                    
                    temp_p_param      = torch.nn.parameter.Parameter(data)
                    temp_p_grad_param = torch.nn.parameter.Parameter(grad)

                    #print("data shape =  grad shape = ", data.shape, grad.shape)
                    #print("dbg at the start : type(p) type(p.data) type(temp_p_param) type(temp_p_grad_param) = ", type(p), type(p.data), type(temp_p_param), type(temp_p_grad_param))

                    #params_with_grad.append(temp_p_param) #(p)
                    #d_p_list.append(temp_p_grad_param) #(p.grad)
                    
                    #params_with_grad_true.append(p)
                    
                    #params_with_grad.append(p)
                    #d_p_list.append(p.grad)
                    

                    local_params_with_grad.append(temp_p_param)
                    local_d_p_list.append(temp_p_grad_param)

                    #local_params_with_grad.append(p)
                    #local_d_p_list.append(p.grad)

                    #state = self.state[p]
                    if 'momentum_buffer' not in state:
                        #momentum_buffer_list.append(None)
                        local_momentum_buffer_list.append(None)
                        #state['momentum_buffer'] = fp._flat_f32_m[s : e].view_as(p.data).copy_(temp_p_grad_param.data).add(temp_p_param, alpha = weight_decay)
                        #print("dbg: appended the momentum from fp")
                        #    self._flat_f32_m[s : e].view_as(p.data).copy_(p.grad.to(torch.float32).data).add(self._flat_f32_w[s : e].view_as(p.data), alpha=weight_decay)
                    else:
                        #momentum_buffer_list.append(state['momentum_buffer'])
                        #print("dbg: appended the existing momentum from the state")
                        local_momentum_buffer_list.append(state['momentum_buffer'])
                    #local_momentum_buffer_list.append(state['momentum_buffer'])

                    state["step"] += 1

                    #print("local_params_with_grad shape = ", temp_p_param.data.shape)
                    #print("dbg: optimizer sgd_bf16 local_params_with_grad = ", local_params_with_grad)
                    #print("dbg: optimizer sgd_bf16 momentum weight_decay, lr, dampening, nesterov = ", momentum, weight_decay, lr, dampening, nesterov)
                    #print("local_d_p_list shape = ", temp_p_grad_param.data.shape)
                    #print("local_d_p_list = ", local_d_p_list)
                    #print("dbg: local_momentum_buffer_list = ", local_momentum_buffer_list)

                    self.sgd(local_params_with_grad,
                          local_d_p_list,
                          local_momentum_buffer_list,
                          weight_decay=weight_decay,
                          momentum=momentum,
                          lr=lr,
                          dampening=dampening,
                          nesterov=nesterov)

                    state['momentum_buffer'] = local_momentum_buffer_list[0]
                    
                    #p.data      = temp_p_param.data.to(p.dtype) #temp_p_param.data.to(p.dtype)
                    #p.grad.data = temp_p_grad_param.data.to(p.grad.dtype) #temp_p_grad_param.data.to(p.grad.dtype)
                    fp._flat_w[s : e].view_as(p.data).copy_(temp_p_param.data.to(p.dtype))
                    fp._flat_g[s : e].view_as(p.data).copy_(temp_p_grad_param.data.to(p.dtype))
                    #print("dbg: temp p, temp p tof32 p.grad after = ", temp_p_param, temp_p_param.to(p.dtype), temp_p_grad_param)
                    #print("dbg: p, p.grad after = ", p, p.grad)

            # end of loop over p in param_list
            #print("dbg: after flat_w, dtype = ", fp._flat_w , fp._flat_w.dtype)
            #print("dbg: after flat_g, dtype = ", fp._flat_g , fp._flat_g.dtype)
            #print("dbg: after flat_m, dtype = ", fp._flat_m , fp._flat_m.dtype)
            #print("dbg: after flat_f32_w, dtype = ", fp._flat_f32_w , fp._flat_f32_w.dtype)
        # end of loop over fp in self.flat_params

        return loss

#"""""""""""""""""""""""""""""""
#"""""""""""""""""""""""""""""""

class SplitSGD_bf16fb_enhanced(torch.optim.SGD):
    r"""Implements a SplitSGD tailored for bf16 training (with an extra bf16 low-part of fp32 weights and flat buffers)."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        #if not is_available():
        #    raise ValueError("Module function 'bf16_update' not available for SplitSGD")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if nesterov:
            raise ValueError("Invalid nesterov value: {}".format(nesterov))
        if dampening != 0:
            raise ValueError("Invalid dampening value: {}".format(dampening))

        super(SplitSGD_bf16fb_enhanced, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.state['_one_time_setup_done'] = False
        self.state['block_size'] = 64
        self._one_time_setup_done = False
        self.block_size = 64
        print("Caution: setting block_size in SplitSGD_bf16fb_enhanced constructor to ", self.block_size)
        # FIXME: Disable after the splitsgd is properly implemented
        self.with_f32_weight_copy = False #True #False
        #print("Caution: setting with_weight_f32_copy to True in SplitSGD_bf16fb_enhanced constructor")
        self.with_bf16_momentum = False and momentum != 0 #True if momentum != 0 else False
        self.with_fp32_momentum = True  and momentum != 0 #True if momentum != 0 else False
        self.with_split = True
        print("dbg: advanced split-sgd flags: with_f32_weight_copy, with_bf16_momentum, with_fp32_momentum, with_split = ",
                self.with_f32_weight_copy, self.with_bf16_momentum, self.with_fp32_momentum, self.with_split)

        self.state['_one_time_map_done'] = False
        self._one_time_map_done = False

        if not self.with_split:
            print("Error: with_split flag must be set to True for SplitSGD optimizer")
            exit()

    
    def __getstate__(self):
        print("Error: __getstate__ has not been implemented for SplitSGD_bf16fb_enhanced")
        exit()

    # One important difference w.r.t to standard SGD's state_dict: setting packed_state['_one_time_map_done'] = False
    # Without it, remap is not doine in the step() after the optimizer state_dict is loaded and hence optimizer works incorrectly
    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0
        start_index_dbg = 0

        def pack_group(group):
            nonlocal start_index
            nonlocal start_index_dbg
            packed = {k: v for k, v in group.items() if k != 'params'}
            #print("debug: group params = ", group['params'])
            #for i, p in enumerate(group['params'], start_index_dbg):
            #    print("i p id(p) is_in = ", i, p, id(p), 1 if id(p) in param_mappings else 0)
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            #print("debug: packed, param_mappings = ", packed, param_mappings)
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        #print("debug: self.param_groups = ", self.param_groups)
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        #print("state keys = ", self.state.keys())
        #print("param_mappings = ", param_mappings)
        #for k, v in self.state.items():
        #    #print("type(k) type(v) id(k) = ", type(k), type(v), id(k))
        #    if isinstance(k, torch.Tensor):
        #        print("id(k) param_mappings[id(k)] = ", id(k), param_mappings[id(k)])
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        packed_state['_one_time_map_done'] = False
        #print("dbg: packed_state = ", packed_state)
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    
    def _one_time_setup(self):
        if self._one_time_setup_done == True and self.state['_one_time_setup_done'] == True:
            return
        print("dbg: doing one_time_setup")
        from collections import defaultdict
        #self.flat_params = [] #[]
        self.state['flat_params'] = []
        for i, group in enumerate(self.param_groups):
            #print("debug: group = ", group)
            model_params = defaultdict(list)
            for p in group['params']:
                #print("debug: in group params p  = ", p)
                #torch.distributed.broadcast(p, 0)
                if not p.requires_grad:
                    continue
                dt = p.dtype
                model_params[dt].append(p)
                #print("debug: appended to model_params (p, dt) = ", p, dt)
            for dt, param_list in model_params.items():
                #print("debug: (dt, param_list) = ", dt, param_list)
                #flat_buf = self.FlatBuffer(param_list, group, dt, self.block_size)
                flat_buf = FlatBufferSGD(param_list, group, dt, self.block_size, self.with_f32_weight_copy, self.with_bf16_momentum, self.with_fp32_momentum, self.with_split) #, True if momentum != 0 else False )
                #print("dbg: type of group, group = ", type(group), group)
                #self.flat_params.append(flat_buf)
                self.state['flat_params'].append(flat_buf)

            #value_list = self.flat_params
            #print("debug: value_list, type = ", value_list, type(value_list))
            #exit()

        self._step = 0
        self._one_time_setup_done = True
        self.state['_one_time_setup_done'] = True

        self.state['_one_time_map_done'] = True
        self._one_time_map_done = True

    def _one_time_remap_onto(self):
        if self._one_time_map_done == True and self.state['_one_time_map_done'] == True:
            return
        #print("dbg: doing one_time_remap_onto")

        from collections import defaultdict

        for i, group in enumerate(self.param_groups):
            model_params = defaultdict(list)
            for p in group['params']:
                if not p.requires_grad:
                    continue
                dt = p.dtype
                model_params[dt].append(p)
            #print("dbg: model_params.items() = ", model_params.items())
            #print("dbg: list of model_params.items() = ", list(model_params.items()))
            #print("dbg: zip with a list of model_params.items() = ", zip(list(model_params.items()), self.state['flat_params']))
            for (dt, param_list), flat_buf in zip(list(model_params.items()), self.state['flat_params']):
                #print("dbg: in the one_time_remap, flat_buf = ", flat_buf, flat_buf._flat_w.data_ptr())
                #for p in param_list:
                #    print("dbg: in the one_time_remap, p in param_list = ", p, p.data_ptr())
                flat_buf.remap_onto(param_list)
                #flat_buf = FlatBufferSGD(param_list, group, dt, self.block_size, self.with_f32_weight_copy, self.with_bf16_momentum, self.with_fp32_momentum, self.with_split) #, True if momentum != 0 else False )
                #self.flat_params.append(flat_buf)
                #self.state['flat_params'].append(flat_buf)

        self.state['_one_time_map_done'] = True
        self._one_time_map_done = True

    
    def zero_grad(self):
        if hasattr(self, 'flat_params'):
            #for fp in self.flat_params:
            for fp in self.state['flat_params']:
                fp._flat_g.zero_()
        else:
            super(SplitSGD_bf16fb_enhanced, self).zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._one_time_setup()
        self._one_time_remap_onto()
        # Warning: unlike some other optimizers, _step is enumerated starting with 0 (which is important
        # for the momentum check done in fused_sgd_vX implementations)
        # Therefore, the addition is done AFTER the actual computations of the step
        #self._step += 1

        #for i, group in enumerate(self.param_groups):
        #    for p in group['params']:
        #        print("dbg: in self.param_groups before p = ", p, p.data_ptr())
        #        print("dbg: in self.param_groups before p.grad = ", p.grad, p.grad.data_ptr())

        #for ii, fp in enumerate(self.flat_params):
        for ii, fp in enumerate(self.state['flat_params']):
            group = fp.group
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group["lr"]

            #print("dbg: inside step loop, fp, fp flat_w data_ptr = ", fp, fp._flat_w.data_ptr())
            #print("dbg: inside step loop, fp, fp flat_g data_ptr = ", fp, fp._flat_g.data_ptr())

            #for p in group['params']:
            #    print("p in group params before = ", p, p.dtype, p.data_ptr())

            if momentum != 0 and self.with_fp32_momentum:
                #optim_cpp.fused_sgd_v2(fp._flat_w, fp._flat_g, fp._flat_m, fp._flat_wl, fp._offsets, fp._block_sizes, fp._block2param,
                #                    weight_decay, momentum, dampening, nesterov, lr, fp.block_size, self._step, None)
                optim_cpp.fused_sgd_v3(fp._flat_w, fp._flat_g, fp._flat_m, fp._flat_wl, fp._offsets, fp._block_sizes, fp._block2param,
                                       weight_decay, momentum, dampening, nesterov, lr, fp.block_size, self._step)
            elif momentum != 0 and self.with_bf16_momentum:
                #optim_cpp.fused_sgd_v0(fp._flat_w, fp._flat_g, fp._flat_m, fp._flat_wl, fp._offsets, fp._block_sizes, fp._block2param,
                #                    weight_decay, momentum, dampening, nesterov, lr, fp.block_size, self._step, None)
                optim_cpp.fused_sgd_v1(fp._flat_w, fp._flat_g, fp._flat_m, fp._flat_wl, fp._offsets, fp._block_sizes, fp._block2param,
                                       weight_decay, momentum, dampening, nesterov, lr, fp.block_size, self._step)

            #for p in group['params']:
            #    print("p in group params after = ", p, p.dtype, p.data_ptr())

            #print("dbg: after step, flat_w = ", fp._flat_w, fp._flat_w.data_ptr())
            #print("dbg: after step, flat_g = ", fp._flat_g, fp._flat_g.data_ptr())

        #for i, group in enumerate(self.param_groups):
        #    for p in group['params']:
        #        print("dbg: in self.param_groups after p = ", p, p.data_ptr())

        # end of loop over fp in self.state['flat_params']
        self._step += 1
        return loss
