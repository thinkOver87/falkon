import abc
from typing import Sequence, Optional, Tuple

import torch


class AbsHypergradModel(abc.ABC):
    @abc.abstractmethod
    def val_loss(self,
                 params: Sequence[torch.Tensor],
                 hparams: Sequence[torch.Tensor]) -> float:
        pass

    @abc.abstractmethod
    def param_derivative(self,
                         params: Sequence[torch.Tensor],
                         hparams: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        pass

    @abc.abstractmethod
    def hessian_vector_product(self,
                               params: Sequence[torch.Tensor],
                               first_derivative: Sequence[torch.Tensor],
                               vector: Union[torch.Tensor, Sequence[torch.Tensor]]) -> \
                                Sequence[torch.Tensor]:
        pass

    @abc.abstractmethod
    def mixed_vector_product(self,
                             hparams: Sequence[torch.Tensor],
                             first_derivative: Sequence[torch.Tensor],
                             vector: Union[torch.Tensor, Sequence[Optional[torch.Tensor]], None]) -> \
                                Sequence[Optional[torch.Tensor]]:
        pass

    @abc.abstractmethod
    def val_loss_grads(self,
                       params: Sequence[torch.Tensor],
                       hparams: Sequence[torch.Tensor]) -> \
                        Tuple[Sequence[Optional[torch.Tensor]], Sequence[Optional[torch.Tensor]]]:
        pass
