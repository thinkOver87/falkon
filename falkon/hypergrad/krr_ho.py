import torch


from falkon.hypergrad.common import AbstractHypergrad


def naive_diff_rbf_kernel(self, X1, X2, sigma):
    D = torch.norm(X1, p=2, dim=1, keepdim=True).pow_(2) + \
        torch.norm(X2, p=2, dim=1, keepdim=True).pow_(2).T - \
        2 * (X1 @ X2.T)
    D = D / (-2 * sigma ** 2)
    return torch.exp(D)


class KRR(AbstractHypergrad):
    def __init__(self, lr, Xtr, Ytr, Xts, Yts):
        super().__init__()
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xts = Xts
        self.Yts = Yts
        self.lr = lr

    def inner_fp_map(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams
        N = alpha.shape[0]
        K = naive_diff_rbf_kernel(self.Xtr, self.Xtr, sigma)

        # loss = torch.mean((K @ alpha - self.Ytr)**2) + penalty * alpha.T @ K @ alpha
        # update = torch.autograd.grad(loss, params)[0]
        Kalpha = K @ alpha
        update = (2 / N) * K @ (Kalpha - self.Ytr) + 2 * penalty * Kalpha
        return [alpha - self.lr * update]

    def val_loss(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams

        Kts = naive_diff_rbf_kernel(self.Xts, self.Xtr, sigma)
        preds = Kts @ alpha
        return torch.mean((preds - self.Yts) ** 2)

    def tr_loss(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams

        Ktr = naive_diff_rbf_kernel(self.Xtr, self.Xtr, sigma)
        preds = Ktr @ alpha
        return torch.mean((preds - self.Ytr) ** 2)

    def param_derivative(self, params, hparams):
        """Derivative of the training loss, with respect to the parameters"""
        alpha = params[0]
        penalty, sigma = hparams
        N = alpha.shape[0]
        K = naive_diff_rbf_kernel(self.Xtr, self.Xtr, sigma)

        # loss = torch.mean((K @ alpha - self.Ytr)**2) + penalty * alpha.T @ K @ alpha
        # update = torch.autograd.grad(loss, params)[0]
        Kalpha = K @ alpha
        update = (2 / N) * K @ (Kalpha - self.Ytr) + 2 * penalty * Kalpha
        return update

    def hessian_vector_product(self, params, first_derivative, vector):
        """
        Calculates the Hessian of the training loss (with respect to the parameters), multiplied
        by an arbitrary vector using the magic of autodiff.

        Parameters
        ----------
        params
        hparams
        first_derivative
            The derivative of the training loss, wrt params
        vector
            The vector to multiply the Hessian by
        """
        hvp = torch.autograd.grad(first_derivative, params, grad_outputs=vector, retain_graph=True)
        return hvp

    def mixed_vector_product(self, hparams, first_derivative, vector):
        """
        Calculates the mixed-derivative of the training loss, with respect to the parameters
        first (that derivative should be in the `first_derivative` parameter), and then
        with respect to the hyper-parameters; all multiplied by an arbitrary vector.

        Parameters
        ----------
        params
        hparams
        first_derivative
            The derivative of the training loss, wrt params
        vector
            The vector to multiply the derivative by
        """
        return torch.autograd.grad(first_derivative, hparams, grad_outputs=vector,
                                   allow_unused=True)

    def val_loss_grads(self, params, hparams):
        o_loss = self.val_loss(params, hparams)
        return (
            torch.autograd.grad(o_loss, params, allow_unused=True, create_graph=True,
                                retain_graph=True),
            torch.autograd.grad(o_loss, hparams, allow_unused=True, create_graph=True,
                                retain_graph=True)
        )
