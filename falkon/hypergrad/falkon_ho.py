
class FalkonHO(AbstractHypergradModule):
    def __init__(self, M, Xtr, Ytr, Xts, Yts):
        super().__init__()
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xts = Xts
        self.Yts = Yts

        self.M = M
        self.opt = FalkonOptions(use_cpu=True)

        centers = UniformSelector(np.random.default_rng(10)).select(self.Xtr, None, M)
        self.center_selection = FixedSelector(centers)
        self.maxiter = 10

    def inner_opt(self, params, hparams):
        """This is NOT DIFFERENTIABLE"""
        alpha = params[0]
        penalty, sigma = hparams

        with torch.autograd.no_grad():
            kernel = DiffGaussianKernel(sigma.detach(), self.opt)
            def sq_err(y_true, y_pred):
                return torch.mean((y_true - y_pred)**2)
            self.flk = falkon.Falkon(
                kernel,
                torch.exp(-penalty.detach()).item(),
                self.M,
                center_selection=self.center_selection,
                maxiter=self.maxiter,
                seed=129,
                options=self.opt)
            self.flk.fit(self.Xtr, self.Ytr, alpha=alpha.detach())
        return [self.flk.alpha_]

    def val_loss(self, params, hparams):
        alpha = params[0]
        penalty, sigma = hparams
        kernel = DiffGaussianKernel(sigma, self.opt)
        ny_points = self.flk.ny_points_
        preds = kernel.mmv(self.Xts, ny_points, alpha)
        return torch.mean((preds - self.Yts) ** 2)

    def val_loss_grads(self, params, hparams):
        o_loss = self.val_loss(params, hparams)
        return (
            torch.autograd.grad(o_loss, params, allow_unused=True, create_graph=False,
                                retain_graph=True),
            torch.autograd.grad(o_loss, hparams, allow_unused=True, create_graph=False,
                                retain_graph=False)
        )

    def param_derivative(self, params, hparams):
        """Derivative of the training loss, with respect to the parameters"""
        alpha = params[0]
        penalty, sigma = hparams
        N = self.Xtr.shape[0]
        ny_points = self.flk.ny_points_

        kernel = DiffGaussianKernel(sigma, self.opt)

        # 2/N * (K_MN(K_NM @ alpha - Y)) + 2*lambda*(K_MM @ alpha)
        out = 2 * ((1/N) * kernel.mmv(
            ny_points, self.Xtr, kernel.mmv(self.Xtr, ny_points, alpha, opt=self.opt) - self.Ytr, opt=self.opt) +
              torch.exp(-penalty) * kernel.mmv(ny_points, ny_points, alpha, opt=self.opt))
        return [out]

    def mixed_vector_product(self, hparams, first_derivative, vector):
        return torch.autograd.grad(first_derivative, hparams, grad_outputs=vector,
                                   allow_unused=True)

    def hessian_vector_product(self, params, first_derivative, vector):
        # Here we need to retain the graph, since we will call the function
        # multiple times within the conjugate-gradient procedure.
        hvp = torch.autograd.grad(first_derivative, params, grad_outputs=vector,
                                  retain_graph=True)
        return hvp
