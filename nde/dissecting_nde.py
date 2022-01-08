"""
https://github.com/DiffEqML/torchdyn/tree/master/tutorials
https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module3-tasks/m3c_continuous_normalizing_flows.ipynb
https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
"""
import pytorch_lightning as pl
import torch.utils.data as data
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn.nn import DepthCat, GalLinear, Fourier
from torchdyn.utils import *


class Learner(pl.LightningModule):
    def __init__(self, t_span: torch.Tensor, model: nn.Module):
        super().__init__()
        self.model, self.t_span = model, t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader


# TODO
"""
1. understand torchdyn.core.utils.standardize_vf_call_signature , torchdyn.core.defunc.DEFuncBase , 
    torchdyn.core.defunc.DEFunc
2. 
"""
if __name__ == '__main__':
    # configs
    dry_run = False
    device = torch.device('cpu')
    #
    d = ToyDataset()
    X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.1)
    colors = ['orange', 'blue']
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    for i in range(len(X)):
        ax.scatter(X[i, 0], X[i, 1], s=1, color=colors[yn[i].int()])
    fig.savefig('fig1.png')

    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    train = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

    # Galerkin Model
    # vector field parametrized by a NN with "GalLinear" layer
    # notice how DepthCat is still used since Galerkin layers make use of `t` (though in a different way compared to concatenation)
    f = nn.Sequential(DepthCat(1),
                      GalLinear(2, 32, expfunc=Fourier(5)),
                      nn.Tanh(),
                      nn.Linear(32, 2))

    t_span = torch.linspace(0, 1, 2)

    # Neural ODE
    model = NeuralODE(f, sensitivity='interpolated_adjoint', solver='tsit5', atol=1e-3, rtol=1e-3).to(device)
    learn = Learner(t_span, model)
    if dry_run:
        trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    else:
        trainer = pl.Trainer(min_epochs=100, max_epochs=100, progress_bar_refresh_rate=10)
    trainer.fit(learn)


    # torchdyn with CNF


