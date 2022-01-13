from torchdyn.datasets import *


def sample_annuli(n_samples=1 << 10, device=torch.device('cpu')):
    X, y = ToyDataset().generate(n_samples, 'spheres', dim=2, noise=.05)
    return 2 * X.to(device), y.long().to(device)


if __name__ == '__main__':
    X, y = sample_annuli()
    pass
