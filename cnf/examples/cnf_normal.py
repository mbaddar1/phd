import torch


class MultiVariateUniform():
    def __init__(self, a=0, b=1, d=1):
        self.a = a
        self.b = b
        self.d = d

    def sample(self, n):
        pass

    def logprob(self, x):
        pass


if __name__ == '__main__':
    """
    Understanding Shapes in PyTorch
    https://bochang.me/blog/posts/pytorch-distributions/ 
    """
    """
    Probability Density Function Transformation
    https://www.cl.cam.ac.uk/teaching/2003/Probability/prob11.pdf 
    """
    d = 4
    batch_size = 50
    bast_dist = torch.distributions.Normal(loc=torch.zeros(d), scale=torch.ones(d))
    s = bast_dist.sample_n(1000)
    print(s.shape)
    mio = 5

    target_dist = torch.distributions.Normal(loc=torch.tensor([mio] * d), scale=torch.tensor([sigma] * d))
    # code to get scale-tril mtx, comment it and make it fixed for re-reproducibility
    # scale_tril = torch.tril(torch.distributions.Uniform(low=0.1,high=1.5).sample((d,d)))
    scale_tril = torch.tensor([[0.1999, 0.0000, 0.0000, 0.0000],
                               [0.3015, 0.2242, 0.0000, 0.0000],
                               [1.1173, 0.3950, 0.9838, 0.0000],
                               [0.2646, 1.1769, 0.8492, 0.8105]])
    assert scale_tril.shape[0]==d,f"scale-tril shape doesn't match d , {scale_tril.shape[0]}!= {d}"
    target_dist_2 = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * d), scale_tril=scale_tril)
