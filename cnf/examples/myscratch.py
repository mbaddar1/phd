# mix = torch.distributions.Categorical(torch.ones(3))
# x_ = mix.sample_n(10)
# print(x_)
# print(x_.shape)
#
# n_components = 3
# dim = 2
# base_distribution_dummy = torch.distributions.MultivariateNormal(loc=torch.tensor([0.0, 0.0]),
#                                                                  scale_tril=torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
# print(base_distribution_dummy.batch_shape,base_distribution_dummy.event_shape)
import numpy as np
import torch

# t = torch.randn(3, 2, 2)
# print(t)
# t = torch.tril(t)
# print(t)
# t.fill_diagonal_(1, )

n_components = 4
dim = 3
list_ = []
for i in range(n_components):
    t = torch.randn(dim, dim)
    t = torch.tril(t)
    t.fill_diagonal_(np.random.uniform(0.1, 10))
    list_.append(t)
t_stk = torch.stack(list_)
print(t_stk)
dist_ = torch.distributions.MultivariateNormal(loc=torch.randn(n_components,dim),scale_tril=t_stk)
print(dist_.batch_shape,dist_.event_shape)
comp = torch.distributions.Independent(
        torch.distributions.MultivariateNormal(loc=torch.randn(n_components, dim), scale_tril=t_stk),0)
print(comp.batch_shape,comp.event_shape)
mix = torch.distributions.Categorical(torch.ones(n_components))
gmm_ = torch.distributions.MixtureSameFamily(mix, comp)
x = gmm_.sample_n(100)  # dummy line , for illustration only
print(x.shape)
# print('shape')
# print(base_distribution_dummy.batch_shape, base_distribution_dummy.event_shape)
#
# print(base_distribution_dummy.loc)
# x2_ = base_distribution_dummy.sample_n(10)
# print(x2_)
# comp = torch.distributions.Independent(
#     base_distribution=torch.distributions.Normal(torch.randn(n_components, dim), torch.rand(n_components, dim)),
#     reinterpreted_batch_ndims=1)
