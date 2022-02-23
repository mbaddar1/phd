import seaborn as sns
from matplotlib import pyplot as plt

from distributions.vector_gmm import gen_vec_gaussian_mixture

if __name__ == '__main__':
    k = 3
    d = 1
    x = gen_vec_gaussian_mixture(n_components=k, dim=d)
    samples = x.sample_n(1000)
    sns.kdeplot(x=samples[:, 0])
    plt.savefig(f'kde_gmm_k_{k}_d_{1}.png')
