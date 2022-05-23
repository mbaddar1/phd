import logging
from typing import Iterable, Dict, Union

import torch

# TODO read and experiment
from torch.distributions import Normal, MultivariateNormal

"""
Probability Density Function Transformation
https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/03%3A_Distributions/3.07%3A_Transformations_of_Random_Variables
https://www.cl.cam.ac.uk/teaching/2003/Probability/prob11.pdf 
https://en.wikibooks.org/wiki/Probability/Transformation_of_Random_Variables
https://en.wikibooks.org/wiki/Probability/Transformation_of_Probability_Densities
https://www.math.arizona.edu/~jwatkins/f-transform.pdf 
https://www.math.arizona.edu/~jwatkins/f-transform.pdf
"""
# TODO implementation technicalities
"""
Understanding Shapes in PyTorch
https://bochang.me/blog/posts/pytorch-distributions/ 
"""


def get_target_dist(d: int) -> dict[str, Union[Normal, MultivariateNormal]]:
    mio = 5

    # code to get scale-tril mtx, comment it and make it fixed for re-reproducibility
    # scale_tril = torch.tril(torch.distributions.Uniform(low=0.1,high=1.5).sample((d,d)))
    scale_tril = torch.tensor([[0.1999, 0.0000, 0.0000, 0.0000],
                               [0.3015, 0.2242, 0.0000, 0.0000],
                               [1.1173, 0.3950, 0.9838, 0.0000],
                               [0.2646, 1.1769, 0.8492, 0.8105]])
    scale_tril_diag = torch.diag(scale_tril)

    assert scale_tril.shape[0] == d, f"scale-tril shape doesn't match d , {scale_tril.shape[0]}!= {d}"
    target_dist_diag = torch.distributions.Normal(loc=torch.tensor([mio] * d), scale=scale_tril_diag)
    target_dist_tril = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * d), scale_tril=scale_tril)
    return {'target_dist_diag': target_dist_diag, 'target_dist_tril': target_dist_tril}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # config params
    D_z = 4  # dimension of latent variable

    # Start main code
    logger.info('Getting base distribution')
    bast_dist = torch.distributions.Normal(loc=torch.zeros(D_z), scale=torch.ones(D_z))

    # get target distribution
    get_target_dist(d=D_z)
