"""
Module for evaluating MMD generative models.
Yujia Li, 11/2014
"""

# import cPickle as pickle
import time
import numpy as np

def safe_diag(x):
    if isinstance(x, np.ndarray):
        return x.diagonal()
    if isinstance(x, np.ndarray):
        if x.shape[0] > 4000:
            return np.array(x.asarray().diagonal())
        else:
            return x.diag()

    raise Exception()

class Kernel(object):
    def __init__(self):
        pass

    def compute_kernel_matrix(self, x):
        """
        x: n_examples * n_dims input data matrix
        Return: n_examples * n_examples kernel matrix
        """
        return self.compute_kernel_transformation(x, x)

    def compute_kernel_transformation(self, x_base, x_new):
        """
        x_base: n_examples_1 * n_dims data matrix
        x_new: n_examples_2 * n_dims data matrix
        For each example in x_new, compute its kernel distance with each of the
        examples in x_base, return a n_examples_2 * n_examples_1 matrix as the
        transformed representation of x_new.
        """
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def compute_kernel_matrix(self, x):
        x = x if isinstance(x, np.ndarray) else np.array(x)
        xx = x.dot(x.T)
        x_diag = safe_diag(xx)

        return np.exp(-1.0 / (2 * self.sigma**2) * (-2 * xx + x_diag + x_diag[:,np.newaxis]))

    def compute_kernel_transformation(self, x_base, x_new):
        x_base = x_base if isinstance(x_base, np.ndarray) else np.array(x_base)
        x_new = x_new if isinstance(x_new, np.ndarray) else np.array(x_new)

        xx = x_new.dot(x_base.T)
        xx_base = (x_base**2).sum(axis=1)
        xx_new = (x_new**2).sum(axis=1)
        return np.exp(-1.0 / (2 * self.sigma**2) * (-2 * xx + xx_base + xx_new[:,np.newaxis]))

    def get_name(self):
        return 'gaussian_kernel'

class EuclideanKernel(Kernel):
    def __init__(self):
        pass

    def compute_kernel_matrix(self, x):
        x = x if isinstance(x, np.ndarray) else np.array(x)
        xx = x.dot(x.T)
        x_diag = safe_diag(xx)

        return (-2 * xx + x_diag + x_diag[:,np.newaxis])

    def compute_kernel_transformation(self, x_base, x_new):
        x_base = x_base if isinstance(x_base, np.ndarray) else np.array(x_base)
        x_new = x_new if isinstance(x_new, np.ndarray) else np.array(x_new)

        xx = x_new.dot(x_base.T)
        xx_base = (x_base**2).sum(axis=1)
        xx_new = (x_new**2).sum(axis=1)
        return (-2 * xx + xx_base + xx_new[:,np.newaxis])


def log_exp_sum_1d(x):
    """
    This computes log(exp(x_1) + exp(x_2) + ... + exp(x_n)) as 
    x* + log(exp(x_1-x*) + exp(x_2-x*) + ... + exp(x_n-x*)), where x* is the
    max over all x_i.  This can avoid numerical problems.
    """
    x_max = x.max()
    if isinstance(x, np.ndarray):
        return x_max + np.log(np.exp(x - x_max).sum())
    else:
        return x_max + np.log(np.exp(x - x_max).sum())

def log_exp_sum(x, axis=1):
    x_max = x.max(axis=axis)
    if isinstance(x, np.ndarray):
        return (x_max + np.log(np.exp(x - x_max[:,np.newaxis]).sum(axis=axis)))
    else:
        return x_max + np.log(np.exp(x - x_max[:,np.newaxis]).sum(axis=axis))

class KDE(object):
    """
    Kernel density estimation.
    """
    def __init__(self, data, sigma):
        self.x = np.array(data) if not isinstance(data, np.ndarray) else data
        self.sigma = sigma
        self.N = self.x.shape[0]
        self.d = self.x.shape[1]
        self._ek =  EuclideanKernel()

        self.factor = float(-np.log(self.N) - self.d / 2.0 * np.log(2 * np.pi * self.sigma**2))

    def _log_likelihood(self, data):
        print(self.x.shape, data.shape)
        transform = -self._ek.compute_kernel_transformation(self.x, data)
        return log_exp_sum( transform / (2 * self.sigma**2), axis=1) + self.factor

    def log_likelihood(self, data, batch_size=1000):
        n_cases = data.shape[0]
        if n_cases <= batch_size:
            return self._log_likelihood(data)
        else:
            n_batches = (n_cases + batch_size - 1) // batch_size
            log_like = np.zeros(n_cases, dtype=np.float)

            for i_batch in range(n_batches):
                i_start = i_batch * batch_size
                i_end = n_cases if (i_batch + 1 == n_batches) else (i_start + batch_size)
                log_like[i_start:i_end] = self._log_likelihood(data[i_start:i_end])

            return log_like

    def likelihood(self, data):
        """
        data is a n_example x n_dims matrix.
        """
        return np.exp(self.log_likelihood(data))

    def average_likelihood(self, data):
        return self.likelihood(data).mean()

    def average_log_likelihood(self, data, batch_size=1000):
        return self.log_likelihood(data, batch_size=batch_size).mean()

    def average_std_log_likelihood(self, data, batch_size=1000):
        l = self.log_likelihood(data)
        return l.mean(), l.std()

    def average_se_log_likelihood(self, data, batch_size=1000):
        l = self.log_likelihood(data)
        return l.mean(), l.std() / np.sqrt(data.shape[0])


def kde_evaluation(test_data, samples, sigma_range=np.arange(0.1, 1.5, 0.02), verbose=True):
    best_log_likelihood = float('-inf')
    for sigma in sigma_range:
        log_likelihood = KDE(samples, sigma).average_log_likelihood(test_data, batch_size=100)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
        if verbose:
            print('sigma=%g, log_likelihood=%.2f' % (sigma, log_likelihood))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f' % best_log_likelihood)
        print('')
    return best_log_likelihood



def kde_eval_mnist(net, test_data, n_samples=10000, sigma_range=np.arange(0.1, 0.3, 0.01), verbose=True):
    s = net.generate_samples(n_samples=n_samples)
    best_log_likelihood = float('-inf')
    best_se = 0
    best_sigma = 0
    for sigma in sigma_range:
        log_likelihood, se = KDE(s, sigma).average_se_log_likelihood(test_data)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_se = se 
            best_sigma = sigma
        if verbose:
            print('sigma=%g, log_likelihood=%.2f (%.2f)' % (sigma, log_likelihood, se))

    if verbose:
        print('====================')
        print('Best log_likelihood=%.2f (%.2f)' % (best_log_likelihood, best_se))
        print('')
    return best_log_likelihood, best_se, best_sigma

