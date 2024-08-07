import torch
import gzip
import torch.utils.data
import numpy as np


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    

def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y

def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    
def trace(x):
  print(x)
  return x

# See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
def KL_Normals(d0, d1):
  assert d0.mu.size() == d1.mu.size()
  sigma0_sqr = torch.pow(d0.sigma, 2)
  sigma1_sqr = torch.pow(d1.sigma, 2)
  return torch.sum(
    -0.5
    + (sigma0_sqr + torch.pow(d0.mu - d1.mu, 2)) / (2 * sigma1_sqr)
    + torch.log(d1.sigma)
    - torch.log(d0.sigma)
  )

class Lambda(torch.nn.Module):
  def __init__(self, func, extra_args=(), extra_kwargs={}):
    super(Lambda, self).__init__()
    self.func = func
    self.extra_args = extra_args
    self.extra_kwargs = extra_kwargs

  def forward(self, x):
    return self.func(x, *self.extra_args, **self.extra_kwargs)

# def softplus(x):
#   return torch.log(torch.exp(x) + 1)

# Stolen from upstream master
def split(tensor, split_size_or_sections, dim=0):
  """Splits the tensor into chunks.
  If ``split_size_or_sections`` is an integer type, then ``tensor`` will be
  split into equally sized chunks (if possible).
  Last chunk will be smaller if the tensor size along a given dimension
  is not divisible by ``split_size``.
  If ``split_size_or_sections`` is a list, then ``tensor`` will be split
  into ``len(split_size_or_sections)`` chunks with sizes in ``dim`` according
  to ``split_size_or_sections``.

  Arguments:
      tensor (Tensor): tensor to split.
      split_size_or_sections (int) or (list(int)): size of a single chunk or
      list of sizes for each chunk
      dim (int): dimension along which to split the tensor.
  """
  if dim < 0:
    dim += tensor.dim()
  dim_size = tensor.size(dim)

  if isinstance(split_size_or_sections, int):
    split_size = split_size_or_sections
    num_splits = (dim_size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - dim_size)

    def get_split_size(i):
      return split_size if i < num_splits - 1 else last_split_size
    return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i)))
                 for i in range(0, num_splits))

  else:
    if dim_size != sum(split_size_or_sections):
      raise ValueError("Sum of split sizes exceeds tensor dim")
    split_indices = [0] + split_size_or_sections
    split_indices = torch.cumsum(torch.Tensor(split_indices), dim=0)

    return tuple(
      tensor.narrow(int(dim), int(start), int(length))
      for start, length in zip(split_indices, split_size_or_sections))

class Bijection(object):
  def __init__(
      self,
      forward,
      inverse,
      forward_log_abs_det_jacobian,
      inverse_log_abs_det_jacobian
  ):
    self.forward = forward
    self.inverse = inverse
    self.forward_log_abs_det_jacobian = forward_log_abs_det_jacobian
    self.inverse_log_abs_det_jacobian = inverse_log_abs_det_jacobian

  def __call__(self, x):
    return self.forward(x)

def invert_bijection(f):
  return Bijection(
    forward=f.inverse,
    inverse=f.forward,
    forward_log_abs_det_jacobian=f.inverse_log_abs_det_jacobian,
    inverse_log_abs_det_jacobian=f.forward_log_abs_det_jacobian
  )

softplus = Bijection(
  forward=lambda x: torch.log(torch.exp(x) + 1),
  inverse=lambda y: torch.log(torch.exp(y) - 1),
  forward_log_abs_det_jacobian=lambda x: torch.sum(x - torch.log(torch.exp(x) + 1)),
  inverse_log_abs_det_jacobian=lambda y: torch.sum(y - torch.log(torch.abs(torch.exp(y) - 1)))
)
