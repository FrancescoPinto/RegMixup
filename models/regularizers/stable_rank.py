#Contributor: Amartya Sanyal
import torch
import math
from torch.nn.functional import normalize


class StableRank(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, name='weight',  n_power_iterations=1, dim=0,
                 eps=1e-12, rank=0.7):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(
                                 n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.rank_ratio = rank
        self.rank = None

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim,
                *[d for d in range(weight_mat.dim())
                  if d != self.dim])
        height = weight_mat.size(0)
        weight_mat_reshaped = weight_mat.reshape(height, -1)
        max_rank = min(
            weight_mat_reshaped.shape[0], weight_mat_reshaped.shape[1])
        self.rank = self.rank_ratio * max_rank
        return weight_mat_reshaped

    def reshape_matrix_to_weight(self, weight_matrix, shape):
        weight_mat = weight_matrix
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim,
                *[d for d in range(weight_mat.dim())
                  if d != self.dim])
        return weight_mat.reshape(shape)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        curr_shape = weight.shape
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`,
                    # where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations
                    # of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u),
                                  dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v),
                                  dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        if self.rank_ratio > 0.9999:
            return weight / sigma
        weight_mat = weight_mat / sigma

        rank_1 = torch.ger(u, v)
        residual = weight_mat - rank_1
        frob = torch.norm(residual)
        num = math.sqrt(self.rank - 1)
        if frob > num:
            weight_mat = residual * num / frob + rank_1
        weight_ = self.reshape_matrix_to_weight(weight_mat, curr_shape)
        return weight_

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(
            self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(
            module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(
            weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps, rank):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, StableRank) and hook.name == name:
                raise RuntimeError("Cannot register two stable_rank hooks on"
                                   " the same parameter {}".format(name))

        fn = StableRank(name, n_power_iterations, dim, eps, rank)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # plain attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(StableRankStateDictHook(fn))
        module._register_load_state_dict_pre_hook(
            StableRankLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class StableRankLoadStateDictPreHook(object):
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('stable_rank', {}).get(
            fn.name + '.version', None)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all(weight_key + s in state_dict
                                       for s in ('_orig', '_u', '_v')) and \
                    weight_key not in state_dict:
                # Detect if it is the updated state dict
                # and just missing metadata.
                # This could happen if the users are crafting
                # a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + '_orig']
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class StableRankStateDictHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'stable_rank' not in local_metadata:
            local_metadata['stable_rank'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['stable_rank']:
            raise RuntimeError(
                "Unexpected key in metadata['stable_rank']: {}".format(key))
        local_metadata['stable_rank'][key] = self.fn._version


def stable_rank(module, name='weight', n_power_iterations=1, eps=1e-12,
                dim=None, rank=0.7):

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0

    StableRank.apply(module, name, n_power_iterations, dim, eps, rank=rank)
    return module


def remove_stable_rank(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, StableRank) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("stable_rank of '{}' not found in {}".format(
            name, module))

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, StableRankStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
