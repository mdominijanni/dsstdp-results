import einops as ein
import math
import torch


@torch.no_grad()
def victor_purpura_dist(
    s0: torch.Tensor,
    s1: torch.Tensor,
    cost: float,
    step_time: float,
    time_first: bool = True,
) -> torch.Tensor:
    r"""Victor–Purpura distance between spike trains.

    CAUTION: This algorithm produces incorrect results in some cases. It doesn't look
    ahead and therefor may not find a better future option. Not currently fit for purpose.

    This function is not fully vectorized and may be slow. It take care when using it
    on performance critical pathways.

    Args:
        s0 (torch.Tensor): first set of spike trains.
        s1 (torch.Tensor): second set of spike trains.
        cost (float): cost to move a spike by one unit of time.
        step_time (float): length of time between each spike record (0 or 1).
        time_first (bool, optional): if the time dimension is given first rather than
            last. Defaults to ``True``.

    Returns:
        torch.Tensor: distance between the spike trains for each cost.

    .. admonition:: Shape
        :class: tensorshape

        ``s0``, ``s1``:

        :math:`T \times N_0 \times \cdots` or :math:`N_0 \times \cdots \times T`

        ``return``:

        :math:`N_0 \times \cdots`

        Where:
            * :math:`T` the length of the spike trains.
            * :math:`N_0, \ldots` shape of the generating population (batch, neuron shape, etc).

    Note:
        If ``s0`` and ``s1`` must have the same number of spike trains and the time
        dimension must be the same size, but the shape of the spike train dimensions
        can vary. The output will have the shape of ``s0`` excluding the time dimension.

    Note:
        Because Inferno spike trains are typically stored as boolean tensors, they
        will be internally cast to ``torch.int8`` for numerical operations.
    """
    # turn spike trains into T x N matrices
    if time_first:
        shape = tuple(s0.shape[1:])
        s0 = ein.rearrange(s0, "t ... -> t (...)")
        s1 = ein.rearrange(s1, "t ... -> t (...)")
    else:
        shape = tuple(s0.shape[:-1])
        s0 = ein.rearrange(s0, "... t -> t (...)")
        s1 = ein.rearrange(s1, "... t -> t (...)")

    numtr = math.prod(shape)

    # running distance total
    dist = torch.zeros(numtr, device=s0.device)

    # sign associated with adding to that stack (when zero, no stack)
    ssgn = torch.zeros(numtr, device=s0.device)

    # current stacks
    stack = [torch.tensor([], device=s0.device) for _ in range(numtr)]

    # iterate over spike diffs at each time step
    for diff in s0.to(dtype=torch.int8) - s1.to(dtype=torch.int8):
        # check for stack interactions
        for nz in torch.nonzero(diff)[..., -1]:
            # creating a new stack
            if ssgn[nz] == 0:
                ssgn[nz] = diff[nz]

            # push to stack
            if ssgn[nz] == diff[nz]:
                stack[nz] = torch.cat(
                    (stack[nz], torch.zeros(1, device=s0.device)), dim=0
                )

            # pop from stack
            else:
                d, stack[nz] = stack[nz][-1] * cost * step_time, stack[nz][:-1]

                # move is efficient
                if d <= 2:
                    # add the move distance
                    dist[nz] += d

                    # check and mark if the stack is empty now
                    if stack[nz].numel() == 0:
                        ssgn[nz] = 0

                # move is inefficient
                else:
                    # add the indel distance
                    dist[nz] += stack[nz].numel() + 1

                    # recreate stack with current element only
                    ssgn[nz] = diff[nz]
                    stack[nz] = torch.zeros(1, device=s0.device)

        # increment stack values
        for nz in torch.nonzero(ssgn)[..., -1]:
            stack[nz] += 1

    # add stack remainder to distance as indel cost
    for nz in torch.nonzero(ssgn)[..., -1]:
        dist[nz] += stack[nz].numel()

    return dist.view(*shape)


@torch.no_grad()
def victor_purpura_pair_dist(
    t0: torch.Tensor, t1: torch.Tensor, cost: float | torch.Tensor
) -> torch.Tensor:
    r"""Victor–Purpura distance between a pair of spike trains.

    This function is not fully vectorized and may be slow. It take care when using it
    on performance critical pathways.

    Uses a Needleman–Wunsch approach. Translated from the
    `MATLAB code <http://www-users.med.cornell.edu/~jdvicto/spkd_qpara.html>`_
    by Thomas Kreuz.

    Args:
        t0 (torch.Tensor): spike times of the first spike train.
        t1 (torch.Tensor): spike times of the second spike train.
        cost (float | torch.Tensor): cost to move a spike by one unit of time.

    Returns:
        torch.Tensor: distance between the spike trains for each cost.

    .. admonition:: Shape
        :class: tensorshape

        ``t0``:

        :math:`T_m`

        ``t1``:

        :math:`T_n`

        ``cost`` and ``return``:

        :math:`k`

        Where:
            * :math:`T_m` number of spikes in the first spike train.
            * :math:`T_n` number of spikes in the second spike train.
            * :math:`k`, number of cost values to compute distance for, treated
              as :math:`1` when ``cost`` is a float.

    Warning:
        As in the original algorithm, using ``inf`` as the cost will only return
        the total number of spikes, not accounting for spikes occurring at the same
        time in each spike train.
    """
    # check for cost edge conditions and make tensor if not
    if not isinstance(cost, torch.Tensor):
        if cost == 0.0:
            return torch.tensor([float(abs(t0.numel() - t1.numel()))], device=t0.device)
        elif cost == float("inf"):
            return torch.tensor([float(t0.numel() + t1.numel())], device=t0.device)
        else:
            cost = torch.tensor([float(cost)], device=t0.device)

    # create grid for Needleman–Wunsch
    tckwargs = {"dtype": cost.dtype, "device": cost.device}
    grid = torch.zeros(t0.numel() + 1, t1.numel() + 1, **tckwargs)
    grid[:, 0] = torch.arange(0, t0.numel() + 1, **tckwargs).t()
    grid[0, :] = torch.arange(0, t1.numel() + 1, **tckwargs).t()
    grid = grid.unsqueeze(0).repeat(cost.numel(), 1, 1)

    # dp algorithm
    for r in range(1, t0.numel() + 1):
        for c in range(1, t1.numel() + 1):
            c_add_a = grid[:, r - 1, c] + 1
            c_add_b = grid[:, r, c - 1] + 1
            c_shift = grid[:, r - 1, c - 1] + cost * torch.abs(t0[r - 1] - t1[c - 1])
            grid[:, r, c] = (
                torch.stack((c_add_a, c_add_b, c_shift), 0)
                .nan_to_num(nan=float("inf"))
                .amin(0)
            )

    # return result
    return grid[:, -1, -1]
