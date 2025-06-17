from __future__ import annotations
from inferno import Module
from inferno._internal import argtest
from inferno.learn import IndependentCellTrainer
from inferno.neural import Cell
from inferno.observe import (
    StateMonitor,
    CumulativeTraceReducer,
    NearestTraceReducer,
    PassthroughReducer,
)
import einops as ein
import torch
from typing import Any, Callable, Literal


class DelaySTDP(IndependentCellTrainer):
    r"""Pair-based spike-timing dependent plasticity trainer.

    Rather than recording trace values with amplitudes specified by the learning rates,
    this uses an amplitude of 1. With some testing the difference appears to be minor.
    With limited testing, the maximum difference between this and the "unstable"
    implementation is around 2e-6 times the average weight. Not included with the
    default exports.

    .. math::
        w(t + \Delta t) - w(t) = \eta_\text{post} x_\text{pre}(t) \bigl[t = t^f_\text{post}\bigr]
        + eta_\text{pre} x_\text{post}(t) \bigl[t = t^f_\text{pre}\bigr]

    When ``trace_mode = "cumulative"``:

    .. math::
        \begin{align*}
            x_\text{pre}(t) &= x_\text{pre}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau_\text{pre}}\right)
            + \left[t = t_\text{pre}^f\right] \\
            x_\text{post}(t) &= x_\text{post}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau_\text{post}}\right)
            + \left[t = t_\text{post}^f\right]
        \end{align*}

    When ``trace_mode = "nearest"``:

    .. math::
        \begin{align*}
            x_\text{pre}(t) &=
            \begin{cases}
                1 & t = t_\text{pre}^f \\
                x_\text{pre}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_\text{pre}}\right)
                & t \neq t_\text{pre}^f
            \end{cases} \\
            x_\text{post}(t) &=
            \begin{cases}
                1 & t = t_\text{post}^f \\
                x_\text{post}(t - \Delta t)
                \exp\left(-\frac{\Delta t}{\tau_\text{post}}\right)
                & t \neq t_\text{post}^f
            \end{cases}
        \end{align*}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    The signs of the learning rates :math:`\eta_\text{post}` and :math:`\eta_\text{pre}`
    control which terms are potentiative and which terms are depressive. The terms can
    be scaled for weight dependence on updating.

    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Mode              | :math:`\text{sgn}(\eta_\text{post})` | :math:`\text{sgn}(\eta_\text{pre})` | LTP Term(s)                               | LTD Term(s)                               |
    +===================+======================================+=====================================+===========================================+===========================================+
    | Hebbian           | :math:`+`                            | :math:`-`                           | :math:`\eta_\text{post}`                  | :math:`\eta_\text{pre}`                   |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Anti-Hebbian      | :math:`-`                            | :math:`+`                           | :math:`\eta_\text{pre}`                   | :math:`\eta_\text{post}`                  |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Potentiative Only | :math:`+`                            | :math:`+`                           | :math:`\eta_\text{post}, \eta_\text{pre}` | None                                      |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+
    | Depressive Only   | :math:`-`                            | :math:`-`                           | None                                      | :math:`\eta_\text{post}, \eta_\text{pre}` |
    +-------------------+--------------------------------------+-------------------------------------+-------------------------------------------+-------------------------------------------+

    Args:
        lr_post (float): learning rate for updates on postsynaptic spikes,
            :math:`\eta_\text{post}`.
        lr_pre (float): learning rate for updates on presynaptic spikes,
            :math:`\eta_\text{pre}`.
        tc_post (float): time constant of exponential decay of postsynaptic trace,
            :math:`\tau_\text{post}`, in :math:`ms`.
        tc_pre (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau_\text{pre}`, in :math:`ms`.
        delayed (bool, optional): if the updater should assume that learned delays, if
            present, may change. Defaults to ``False``.
        interp_tolerance (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        trace_mode (Literal["cumulative", "nearest"], optional): method to use for
            calculating spike traces. Defaults to ``"cumulative"``.
        batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None):
            function to reduce updates over the batch dimension, :py:func:`torch.mean`
            when ``None``. Defaults to ``None``.
        inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
            should be performed in-place. Defaults to ``False``.

    Important:
        When ``delayed`` is ``True``, the history for the presynaptic activity (spike
        traces and spike activity) is preserved in its un-delayed form and is then
        accessed using the connection's :py:attr:`~inferno.neural.Connection.selector`.

        When ``delayed`` is ``False``, only the most recent delay-adjusted presynaptic
        activity is preserved.

    Important:
        It is expected for this to be called after every trainable batch. Variables
        used are not stored (or are invalidated) if multiple batches are given before
        an update.

    Note:
        The constructor arguments are hyperparameters and can be overridden on a
        cell-by-cell basis.

    Note:
        ``batch_reduction`` can be one of the functions in PyTorch including but not
        limited to :py:func:`torch.sum`, :py:func:`torch.mean`, and :py:func:`torch.amax`.
        A custom function with similar behavior can also be passed in. Like with the
        included function, it should not keep the original dimensions by default.

    See Also:
        For more details and references, visit
        :ref:`zoo/learning-stdp:Spike-Timing Dependent Plasticity (STDP)` in the zoo.
    """

    def __init__(
        self,
        weight_lr_post: float,
        weight_lr_pre: float,
        delay_lr_post: float,
        delay_lr_pre: float,
        tc_post: float,
        tc_pre: float,
        interp_tolerance: float = 0.0,
        trace_mode: Literal["cumulative", "nearest"] = "cumulative",
        batch_reduction: (
            Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor] | None
        ) = None,
        inplace: bool = False,
        **kwargs,
    ):
        # call superclass constructor
        IndependentCellTrainer.__init__(self, **kwargs)

        # default hyperparameters
        self.weight_lr_post = float(weight_lr_post)
        self.weight_lr_pre = float(weight_lr_pre)
        self.delay_lr_post = float(delay_lr_post)
        self.delay_lr_pre = float(delay_lr_pre)
        self.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        self.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        weight_lr_post = kwargs.get("weight_lr_post", self.weight_lr_post)
        weight_lr_pre = kwargs.get("weight_lr_pre", self.weight_lr_pre)
        delay_lr_post = kwargs.get("delay_lr_post", self.delay_lr_post)
        delay_lr_pre = kwargs.get("delay_lr_pre", self.delay_lr_pre)
        tc_post = kwargs.get("tc_post", self.tc_post)
        tc_pre = kwargs.get("tc_pre", self.tc_pre)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.weight_lr_post = float(weight_lr_post)
        state.weight_lr_pre = float(weight_lr_pre)
        state.delay_lr_post = float(delay_lr_post)
        state.delay_lr_pre = float(delay_lr_pre)
        state.tc_post = argtest.gt("tc_post", tc_post, 0, float)
        state.tc_pre = argtest.gt("tc_pre", tc_pre, 0, float)
        state.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        state.tracemode = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        match state.tracemode:
            case "cumulative":
                state.tracecls = CumulativeTraceReducer
            case "nearest":
                state.tracecls = NearestTraceReducer
            case "_":
                raise RuntimeError(
                    f"an invalid trace mode of '{state.tracemode}' has been set, "
                    "expected one of: 'cumulative', 'nearest'"
                )
        state.batchreduce = (
            batch_reduction if (batch_reduction is not None) else torch.mean
        )
        state.inplace = bool(inplace)

        return state

    def register_cell(
        self,
        name: str,
        cell: Cell,
        /,
        **kwargs: Any,
    ) -> IndependentCellTrainer.Unit:
        r"""Adds a cell with required state.

        Args:
            name (str): name of the cell to add.
            cell (Cell): cell to add.

        Keyword Args:
            lr_post (float): learning rate for updates on postsynaptic spikes.
            lr_pre (float): learning rate for updates on presynaptic spikes.
            tc_post (float): time constant of exponential decay of postsynaptic trace.
            tc_pre (float): time constant of exponential decay of presynaptic trace.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool, optional): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place. Defaults to ``False``.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`STDP` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight", "delay"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic trace monitor (weighs hebbian LTD)
        self.add_monitor(
            name,
            "trace_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_post,
                    amplitude=1.0,
                    target=True,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_post,
            trace=state.tracemode,
            inplace=state.inplace,
        )

        # postsynaptic spike monitor (triggers hebbian LTP)
        self.add_monitor(
            name,
            "spike_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        # presynaptic trace monitor (weighs hebbian LTP)
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.tc_pre,
                    amplitude=1.0,
                    target=True,
                    duration=cell.connection.delayedby,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            tc=state.tc_pre,
            trace=state.tracemode,
            inplace=state.inplace,
        )

        # presynaptic spike monitor (triggers hebbian LTD)
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=cell.connection.delayedby,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(self) -> None:
        r"""Processes update for given layers based on current monitor stored data."""
        # iterate through self
        for cell, state, monitors in self:
            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # spike traces, reshaped into receptive format
            x_post = cell.connection.postsyn_receptive(monitors["trace_post"].peek())
            x_pre = cell.connection.presyn_receptive(
                monitors["trace_pre"].view(cell.connection.selector, state.tolerance)
            )

            # spike presence, reshaped into receptive format
            i_post = cell.connection.postsyn_receptive(monitors["spike_post"].peek())
            i_pre = cell.connection.presyn_receptive(
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
            )

            # partial updates
            dwpost = state.batchreduce(
                ein.einsum(
                    i_post,
                    x_pre * abs(state.weight_lr_post),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dwpre = state.batchreduce(
                ein.einsum(
                    i_pre,
                    x_post * abs(state.weight_lr_pre),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            ddpost = state.batchreduce(
                ein.einsum(
                    i_post,
                    x_pre * abs(state.delay_lr_post),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            ddpre = state.batchreduce(
                ein.einsum(
                    i_pre,
                    x_post * abs(state.delay_lr_pre),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials with mode condition
            match (state.weight_lr_post >= 0, state.weight_lr_pre >= 0):
                case (False, False):  # depressive
                    cell.updater.weight = (None, dwpost + dwpre)
                case (False, True):  # anti-hebbian
                    cell.updater.weight = (dwpre, dwpost)
                case (True, False):  # hebbian
                    cell.updater.weight = (dwpost, dwpre)
                case (True, True):  # potentiative
                    cell.updater.weight = (dwpost + dwpre, None)

            match (state.delay_lr_post >= 0, state.delay_lr_pre >= 0):
                case (False, False):  # depressive
                    cell.updater.delay = (None, ddpost + ddpre)
                case (False, True):  # anti-hebbian
                    cell.updater.delay = (ddpre, ddpost)
                case (True, False):  # hebbian
                    cell.updater.delay = (ddpost, ddpre)
                case (True, True):  # potentiative
                    cell.updater.delay = (ddpost + ddpre, None)
