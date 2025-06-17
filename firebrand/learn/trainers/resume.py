from inferno import Module
from inferno._internal import argtest
from inferno.learn import IndependentCellTrainer
from inferno.neural import Cell
from inferno.observe import (
    CumulativeTraceReducer,
    NearestTraceReducer,
    StateMonitor,
    PassthroughReducer,
)

import einops as ein
import torch
from typing import Any, Callable, Literal


class ExponentialReSuMe(IndependentCellTrainer):
    r"""Remote supervised method trainer using an exponential kernel.

    .. math::
        w(t + \Delta t) - w(t) = \bigl[\bigl[t = t^f_\text{target}\bigr] - \bigl[t = t^f_\text{post}\bigr]\bigr]
        x_\text{pre}(t)

    When ``trace_mode = "cumulative"``:

    .. math::
        x_\text{pre}(t) = x_\text{pre}(t - \Delta t)
        \exp\left(-\frac{\Delta t}{\tau}\right) +
        \eta \left[t = t_\text{pre}^f\right]

    When ``trace_mode = "nearest"``:

    .. math::
        x_\text{pre}(t) =
        \begin{cases}
            \eta & t = t_\text{pre}^f \\
            x_\text{pre}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau}\right)
            & t \neq t_\text{pre}^f
        \end{cases}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        learning_rate (float): learning rate for updates when the target and postsynaptic
            spikes differ, :math:`\eta`.
        time_constant (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau`, in :math:`ms`.
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
        :ref:`zoo/learning-resume:Exponential Remote Supervised Method (Exponential ReSuMe)` in the zoo.
    """

    def __init__(
        self,
        learning_rate: float,
        time_constant: float,
        delayed: bool = False,
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
        self.learning_rate = float(learning_rate)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.delayed = bool(delayed)
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

        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        time_constant = kwargs.get("time_constant", self.time_constant)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.learning_rate = float(learning_rate)
        state.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        state.delayed = bool(delayed)
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
            learning_rate (float): learning rate for updates when the target and postsynaptic
                spikes differ.
            time_constant (float): time constant of exponential decay of presynaptic trace.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`ExponentialReSuMe` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # if delays should be accounted for
        delayed = state.delayed and cell.connection.delayedby is not None

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic spike monitor
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

        # presynaptic trace monitor
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=abs(state.learning_rate),
                    target=True,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.learning_rate),
            tc=state.time_constant,
            trace=state.tracemode,
            delayed=delayed,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(
        self,
        target: dict[str, torch.Tensor],
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        Args:
            target (dict[str, torch.Tensor]): dictionary with the names of the cells
                to update (keys) and the corresponding target spike trains (values).

        .. admonition:: Shape
            :class: tensorshape

            ``target[cell]``:

            ``cell.connection.batched_outshape``
        """
        # iterate through targets
        for name, tgtspike in target.items():
            # get corresponding unit
            unit = self.get_unit(name)
            cell, state, monitors = unit.cell, unit.state, unit.monitors

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # reshape spike events and traces
            i_post = cell.connection.postsyn_receptive(
                monitors["spike_post"].peek().bool()
            )
            i_tgt = cell.connection.postsyn_receptive(tgtspike)
            x_pre = cell.connection.presyn_receptive(
                monitors["trace_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre"].peek()
            )

            # partial updates
            dpos = state.batchreduce(
                ein.einsum(
                    (i_tgt & ~i_post).to(dtype=x_pre.dtype),
                    x_pre,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dneg = state.batchreduce(
                ein.einsum(
                    (~i_tgt & i_post).to(dtype=x_pre.dtype),
                    x_pre,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials, split by sign, flipping for negative learning rates
            if state.learning_rate >= 0:
                cell.updater.weight = (dpos, dneg)
            else:
                cell.updater.weight = (dneg, dpos)


class ExponentialInverseReSuMe(IndependentCellTrainer):
    r"""Remote supervised method trainer using an exponential kernel.

    .. math::
        w(t + \Delta t) - w(t) = \bigl[\bigl[t = t^f_\text{target}\bigr] - \bigl[t = t^f_\text{post}\bigr]\bigr]
        x_\text{pre}(t)

    When ``trace_mode = "cumulative"``:

    .. math::
        x_\text{pre}(t) = x_\text{pre}(t - \Delta t)
        \exp\left(-\frac{\Delta t}{\tau}\right) +
        \eta \left[t = t_\text{pre}^f\right]

    When ``trace_mode = "nearest"``:

    .. math::
        x_\text{pre}(t) =
        \begin{cases}
            \eta & t = t_\text{pre}^f \\
            x_\text{pre}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau}\right)
            & t \neq t_\text{pre}^f
        \end{cases}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        learning_rate (float): learning rate for updates when the target and postsynaptic
            spikes differ, :math:`\eta`.
        time_constant (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau`, in :math:`ms`.
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
        :ref:`zoo/learning-resume:Exponential Remote Supervised Method (Exponential ReSuMe)` in the zoo.
    """

    def __init__(
        self,
        learning_rate: float,
        time_constant: float,
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
        self.learning_rate = float(learning_rate)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
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

        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        time_constant = kwargs.get("time_constant", self.time_constant)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.learning_rate = float(learning_rate)
        state.time_constant = argtest.gt("time_constant", time_constant, 0, float)
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
            learning_rate (float): learning rate for updates when the target and postsynaptic
                spikes differ.
            time_constant (float): time constant of exponential decay of presynaptic trace.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`ExponentialReSuMe` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic trace monitor
        self.add_monitor(
            name,
            "trace_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=abs(state.learning_rate),
                    target=True,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.learning_rate),
            tc=state.time_constant,
            trace=state.tracemode,
            inplace=state.inplace,
        )

        # presynaptic spike monitor
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "connection.synspike",
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

        # target trace monitor
        # this is part of the auxiliary state and not the normal monitor pool
        state.trace_reducer = state.tracecls(
            cell.connection.dt,
            state.time_constant,
            amplitude=state.learning_rate,
            target=True,
            duration=0.0,
            inclusive=True,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(
        self,
        target: dict[str, torch.Tensor],
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        Args:
            target (dict[str, torch.Tensor]): dictionary with the names of the cells
                to update (keys) and the corresponding target spike trains (values).

        .. admonition:: Shape
            :class: tensorshape

            ``target[cell]``:

            ``cell.connection.batched_outshape``
        """
        # iterate through targets
        for name, tgtspike in target.items():
            # get corresponding unit
            unit = self.get_unit(name)
            cell, state, monitors = unit.cell, unit.state, unit.monitors

            # record new trace (occurs even when not in training mode)
            state.trace_reducer(tgtspike)

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # reshape spike events and traces
            x_post = cell.connection.postsyn_receptive(monitors["trace_post"].peek())
            x_tgt = cell.connection.postsyn_receptive(state.trace_reducer.peek())
            i_pre = cell.connection.presyn_receptive(monitors["spike_pre"].peek())

            # partial updates
            dpos = state.batchreduce(
                ein.einsum(
                    (x_tgt - x_post).clamp_min(0.0),
                    i_pre,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dneg = state.batchreduce(
                ein.einsum(
                    (x_post - x_tgt).clamp_min(0.0),
                    i_pre,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials, split by sign, flipping for negative learning rates
            if state.learning_rate >= 0:
                cell.updater.weight = (dpos, dneg)
            else:
                cell.updater.weight = (dneg, dpos)


class ExponentialPreSuMe(IndependentCellTrainer):
    r"""Presynaptic remote supervised method trainer using an exponential kernel.

    .. math::
        w(t + \Delta t) - w(t) = \left[t = t^f_\text{post}\right] \left(x_\text{pre}(t)
        - x_\text{target}(t)\right)

    When ``trace_mode = "cumulative"``:

    .. math::
        x_n(t) = x_n(t - \Delta t)
        \exp\left(-\frac{\Delta t}{\tau}\right) +
        \eta \left[t = t_n^f\right]

    When ``trace_mode = "nearest"``:

    .. math::
        x_n(t) =
        \begin{cases}
            \eta & t = t_n^f \\
            x_n(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau}\right)
            & t \neq t_n^f
        \end{cases}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        learning_rate (float): learning rate for updates when the target and postsynaptic
            spikes differ, :math:`\eta`.
        time_constant (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau`, in :math:`ms`.
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
        :ref:`zoo/learning-resume:Exponential Presynaptic Remote Supervised Method (Exponential PreSuMe)`
        in the zoo.
    """

    def __init__(
        self,
        learning_rate: float,
        time_constant: float,
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
        self.learning_rate = argtest.gt("learning_rate", learning_rate, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def clear(self, **kwargs) -> None:
        r"""Clears all of the monitors for the trainer.

        Note:
            Keyword arguments are passed to :py:meth:`~inferno.observe.Monitor.clear`
            and :py:meth:`~inferno.observe.Reducer.clear` calls.
        """
        # superclass clear
        IndependentCellTrainer.clear(self, **kwargs)

        # clear presynaptic target trace reducers
        for _, state, _ in self:
            state.trace_reducer.clear(**kwargs)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        time_constant = kwargs.get("time_constant", self.time_constant)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.learning_rate = argtest.gt("learning_rate", learning_rate, 0, float)
        state.time_constant = argtest.gt("time_constant", time_constant, 0, float)
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
            learning_rate (float): learning rate for updates when the target and postsynaptic
                spikes differ.
            time_constant (float): time constant of exponential decay of presynaptic trace.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`ExponentialReSuMe` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic spike monitor
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

        # presynaptic trace monitor
        self.add_monitor(
            name,
            "trace_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=state.learning_rate,
                    target=True,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=state.learning_rate,
            tc=state.time_constant,
            trace=state.tracemode,
            inplace=state.inplace,
        )

        # target trace monitor
        # this is part of the auxiliary state and not the normal monitor pool
        state.trace_reducer = state.tracecls(
            cell.connection.dt,
            state.time_constant,
            amplitude=state.learning_rate,
            target=True,
            duration=0.0,
            inclusive=True,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(
        self,
        target: dict[str, torch.Tensor],
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        Args:
            target (dict[str, torch.Tensor]): dictionary with the names of the cells
                to update (keys) and the corresponding target presynaptic spike trains (values).

        .. admonition:: Shape
            :class: tensorshape

            ``target[cell]``:

            ``cell.connection.batched_outshape``
        """
        # iterate through targets
        for name, tgtspike in target.items():
            # get corresponding unit
            unit = self.get_unit(name)
            cell, state, monitors = unit.cell, unit.state, unit.monitors

            # record new trace (occurs even when not in training mode)
            state.trace_reducer(cell.connection.like_synaptic(tgtspike))

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # reshape spike events and traces
            i_post = cell.connection.postsyn_receptive(
                monitors["spike_post"].peek().bool()
            )
            x_tgt = cell.connection.presyn_receptive(state.trace_reducer.peek())
            x_pre = cell.connection.presyn_receptive(monitors["trace_pre"].peek())

            # compute update scale
            scale = x_pre - x_tgt

            # partial updates
            dpos = state.batchreduce(
                ein.einsum(
                    i_post.to(dtype=x_pre.dtype),
                    scale.clamp_min(0.0),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dneg = state.batchreduce(
                ein.einsum(
                    i_post.to(dtype=x_pre.dtype),
                    scale.clamp_max(0.0).abs(),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials, split by sign
            cell.updater.weight = (dpos, dneg)


class ExponentialRemoteAlignment(IndependentCellTrainer):
    r"""Aligned remote supervised method trainer using an exponential kernel.

    .. math::
        w(t + \Delta t) - w(t) =
        \left(\left[t = t^f_\text{target}\right]
        - \left[t = t^f_\text{post}\right]\right)
        \left(x_\text{pre}(t)
        - x_\text{target}(t)\right)

    When ``trace_mode = "cumulative"``:

    .. math::
        x_n(t) = x_n(t - \Delta t)
        \exp\left(-\frac{\Delta t}{\tau}\right) +
        \eta \left[t = t_n^f\right]

    When ``trace_mode = "nearest"``:

    .. math::
        x_n(t) =
        \begin{cases}
            \eta & t = t_n^f \\
            x_n(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau}\right)
            & t \neq t_n^f
        \end{cases}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        learning_rate (float): learning rate for updates when the target and postsynaptic
            spikes differ, :math:`\eta`.
        time_constant (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau`, in :math:`ms`.
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
        :ref:`zoo/learning-resume:Exponential Presynaptic Remote Supervised Method (Exponential PreSuMe)`
        in the zoo.
    """

    def __init__(
        self,
        learning_rate: float,
        time_constant: float,
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
        self.learning_rate = argtest.gt("learning_rate", learning_rate, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.tolerance = argtest.gte("interp_tolerance", interp_tolerance, 0, float)
        self.trace = argtest.oneof(
            "trace_mode", trace_mode, "cumulative", "nearest", op=(lambda x: x.lower())
        )
        self.batchreduce = batch_reduction if batch_reduction else torch.mean
        self.inplace = bool(inplace)

    def clear(self, **kwargs) -> None:
        r"""Clears all of the monitors for the trainer.

        Note:
            Keyword arguments are passed to :py:meth:`~inferno.observe.Monitor.clear`
            and :py:meth:`~inferno.observe.Reducer.clear` calls.
        """
        # superclass clear
        IndependentCellTrainer.clear(self, **kwargs)

        # clear presynaptic target trace reducers
        for _, state, _ in self:
            state.trace_reducer.clear(**kwargs)

    def _build_cell_state(self, **kwargs) -> Module:
        r"""Builds auxiliary state for a cell.

        Keyword arguments will override module-level hyperparameters.

        Returns:
            Module: state module.
        """
        state = Module()

        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        time_constant = kwargs.get("time_constant", self.time_constant)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.learning_rate = argtest.gt("learning_rate", learning_rate, 0, float)
        state.time_constant = argtest.gt("time_constant", time_constant, 0, float)
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
            learning_rate (float): learning rate for updates when the target and postsynaptic
                spikes differ.
            time_constant (float): time constant of exponential decay of presynaptic trace.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`ExponentialReSuMe` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic spike monitor
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

        # presynaptic trace monitor
        self.add_monitor(
            name,
            "trace_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=state.learning_rate,
                    target=True,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=state.learning_rate,
            tc=state.time_constant,
            trace=state.tracemode,
            inplace=state.inplace,
        )

        # target trace monitor
        # this is part of the auxiliary state and not the normal monitor pool
        state.trace_reducer = state.tracecls(
            cell.connection.dt,
            state.time_constant,
            amplitude=state.learning_rate,
            target=True,
            duration=0.0,
            inclusive=True,
            inplace=state.inplace,
        )

        return self.get_unit(name)

    def forward(
        self,
        target: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        r"""Processes update for given layers based on current monitor stored data.

        Args:
            target (dict[str, tuple[torch.Tensor, torch.Tensor]]): dictionary with the names of the cells
                to update (keys) and the corresponding target presynaptic and postsynaptic spike trains (values).

        .. admonition:: Shape
            :class: tensorshape

            ``target[cell][0]``:

            ``cell.connection.batched_inshape``

            ``target[cell][1]``:

            ``cell.connection.batched_outshape``
        """
        # iterate through targets
        for name, (pre_tgt, post_tgt) in target.items():
            # get corresponding unit
            unit = self.get_unit(name)
            cell, state, monitors = unit.cell, unit.state, unit.monitors

            # record new trace (occurs even when not in training mode)
            state.trace_reducer(cell.connection.like_synaptic(pre_tgt))

            # skip if self or cell is not in training mode or has no updater
            if not cell.training or not self.training or not cell.updater:
                continue

            # reshape spike events and traces
            i_post = cell.connection.postsyn_receptive(
                monitors["spike_post"].peek().bool()
            )
            i_post_tgt = cell.connection.postsyn_receptive(post_tgt)
            x_pre_tgt = cell.connection.presyn_receptive(state.trace_reducer.peek())
            x_pre = cell.connection.presyn_receptive(monitors["trace_pre"].peek())

            # compute intermediates
            scale = x_pre - x_pre_tgt
            posscale = scale.clamp_min(0.0)
            negscale = scale.clamp_max(0.0).abs()

            # partial updates
            dpospos = state.batchreduce(
                ein.einsum(
                    (i_post_tgt & ~i_post).to(dtype=x_pre.dtype),
                    posscale,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dnegpos = state.batchreduce(
                ein.einsum(
                    (~i_post_tgt & i_post).to(dtype=x_pre.dtype),
                    posscale,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dposneg = state.batchreduce(
                ein.einsum(
                    (i_post_tgt & ~i_post).to(dtype=x_pre.dtype),
                    negscale,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dnegneg = state.batchreduce(
                ein.einsum(
                    (~i_post_tgt & i_post).to(dtype=x_pre.dtype),
                    negscale,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials, split by sign
            cell.updater.weight = (dpospos + dnegneg, dposneg + dnegpos)


class ExponentialAutoReSuMe(IndependentCellTrainer):
    r"""Remote supervised method trainer using an exponential kernel.

    .. math::
        w(t + \Delta t) - w(t) = \bigl[\bigl[t = t^f_\text{target}\bigr] - \bigl[t = t^f_\text{post}\bigr]\bigr]
        x_\text{pre}(t)

    When ``trace_mode = "cumulative"``:

    .. math::
        x_\text{pre}(t) = x_\text{pre}(t - \Delta t)
        \exp\left(-\frac{\Delta t}{\tau}\right) +
        \eta \left[t = t_\text{pre}^f\right]

    When ``trace_mode = "nearest"``:

    .. math::
        x_\text{pre}(t) =
        \begin{cases}
            \eta & t = t_\text{pre}^f \\
            x_\text{pre}(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau}\right)
            & t \neq t_\text{pre}^f
        \end{cases}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        learning_rate (float): learning rate for updates when the target and postsynaptic
            spikes differ, :math:`\eta`.
        time_constant (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau`, in :math:`ms`.
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
        :ref:`zoo/learning-resume:Exponential Remote Supervised Method (Exponential ReSuMe)` in the zoo.
    """

    def __init__(
        self,
        learning_rate: float,
        time_constant: float,
        delayed: bool = False,
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
        self.learning_rate = float(learning_rate)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        self.delayed = bool(delayed)
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

        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        time_constant = kwargs.get("time_constant", self.time_constant)
        delayed = kwargs.get("delayed", self.delayed)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.learning_rate = float(learning_rate)
        state.time_constant = argtest.gt("time_constant", time_constant, 0, float)
        state.delayed = bool(delayed)
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
            learning_rate (float): learning rate for updates when the target and postsynaptic
                spikes differ.
            time_constant (float): time constant of exponential decay of presynaptic trace.
            delayed (bool): if the updater should assume that learned delays,
                if present, may change.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`ExponentialReSuMe` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # if delays should be accounted for
        delayed = state.delayed and cell.connection.delayedby is not None

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic spike monitor
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

        # presynaptic spike monitor
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "spike_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=PassthroughReducer(
                    cell.connection.dt,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            delayed=delayed,
            inplace=state.inplace,
        )

        # presynaptic trace monitor
        # when the delayed condition is true, using synapse.spike records the raw
        # spike times rather than the delay adjusted times of synspike.
        self.add_monitor(
            name,
            "trace_pre",
            "synapse.spike" if delayed else "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=abs(state.learning_rate),
                    target=True,
                    duration=cell.connection.delayedby if delayed else 0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=abs(state.learning_rate),
            tc=state.time_constant,
            trace=state.tracemode,
            delayed=delayed,
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
            x_pre = cell.connection.presyn_receptive(
                monitors["trace_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["trace_pre"].peek()
            )

            # spike presence, reshaped into receptive format
            i_post = cell.connection.postsyn_receptive(
                monitors["spike_post"].peek()
            ).bool()
            i_pre = cell.connection.presyn_receptive(
                monitors["spike_pre"].view(cell.connection.selector, state.tolerance)
                if state.delayed and cell.connection.delayedby
                else monitors["spike_pre"].peek()
            ).bool()

            # partial updates
            dpos = state.batchreduce(
                ein.einsum(
                    (i_pre & ~i_post).to(dtype=x_pre.dtype),
                    x_pre,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dneg = state.batchreduce(
                ein.einsum(
                    (~i_pre & i_post).to(dtype=x_pre.dtype),
                    x_pre,
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials, split by sign, flipping for negative learning rates
            if state.learning_rate >= 0:
                cell.updater.weight = (dpos, dneg)
            else:
                cell.updater.weight = (dneg, dpos)


class ExponentialAutoPReSuMe(IndependentCellTrainer):
    r"""Presynaptic remote supervised method trainer using an exponential kernel.

    .. math::
        w(t + \Delta t) - w(t) = \left[t = t^f_\text{post}\right] \left(x_\text{pre}(t)
        - x_\text{target}(t)\right)

    When ``trace_mode = "cumulative"``:

    .. math::
        x_n(t) = x_n(t - \Delta t)
        \exp\left(-\frac{\Delta t}{\tau}\right) +
        \eta \left[t = t_n^f\right]

    When ``trace_mode = "nearest"``:

    .. math::
        x_n(t) =
        \begin{cases}
            \eta & t = t_n^f \\
            x_n(t - \Delta t)
            \exp\left(-\frac{\Delta t}{\tau}\right)
            & t \neq t_n^f
        \end{cases}

    Where:

    Times :math:`t` and :math:`t_n^f` are the current time and the time of the most
    recent spike from neuron :math:`n`, respectively, and :math:`\Delta t` is the
    duration of the simulation step.

    Args:
        learning_rate (float): learning rate for updates when the target and postsynaptic
            spikes differ, :math:`\eta`.
        time_constant (float): time constant of exponential decay of presynaptic trace,
            :math:`\tau`, in :math:`ms`.
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
        :ref:`zoo/learning-resume:Exponential Presynaptic Remote Supervised Method (Exponential PreSuMe)`
        in the zoo.
    """

    def __init__(
        self,
        learning_rate: float,
        time_constant: float,
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
        self.learning_rate = argtest.gt("learning_rate", learning_rate, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)
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

        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        time_constant = kwargs.get("time_constant", self.time_constant)
        interp_tolerance = kwargs.get("interp_tolerance", self.tolerance)
        trace_mode = kwargs.get("trace_mode", self.trace)
        batch_reduction = kwargs.get("batch_reduction", self.batchreduce)
        inplace = kwargs.get("inplace", self.inplace)

        state.learning_rate = argtest.gt("learning_rate", learning_rate, 0, float)
        state.time_constant = argtest.gt("time_constant", time_constant, 0, float)
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
            learning_rate (float): learning rate for updates when the target and postsynaptic
                spikes differ.
            time_constant (float): time constant of exponential decay of presynaptic trace.
            interp_tolerance (float): maximum difference in time from an observation
                to treat as co-occurring.
            trace_mode (Literal["cumulative", "nearest"]): method to use for
                calculating spike traces.
            batch_reduction (Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]):
                function to reduce updates over the batch dimension.
            inplace (bool): if :py:class:`~inferno.RecordTensor` write operations
                should be performed in-place.

        Returns:
            IndependentCellTrainer.Unit: specified cell, auxiliary state, and monitors.

        Important:
            Any specified keyword arguments will override the default hyperparameters
            set on initialization. See :py:class:`ExponentialReSuMe` for details.
        """
        # add the cell with additional hyperparameters
        cell, state = self.add_cell(
            name, cell, self._build_cell_state(**kwargs), ["weight"]
        )

        # common and derived arguments
        monitor_kwargs = {
            "as_prehook": False,
            "train_update": True,
            "eval_update": False,
            "prepend": True,
        }

        # postsynaptic spike monitor
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

        # presynaptic trace monitor
        self.add_monitor(
            name,
            "trace_pre",
            "connection.synspike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=state.learning_rate,
                    target=True,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=state.learning_rate,
            tc=state.time_constant,
            trace=state.tracemode,
            inplace=state.inplace,
        )

        # postsynaptic trace monitor
        self.add_monitor(
            name,
            "trace_post",
            "neuron.spike",
            StateMonitor.partialconstructor(
                reducer=state.tracecls(
                    cell.connection.dt,
                    state.time_constant,
                    amplitude=state.learning_rate,
                    target=True,
                    duration=0.0,
                    inclusive=True,
                    inplace=state.inplace,
                ),
                **monitor_kwargs,
            ),
            False,
            dt=cell.connection.dt,
            amp=state.learning_rate,
            tc=state.time_constant,
            trace=state.tracemode,
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

            # reshape spike events and traces
            i_post = cell.connection.postsyn_receptive(
                monitors["spike_post"].peek().bool()
            )
            x_post = cell.connection.postsyn_receptive(monitors["trace_post"].peek())
            x_pre = cell.connection.presyn_receptive(monitors["trace_pre"].peek())

            # compute update scale
            scale = x_pre - x_post

            # partial updates
            dpos = state.batchreduce(
                ein.einsum(
                    i_post.to(dtype=x_pre.dtype),
                    scale.clamp_min(0.0),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )
            dneg = state.batchreduce(
                ein.einsum(
                    i_post.to(dtype=x_pre.dtype),
                    scale.clamp_max(0.0).abs(),
                    "b ... r, b ... r -> b ...",
                ),
                0,
            )

            # accumulate partials, split by sign
            cell.updater.weight = (dpos, dneg)
