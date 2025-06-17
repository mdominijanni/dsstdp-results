from inferno import Module
from inferno._internal import argtest
from collections.abc import Sequence
import einops as ein
import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstSpikeClassifier(Module):
    r"""Classifies spikes by shortest per-class time to first spike.

    The classifier uses an internal parameter :py:attr:`offsets` for other
    calculations. When learning, the existing offsets are decayed, multiplying them by
    :math:`\exp (-\lambda b_k)` where :math:`b_k` is the number of elements of class
    :math:`k` in the batch.
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        num_classes: int,
        duration: float,
        step_time: float,
        *,
        decay: float = 0.0,
    ):
        # call superclass constructor
        Module.__init__(self)

        # validate parameters
        try:
            shape = (argtest.gt("shape", shape, 0, int),)
        except TypeError:
            if isinstance(shape, Sequence):
                shape = argtest.ofsequence("shape", shape, argtest.gt, 0, int)
            else:
                raise TypeError(
                    f"'shape' ({argtest._typename(type(shape))}) cannot be interpreted "
                    "as an integer or a sequence thereof"
                )

        num_classes = argtest.gt("num_classes", num_classes, 0, int)
        self._step_time = argtest.gt("step_time", step_time, 0, float)
        self._duration = argtest.gt("duration", duration, 0, float)

        # register parameter
        self.register_parameter(
            "offsets_",
            nn.Parameter(torch.zeros(*shape, num_classes).float(), False),
        )

        # register derived buffers
        self.register_buffer(
            "assignments_", torch.zeros(*shape).long(), persistent=False
        )
        self.register_buffer(
            "occurrences_", torch.zeros(num_classes).long(), persistent=False
        )
        self.register_buffer(
            "proportions_", torch.zeros(*shape, num_classes).float(), persistent=False
        )

        # class attribute
        self.decay = argtest.gte("decay", decay, 0, float)

        # run after loading state_dict to recompute non-persistent buffers
        def sdhook(module, incompatible_keys) -> None:
            module.offsets = module.offsets

        self.register_load_state_dict_post_hook(sdhook)

    @property
    def assignments(self) -> torch.Tensor:
        r"""Class assignments per-neuron.

        The label, computed as the argument of the maximum of normalized offsets
        (proportions), per neuron.

        Returns:
            torch.Tensor: present class assignments per-neuron.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        return self.assignments_

    @property
    def occurrences(self) -> torch.Tensor:
        r"""Number of assigned neurons per-class.

        The number of neurons which are assigned to each label.

        Returns:
            torch.Tensor: present number of assigned neurons per-class.

        .. admonition:: Shape
            :class: tensorshape

            :math:`K`

            Where:
                * :math:`K` is the number of possible classes.
        """
        return self.occurrences_

    @property
    def proportions(self) -> torch.Tensor:
        r"""Class-normalized spike offsets.

        The offsets :math:`L_1`-normalized such that for a given neuron, such that the
        normalized offsets for it over the different classes sum to 1.

        Returns:
            torch.Tensor: present class-normalized spike offsets.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots \times K`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        return self.proportions_

    @property
    def offsets(self) -> torch.Tensor:
        r"""Computed per-class, per-neuron spike time offsets.

        These are the raw time-to-first-spike offsets
        :math:`\left(\frac{\text{duration} - \text{TTFS}\right)}{\text{duration}}`
        for each neuron, per class. When there is no spike, the TTFS is set to the
        duration. This way larger values are associated with faster neurons.

        Args:
            value (torch.Tensor): new computed per-class, per-neuron spike time offsets.

        Returns:
            torch.Tensor: present computed per-class, per-neuron spike time offsets.

        Note:
            The attributes :py:attr:`proportions`, :py:attr:`assignments`, and
            :py:attr:`occurrences` are automatically recalculated on assignment.

        .. admonition:: Shape
            :class: tensorshape

            :math:`N_0 \times \cdots \times K`

            Where:
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        return self.offsets_.data

    @offsets.setter
    def offsets(self, value: torch.Tensor) -> None:
        # offsets are assigned directly
        self.offsets_.data = value
        self.proportions_ = F.normalize(self.offsets, p=1, dim=-1)
        self.assignments_ = torch.argmax(self.proportions, dim=-1)
        self.occurrences_ = torch.bincount(self.assignments.view(-1), None, self.nclass)

    @property
    def duration(self) -> float:
        r"""Length of the training window.

        Args:
            value (float): updated length of the training window.

        Returns:
            float: length of the training window.
        """
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        duration = argtest.gt("duration", value, 0, float)
        self.offsets = (self.offsets / self._duration) * duration
        self._duration = duration

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.

        Note:
            This is only used when spike trains are passed in rather than
            spike time offsets or times-to-first-spike.
        """
        return self._step_time

    @dt.setter
    def dt(self, value: float) -> None:
        self._step_time = argtest.gt("dt", value, 0, float)

    @property
    def ndim(self) -> int:
        r"""Number of dimensions of the spikes being classified, excluding batch and time.

        Returns:
            tuple[int, ...]: number of dimensions of the spikes being classified
        """
        return self.assignments.ndim

    @property
    def nclass(self) -> int:
        r"""Number of possible classes

        Returns:
            int: number of possible classes.
        """
        return self.occurrences.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the spikes being classified, excluding batch and time.

        Returns:
            tuple[int, ...]: shape of spikes being classified.
        """
        return tuple(self.assignments.shape)

    def regress(
        self, inputs: torch.Tensor, proportional: bool = True, from_ttfs: bool = True
    ) -> torch.Tensor:
        r"""Computes class logits from times-to-first-spike.

        Args:
            inputs (torch.Tensor): batched offsets or times-to-first-spike to classify.
            proportional (bool, optional): if inference is weighted by class-average
                offsets. Defaults to ``True``.
            from_ttfs (bool, optional): if inputs are given as times-to-first spike
                rather than as offsets. Defaults to ``True``.

        Returns:
            torch.Tensor: predicted logits.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B \times K`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.
        """
        # convert ttfs into offsets
        if from_ttfs:
            inputs = (self.duration - inputs) / self.duration

        # associations between neurons and classes
        if proportional:
            assocs = F.one_hot(self.assignments.view(-1), self.nclass) * ein.rearrange(
                self.proportions, "... k -> (...) k"
            )
        else:
            assocs = F.one_hot(self.assignments.view(-1), self.nclass).to(
                dtype=self.proportions_.dtype
            )

        # compute logits
        ylogits = (
            torch.mm(
                ein.rearrange(inputs, "b ... -> b (...)"),
                assocs,
            )
            .div(self.occurrences)
            .nan_to_num(nan=0, posinf=0)
        )

        # return logits or predictions
        return ylogits

    def classify(
        self, inputs: torch.Tensor, proportional: bool = True, from_ttfs: bool = True
    ) -> torch.Tensor:
        r"""Computes class labels from times-to-first spike.

        Args:
            inputs (torch.Tensor): batched offsets or times-to-first-spike to classify.
            proportional (bool, optional): if inference is weighted by class-average
                rates. Defaults to ``True``.
            from_ttfs (bool, optional): if inputs are given as times-to-first spike
                rather than as offsets. Defaults to ``True``.

        Returns:
            torch.Tensor: predicted labels.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``return``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        return torch.argmax(self.regress(inputs, proportional, from_ttfs), dim=1)

    def update(
        self, inputs: torch.Tensor, labels: torch.Tensor, from_ttfs: bool = True
    ) -> None:
        r"""Updates stored rates from times-to-first-spike and labels.

        Args:
            inputs (torch.Tensor): batched offsets or times-to-first-spike from which to update state.
            labels (torch.Tensor): ground-truth sample labels.
            from_ttfs (bool, optional): if inputs are given as times-to-first spike
                rather than as offsets. Defaults to ``True``.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`B \times N_0 \times \cdots`

            ``labels``:

            :math:`B`

            Where:
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
        """
        # convert ttfs into offsets
        if from_ttfs:
            inputs = (self.duration - inputs) / self.duration

        # number of instances per-class
        clscounts = torch.bincount(labels, None, self.nclass).to(
            dtype=self.offsets.dtype
        )

        # compute per-class scaled spike offsets
        offsets = (
            torch.scatter_add(
                torch.zeros_like(self.offsets),
                dim=-1,
                index=labels.expand(*self.shape, -1),
                src=ein.rearrange(inputs, "b ... -> ... b"),
            )
            / clscounts
        ).nan_to_num(nan=0, posinf=0)

        # update offsets, other properties update automatically
        self.offsets = torch.exp(-self.decay * clscounts) * self.offsets + offsets

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        logits: bool | None = False,
        proportional: bool = True,
        from_ttfs: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        r"""Performs inference and updates the classifier state.

        Args:
            inputs (torch.Tensor): spikes or times-to-first-spike to classify.
            labels (torch.Tensor | None): ground-truth sample labels.
            logits (bool | None, optional): if predicted class logits should be
                returned along with labels, inference is skipped if ``None``.
                Defaults to ``False``.
            proportional (bool, optional): if inference is weighted by class-average
                rates. Defaults to ``True``.
            from_ttfs (bool, optional): if non-spike inputs are given as times-to-first-spike
                rather than as offsets. Defaults to ``True``.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: predicted class
            labels, with unnormalized logits if specified.

        .. admonition:: Shape
            :class: tensorshape

            ``inputs``:

            :math:`[T] \times B \times N_0 \times \cdots`

            ``labels``:

            :math:`B`

            ``return (logits=False)``:

            :math:`B`

            ``return (logits=True)``:

            :math:`(B, B \times K)`

            Where:
                * :math:`T` is the number of simulation steps over which spikes were gathered.
                * :math:`B` is the batch size.
                * :math:`N_0, \ldots` are the dimensions of the spikes being classified.
                * :math:`K` is the number of possible classes.

        Important:
            This method will always perform the inference step prior to updating the classifier.

        Note:
            Offsets will always be computed from times-to-first-spike when a tensor without
            a time dimension is given.
        """
        # reduce along input time dimension, if present, to generate spike offsets
        if inputs.ndim == self.ndim + 2:
            inputs = inputs.max(0)
            inputs = (
                (inputs.values.to(dtype=self.offsets.dtype) * self.duration)
                - (inputs.indices.to(dtype=self.offsets.dtype) * self.dt)
            ) / self.duration
        elif from_ttfs:
            inputs = self.duration - inputs

        # inference
        if logits is None:
            res = None
        elif not logits:
            res = self.classify(inputs, proportional, False)
        else:
            res = self.regress(inputs, proportional, False)
            res = (torch.argmax(res, dim=1), res)

        # update
        if labels is not None:
            self.update(inputs, labels, False)

        # return inference
        return res
