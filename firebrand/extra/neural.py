from inferno._internal import argtest
from inferno.neural import Neuron
from inferno.neural.neurons.mixins import SpikeRefractoryMixin
import torch
from typing import Literal


def refrac_from_spikes(
    spike: torch.Tensor, refrac: torch.Tensor, abs_refrac: float, step_time: float
) -> torch.Tensor:
    r"""Computes correct refractory periods to generate spikes.

    This works with :py:class:`~inferno.neural.Neuron` classes that use the
    :py:class:`~inferno.neural.neurons.mixins.SpikeRefractoryMixin` for determining
    output spike behavior.

    When used with :py:class:`OverridableNeuron`, the previous ``refrac`` is given, so
    here they are decremented and clamped as would occur during a normal simulation
    step. Then where ``spike`` is ``True``, values are set to ``abs_refrac``.

    Args:
        spike (torch.Tensor): desired spikes as the output for this time step.
        refrac (torch.Tensor): current refractory countdowns from the prior time step.
        abs_refrac (float): length of the absolute refractory period.
        step_time (float): length of the simulation step time.

    Returns:
        torch.Tensor: overriding refractory countdowns.
    """
    return torch.where(spike, abs_refrac, (refrac - step_time).clamp_min(0.0))


class OverridableNeuron(Neuron):
    def __init__(self, neuron: Neuron):
        # call superclass constructor
        Neuron.__init__(self)

        # register submodule
        self.neuron_ = neuron

    @property
    def neuron(self) -> Neuron:
        r"""Wrapped neuron.

        Returns:
            Neuron: wrapped neuron.
        """
        return self.neuron_

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the group of neurons.

        Returns:
            tuple[int, ...]: shape of the group of neurons.
        """
        return self.neuron_.shape

    @property
    def count(self) -> int:
        r"""Number of neurons in the group.

        Returns:
            int: number of neurons in the group.
        """
        return self.neuron_.count

    @property
    def batchsz(self) -> int:
        r"""Batch size of the module.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.
        """
        return self.neuron_.batchsz

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        self.neuron_.batchsz = value

    @property
    def batchedshape(self) -> tuple[int, ...]:
        r"""Batch shape of the module

        Returns:
            tuple[int, ...]: shape of the group of neurons, including batch size.
        """
        return self.neuron_.batchedshape

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.

        Raises:
            NotImplementedError: ``dt`` must be implemented by the subclass.
        """
        return self.neuron_.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.neuron_.dt = value

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.

        Raises:
            NotImplementedError: ``voltage`` must be implemented by the subclass.
        """
        return self.neuron_.voltage

    @voltage.setter
    def voltage(self, value: torch.Tensor) -> None:
        self.neuron_.voltage = value

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.

        Raises:
            NotImplementedError: ``refrac`` must be implemented by the subclass.
        """
        return self.neuron_.refrac

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> torch.Tensor:
        self.neuron_.refrac = value

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.

        Raises:
            NotImplementedError: ``spike`` must be implemented by the subclass.
        """
        return self.neuron_.spike

    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state.

        Raises:
            NotImplementedError: ``clear`` must be implemented by the subclass.
        """
        self.neuron_.clear(**kwargs)

    def forward(
        self,
        inputs: torch.Tensor,
        overrides: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        This calls the inner :py:class:`~inferno.neural.Neuron` with given ``inputs``,
        passing in keyword arguments. Tensors in ``overrides`` are then assigned to
        attributes of the inner neuron with corresponding names (this includes assignment
        via property setters). Then the inner neuron's :py:attr:`~inferno.neural.Neuron.spike`
        getter is called and its value returned.

        Args:
            inputs (torch.Tensor): input currents to the neurons.
            overrides (dict[str, torch.Tensor] | None, optional): assignment of tensors
                to the inner neuron. Defaults to None.

        Returns:
            torch.Tensor: postsynaptic spikes from integration of inputs.
        """
        if overrides is not None:
            _ = self.neuron_(inputs, **kwargs)

            for attr, value in overrides.items():
                setattr(self.neuron_, attr, value)

            return self.neuron_.spike

        else:
            return self.neuron_(inputs, **kwargs)


class RefracOverridableNeuron(Neuron):
    def __init__(self, neuron: Neuron, overrides: Literal["spike", "refrac"]):
        # check that neuron uses required mixin
        if not isinstance(neuron, SpikeRefractoryMixin):
            raise TypeError("'neuron' must subclass SpikeRefractoryMixin")

        # call superclass constructor
        Neuron.__init__(self)

        # register submodule
        self.neuron_ = neuron

        # set override mode
        overrides = argtest.oneof(
            "overrides", overrides, "spike", "refrac", op=(lambda x: x.lower())
        )
        self.use_spike_override = overrides == "spike"

        # register buffer
        self.register_buffer("origrefrac_", neuron.refrac.clone())
        self.register_buffer("origspike_", neuron.spike.clone())

    @property
    def neuron(self) -> Neuron:
        r"""Wrapped neuron.

        Returns:
            Neuron: wrapped neuron.
        """
        return self.neuron_

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the group of neurons.

        Returns:
            tuple[int, ...]: shape of the group of neurons.
        """
        return self.neuron_.shape

    @property
    def count(self) -> int:
        r"""Number of neurons in the group.

        Returns:
            int: number of neurons in the group.
        """
        return self.neuron_.count

    @property
    def batchsz(self) -> int:
        r"""Batch size of the module.

        Args:
            value (int): new batch size.

        Returns:
            int: present batch size.
        """
        return self.neuron_.batchsz

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        self.neuron_.batchsz = value
        self.origrefrac_ = self.neuron_.refrac.clone()
        self.origspike_ = self.neuron_.spike.clone()

    @property
    def batchedshape(self) -> tuple[int, ...]:
        r"""Batch shape of the module

        Returns:
            tuple[int, ...]: shape of the group of neurons, including batch size.
        """
        return self.neuron_.batchedshape

    @property
    def dt(self) -> float:
        r"""Length of the simulation time step, in milliseconds.

        Args:
            value (float): new simulation time step length.

        Returns:
            float: present simulation time step length.

        Raises:
            NotImplementedError: ``dt`` must be implemented by the subclass.
        """
        return self.neuron_.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.neuron_.dt = value
        self.origrefrac_ = self.neuron_.refrac.clone()
        self.origspike_ = self.neuron_.spike.clone()

    @property
    def voltage(self) -> torch.Tensor:
        r"""Membrane voltages in millivolts.

        Args:
            value (torch.Tensor): new membrane voltages.

        Returns:
            torch.Tensor: present membrane voltages.

        Raises:
            NotImplementedError: ``voltage`` must be implemented by the subclass.
        """
        return self.neuron_.voltage

    @voltage.setter
    def voltage(self, value: torch.Tensor) -> None:
        self.neuron_.voltage = value

    @property
    def refrac(self) -> torch.Tensor:
        r"""Remaining refractory periods, in milliseconds.

        Args:
            value (torch.Tensor): new remaining refractory periods.

        Returns:
            torch.Tensor: present remaining refractory periods.
        """
        return self.neuron_.refrac

    @refrac.setter
    def refrac(self, value: torch.Tensor) -> torch.Tensor:
        self.neuron_.refrac = value

    @property
    def spike(self) -> torch.Tensor:
        r"""Action potentials last generated.

        Returns:
            torch.Tensor: if the corresponding neuron generated an action potential
                during the prior step.
        """
        return self.neuron_.spike

    @property
    def origrefrac(self) -> torch.Tensor:
        r"""Remaining refractory periods before override, in milliseconds.

        Returns:
            torch.Tensor: original remaining refractory periods.
        """
        return self.origrefrac_

    @property
    def origspike(self) -> torch.Tensor:
        r"""Remaining refractory periods before override, in milliseconds.

        Returns:
            torch.Tensor: original remaining refractory periods.
        """
        return self.origspike_

    def clear(self, **kwargs) -> None:
        r"""Resets neurons to their resting state."""
        self.neuron_.clear(**kwargs)
        self.origrefrac_ = self.neuron_.refrac.clone()
        self.origspike_ = self.neuron_.spike.clone()

    def forward(
        self,
        inputs: torch.Tensor,
        override: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Runs a simulation step of the neuronal dynamics.

        This calls the inner :py:class:`~inferno.neural.Neuron` with given ``inputs``,
        passing in keyword arguments. Tensors in ``overrides`` are then assigned to
        attributes of the inner neuron with corresponding names (this includes assignment
        via property setters). Then the inner neuron's :py:attr:`~inferno.neural.Neuron.spike`
        getter is called and its value returned.

        Args:
            inputs (torch.Tensor): input currents to the neurons.
            override (torch.Tensor | None, optional): tensor containing the override
                value for the inner neuron. Defaults to None.

        Returns:
            torch.Tensor: postsynaptic spikes from integration of inputs.
        """
        # apply original refractory period
        self.neuron_.refrac = self.origrefrac_

        # apply inputs and preserve original state
        self.origspike_ = self.neuron_(inputs, **kwargs)
        self.origrefrac_ = self.neuron_.refrac.clone()

        # apply override value
        if override is not None:
            if self.use_spike_override:
                setattr(
                    self.neuron_,
                    "refrac",
                    torch.where(
                        override,
                        self.neuron_.absrefrac,
                        torch.where(self.neuron_.spike, 0.0, self.neuron_.refrac),
                    ),
                )
            else:
                setattr(self.neuron_, "refrac", override)

        # return (optionally overridden) spikes
        return self.neuron_.spike
