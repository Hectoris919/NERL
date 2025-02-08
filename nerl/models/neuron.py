import torch
import numpy as np

class MultiCompartmentNeuron:
	def __init__(self, pos, neuron_type="hidden", neuron_subclass="pyramidal", freq=None, amp=None, phase=None):
		"""
		Multi-compartment neuron with different numbers of compartments for different neuron types.

		Args:
		- pos: 3D position of the neuron.
		- neuron_type: "input", "hidden", or "output".
		- neuron_subclass: "pyramidal", "interneuron", "motor", "sensory", "cpg".
		"""
		self.position = torch.tensor(pos, dtype=torch.float32)
		self.neuron_type = neuron_type
		self.neuron_subclass = neuron_subclass

		# Set different numbers of compartments for different neuron types
		if neuron_subclass == "pyramidal":
			self.compartments = 5  # More dendritic processing
			self.synaptic_strength = 1.0  # Stronger excitatory connections
			self.connect_range = 5  # Can connect far away
			self.signal_speed = 40
		elif neuron_subclass == "interneuron":
			self.compartments = 3  # Local inhibition
			self.synaptic_strength = -0.8  # Inhibitory
			self.connect_range = 2  # Short-range inhibition
			self.signal_speed = 5
		elif neuron_subclass == "motor":
			self.compartments = 2  # Direct output to muscles
			self.synaptic_strength = 0.5  # Moderate strength
			self.connect_range = 3  # Output-focused connections
			self.signal_speed = 90
		elif neuron_subclass == "sensory":
			self.compartments = 1  # Simple input processing
			self.synaptic_strength = 0.2  # Weak, mostly feedforward
			self.connect_range = 1  # Only connects to nearby neurons
			self.signal_speed = 50
		elif neuron_subclass == "cpg":
			self.compartments = 3  # Moderate complexity for oscillation
			self.synaptic_strength = 0.7  # Strong but balanced
			self.connect_range = 5  # Can connect to multiple neurons
			self.signal_speed = 10
			self.oscillation_frequency = freq
			self.oscillation_amplitude = amp
			self.oscillation_phase = phase
		else:
			raise ValueError(f"Invalid neuron subclass: {neuron_subclass}")

		# Membrane potentials
		self.V_m = torch.zeros(self.compartments, dtype=torch.float32)
		self.I_syn = torch.zeros(self.compartments, dtype=torch.float32)

		self.threshold = -55.0
		self.resting_potential = -70.0  # Default resting potential
		self.refractory_time = 2  # Number of steps before the neuron can fire again
		self.refractory_counter = 0  # Tracks remaining refractory period
		self.adaptive_threshold = self.threshold  # Start threshold
		self.threshold_adaptation = 2.0  # Amount to increase threshold on spike
		self.threshold_decay = 0.1  # How quickly the threshold returns to normal
		self.leak = 0.01  # Default leak value, simulating passive ion channel loss

		self.last_activity = 0.0

		# Synapses and weights
		self.synapses = {}
		self.connections = []
		self.signal_queue = []

	def connect(self, neuron, weight=None, compartment=None):
		"""
		Connect this neuron to another neuron with a synaptic weight.
		Weight is determined by the neuron's subclass.
		"""
		if self.neuron_type != "output":
			if weight is None:
				weight = self.synaptic_strength * np.random.uniform(0.5, 1.5)

			# If no specific compartment is given, pick a random valid one
			if compartment is None or compartment >= neuron.compartments:
				compartment = np.random.randint(0, neuron.compartments)

			self.synapses[neuron] = (torch.tensor(weight, dtype=torch.float32), compartment)

		self.connections.append(neuron)

	def update(self):
		""" Update membrane potentials based on synaptic inputs, with refractory period. """

		if self.refractory_counter > 0:
			self.refractory_counter -= 1  # Count down refractory period
			self.V_m.fill_(self.resting_potential)  # Maintain resting potential
			return  # Skip update during refractory period

		# If this neuron is a CPG neuron, apply its oscillatory activity
		if self.neuron_subclass == "cpg":
			dt = torch.sin(torch.tensor(self.oscillation_phase))  # Create oscillation
			self.V_m += self.oscillation_amplitude * dt  # Inject oscillatory current

			# Also propagate rhythmic signals to connected neurons
			for neuron, (weight, compartment) in self.synapses.items():
				neuron.I_syn[compartment] += self.oscillation_amplitude * dt * weight

			self.oscillation_phase += self.oscillation_frequency
			self.oscillation_phase %= (2 * torch.pi)
		else:
			for i in range(len(self.signal_queue) - 1, -1, -1):  # Iterate in reverse to remove processed items
				source_neuron, weight, compartment, delay = self.signal_queue[i]
				if delay == 0:
					self.I_syn[compartment] += weight * source_neuron.V_m[min(compartment, source_neuron.compartments - 1)]
					self.signal_queue.pop(i)  # Remove processed signal
				else:
					self.signal_queue[i] = (source_neuron, weight, compartment, delay - 1)  # Decrease delay

		# Adaptive synaptic strength scaling
		avg_Vm = torch.mean(self.V_m)
		if avg_Vm < self.resting_potential + 5:
			self.synaptic_strength *= 1.01  # Slightly increase sensitivity
		elif avg_Vm > self.threshold:
			self.synaptic_strength *= 0.99  # Slightly decrease sensitivity
		self.synaptic_strength = np.clip(self.synaptic_strength, -2.0, 2.0)

		# Check if neuron fires
		if torch.any(self.V_m > self.adaptive_threshold):
			self.fire()  # Emit spike
			self.adaptive_threshold += self.threshold_adaptation
			self.refractory_counter = self.refractory_time

			# Apply STDP to all synapses
			for neuron, (weight, compartment) in list(self.synapses.items()):
				self.stdp_update(neuron, weight, compartment)

		# Apply leak (decay over time)
		self.V_m -= self.leak * (self.V_m - self.resting_potential)

		# Decay adaptive threshold back to baseline
		self.adaptive_threshold = max(self.threshold, self.adaptive_threshold - self.threshold_decay)

		# Reset synaptic inputs for next timestep
		self.I_syn.zero_()

	def stdp_update(self, neuron, weight, compartment, dt=0.1):
		target_compartment = min(compartment, len(self.V_m) - 1)
		source_compartment = min(compartment, len(neuron.V_m) - 1)

		pre_voltage = neuron.V_m[source_compartment].item()
		post_voltage = self.V_m[target_compartment].item()

		# Hebbian Rule: If both pre and post are active together, strengthen connection
		if pre_voltage > -55 and post_voltage > -55:
			weight += 0.02 * np.exp(-dt / 10)  # Strengthen connection
		elif pre_voltage > -55 and post_voltage < -60:
			weight -= 0.01 * np.exp(-dt / 10)  # Weaken connection

		weight = torch.clamp(weight, -2.0, 2.0)
		self.synapses[neuron] = (weight, target_compartment)

	def fire(self):
		"""Handles neuron firing, applying STDP and notifying downstream neurons."""

		# Notify connected neurons
		for neuron, (weight, compartment) in list(self.synapses.items()):
			distance = np.linalg.norm(self.position - neuron.position)
			delay_steps = max(1, int(distance / self.signal_speed))  # Convert distance to time delay (in steps)
			neuron.signal_queue.append((self, weight, compartment, delay_steps))

			# Apply STDP weight update
			self.stdp_update(neuron, weight, compartment)

		# Implement lateral inhibition if this neuron is an interneuron
		if self.neuron_subclass == "interneuron":
			for neuron, (weight, compartment) in self.synapses.items():
				if weight < 0:
					neuron.V_m[compartment] -= abs(weight) * 1.5  # Strong inhibition

		# potential_targets = [n for n in self.connections if n not in self.synapses and np.linalg.norm(n.position - self.position.numpy()) < self.connect_range]
		potential_targets = [n for n in self.connections if n not in self.synapses and torch.norm(n.position - self.position) < self.connect_range]

		np.random.shuffle(potential_targets)
		max_new_connections = 2

		for target in potential_targets[:max_new_connections]:
			if np.random.rand() < 0.3:  # 30% chance to form a new connection
				synaptic_weight = torch.tensor(self.synaptic_strength * np.random.uniform(0.5, 1.5), dtype=torch.float32)
				compartment = np.random.randint(0, target.compartments)  # Choose random compartment
				self.synapses[target] = (synaptic_weight, compartment)

		# Reset membrane potential and enter refractory period
		self.V_m.fill_(self.resting_potential)
		self.refractory_counter = self.refractory_time

		# Increase threshold temporarily for adaptive response
		self.adaptive_threshold += self.threshold_adaptation

	def is_firing(self):
		"""
		Check if the neuron is firing based on its membrane potential.
		"""
		if isinstance(self.V_m, np.ndarray):
			return np.any(self.V_m > -55)
		else:
			return np.any(self.V_m.detach().cpu().numpy() > -55)  # Convert PyTorch tensor to NumPy

	def receive_spike(self, pre_neuron, compartment):
		"""Handles incoming spikes from presynaptic neurons."""
		target_compartment = min(compartment, len(self.V_m) - 1)

		if self.refractory_counter == 0:  # Only process if not in refractory period
			self.V_m[target_compartment] += pre_neuron.synapses[self][0]  # Apply synaptic weight