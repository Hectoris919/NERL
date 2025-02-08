import gymnasium
import numpy as np
import open3d as o3d
from gymnasium import spaces
from models.neuron import MultiCompartmentNeuron
import random
import torch

NEURON_TYPES = ["pyramidal", "interneuron", "motor", "cpg"]

class ConnectomeEnv(gymnasium.Env):
	def __init__(self, connectome, body_env, n_initial_neurons, evaluation_steps=100, evolve_connectome=True):
		super(ConnectomeEnv, self).__init__()
		self.connectome = connectome
		self.body_env = body_env
		self.evaluation_steps = evaluation_steps
		self.evolve_connectome = evolve_connectome
		self.initial_neuron_count = n_initial_neurons

		self.observation_space = spaces.Box(low=-100.0, high=50.0, shape=(len(self.connectome),), dtype=np.float32)

		self.action_space = spaces.Box(
			low=np.array([-1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1]),
			high=np.array([1, len(self.connectome), len(self.connectome), 1, 1, 1, 1, 1, 1, 1, 1, 1]),
			dtype=np.float32
		)

		self._connect_neurons()

	def render(self, mode="human"):
		# Collect neuron positions and firing states
		neurons = [(tuple(neuron.position), neuron.is_firing()) for neuron in self.connectome]

		# Collect connections (edges)
		connections = []
		neuron_index_map = {id(neuron): i for i, neuron in enumerate(self.connectome)}

		for neuron in self.connectome:
			for target in neuron.connections:
				if id(target) in neuron_index_map:
					connections.append([neuron_index_map[id(neuron)], neuron_index_map[id(target)]])

		print(f"📤 Sending update: {len(neurons)} neurons, {len(connections)} connections", flush=True)

		# Render the body environment
		if hasattr(self.body_env, "render"):
			self.body_env.render()


	def reset(self, seed=None, options=None):
		if seed is not None:
			np.random.seed(seed)

		for neuron in self.connectome:
			neuron.V_m = torch.zeros(neuron.compartments)

		self.body_env.reset()
		self.observation_space = spaces.Box(low=-100.0, high=50.0, shape=(len(self.connectome),), dtype=np.float32)

		obs = self._get_observation()

		if obs.shape[0] != self.observation_space.shape[0]:
			raise RuntimeError("Observation space changed. Restart training with a new environment.")

		return obs, {}

	def step(self, action):
		prev_obs_shape = self.observation_space.shape[0]

		if self.evolve_connectome:
			self._modify_connectome(action)

		empowerment_reward, done, truncated, info = self._evaluate_body_control()

		print("empowerment_reward: ", empowerment_reward)

		# Process neuron activity
		for neuron in self.connectome:
			neuron.update()

		self.render()

		if len(self.connectome) != prev_obs_shape:
			print(f"Observation space changed from {prev_obs_shape} to {len(self.connectome)}, forcing SB3 reset...", flush=True)
			raise RuntimeError("Observation space changed. Restart training with a new environment.")

		return self._get_observation(), empowerment_reward, done, truncated, info

	# ============================================================================ #

	def _get_observation(self):
		return np.array([neuron.V_m.mean().item() for neuron in self.connectome], dtype=np.float32)

	def _update_input_neurons(self, body_obs):
		"""
		Updates input neurons with the latest observations from the environment.
		"""
		input_neurons = [neuron for neuron in self.connectome if neuron.neuron_type == "input"]

		if len(input_neurons) != len(body_obs):
			print(f"⚠️ Mismatch detected! Adjusting {len(input_neurons)} input neurons to match {len(body_obs)} observation values.", flush=True)

			# Adjust input neuron count to match body observations
			if len(input_neurons) < len(body_obs):
				extra_neurons = [MultiCompartmentNeuron(pos=np.random.rand(3), neuron_type="input") for _ in range(len(body_obs) - len(input_neurons))]
				self.connectome.extend(extra_neurons)
			elif len(input_neurons) > len(body_obs):
				self.connectome = [neuron for neuron in self.connectome if neuron.neuron_type != "input"][:len(body_obs)]

			input_neurons = [neuron for neuron in self.connectome if neuron.neuron_type == "input"]

		V_MIN, V_MAX = -70.0, 30.0  # From resting potential to firing threshold

		for i, neuron in enumerate(input_neurons):
			scaled_voltage = V_MIN + (body_obs[i] + 1) * (V_MAX - V_MIN) / 2  # Scale from [-1,1] to [V_MIN,V_MAX]
			neuron.V_m = torch.tensor([scaled_voltage], dtype=torch.float32)  # Assign observation as membrane potential
			# neuron.V_m = torch.tensor([body_obs[i]], dtype=torch.float32)  # Assign observation as membrane potential

	def _connect_neurons(self):
		"""Randomly connect neurons while ensuring valid compartment indexing."""
		connect_prob = 0.1  # Adjust connection probability as needed

		for neuron in self.connectome:
			possible_targets = [n for n in self.connectome if n is not neuron]  # Avoid self-connections
			num_connections = max(1, int(len(possible_targets) * connect_prob))  # Ensure at least one connection

			for target in random.sample(possible_targets, num_connections):
				valid_compartment = min(neuron.compartments - 1, target.compartments - 1)  # Find the smallest valid compartment index
				if valid_compartment >= 0:  # Ensure the connection is valid
					neuron.connect(target, compartment=valid_compartment)

	def _modify_connectome(self, action):
		"""
		Dynamically modifies the connectome based on RL agent decisions.
		The action consists of:
		- action[0]: Neuron index to remove (-1 for no removal)
		- action[1]: Source neuron index for connection (-1 for no connection)
		- action[2]: Target neuron index for connection (-1 for no connection)
		- action[3]: Neuron index to move (-1 for no movement)
		- action[4]: New x-coordinate (only used if moving)
		- action[5]: New y-coordinate (only used if moving)
		- action[6]: New z-coordinate (only used if moving)

		- action[7]: Add a new neuron? (0 for not adding a neuron, 1 for adding a neuron)
		- action[8]: New neuron subtype (0 for pyramidal, 1 for interneuron, 2 for motor, 3 for cpg)
		- action[9]: CPG ONLY: Oscillation frequency
		- action[10]: CPG ONLY: Oscillation amplitude
		- action[11]: CPG ONLY: Oscillation phase
		"""
		neuron_remove_idx, src_idx, tgt_idx, move_idx, new_x, new_y, new_z, add_neuron, neuron_type, cpg_freq, cpg_amp, cpg_phase = map(int, action)

		SCALE = 1

		x = np.tanh(new_x) * SCALE
		y = np.tanh(new_y) * SCALE
		z = np.tanh(new_z) * SCALE

		neuron_type = int((neuron_type + 1) * 1.5)

		# 🚀 1. Remove a specific neuron
		if 0 <= np.tanh(neuron_remove_idx) < len(self.connectome):
			if self.connectome[neuron_remove_idx].neuron_type not in "input output":
				removed_neuron = self.connectome.pop(neuron_remove_idx)
				print(f"❌ Removed neuron {neuron_remove_idx}")

		# 🚀 2. Connect two neurons if valid indices are given
		if 0 <= np.tanh(src_idx) < len(self.connectome) and 0 <= tgt_idx < len(self.connectome) and src_idx != tgt_idx:
			self.connectome[src_idx].connect(self.connectome[tgt_idx])
			print(f"🔗 Connected neuron {src_idx} → {tgt_idx}")

		# 🚀 3. Move a neuron to a new position
		if 0 <= np.tanh(move_idx) < len(self.connectome):
			neuron = self.connectome[move_idx]
			neuron.position = torch.tensor(np.array([x, y, z]), dtype=torch.float32)

			# Recalculate signal delays for affected connections
			for post_neuron in self.connectome:
				if neuron in post_neuron.synapses:
					distance = np.linalg.norm(neuron.position - post_neuron.position.numpy())
					delay_steps = int(distance / neuron.signal_speed)
					post_neuron.synapses[neuron] = (post_neuron.synapses[neuron][0], delay_steps)
			print(f"📍 Moved neuron {move_idx} to ({x}, {y}, {z})")

		if round(add_neuron) == 1:
			print(round(neuron_type))
			neuron_subclass = NEURON_TYPES[round(neuron_type)]  # Ensure valid neuron types
			if neuron_subclass != "cpg":
				new_neuron = MultiCompartmentNeuron(pos=np.random.rand(3) * SCALE, neuron_subclass=neuron_subclass)
			else:
				new_neuron = MultiCompartmentNeuron(pos=np.random.rand(3) * SCALE, neuron_subclass=neuron_subclass, freq=np.tanh(cpg_freq), amp=np.tanh(cpg_amp), phase=np.tanh(cpg_phase))
			self.connectome.append(new_neuron)
			print(f"➕ Added new {neuron_subclass} neuron at ({new_neuron.position[0]}, {new_neuron.position[1]}, {new_neuron.position[2]})")

		# 🚀 **Update observation space dynamically**
		new_obs_space_size = len(self.connectome)

		if new_obs_space_size != self.observation_space.shape[0]:
			print(f"🔄 Observation space changed from {self.observation_space.shape[0]} to {new_obs_space_size}, updating...", flush=True)
			self.observation_space = spaces.Box(low=-100.0, high=50.0, shape=(new_obs_space_size,), dtype=np.float32)

			for neuron in self.connectome:
				neuron.V_m = torch.zeros(neuron.compartments, dtype=torch.float32)  # Reset membrane potential
				neuron.I_syn = torch.zeros(neuron.compartments, dtype=torch.float32)  # Reset synaptic current

	def _generate_body_action(self):
		"""
		Converts connectome activity into body actions.
		Uses output neurons to determine control signals.
		"""
		output_neurons = [neuron for neuron in self.connectome if neuron.neuron_type == "output"]

		output_activity = []
		for neuron in output_neurons:
			integrated_input = sum(weight * pre_neuron.V_m.mean().item() for pre_neuron, (weight, _) in neuron.synapses.items())
			output_signal = np.tanh(integrated_input)  # Use tanh for now, but can be refined

			# Introduce temporal smoothing (low-pass filtering)
			neuron.last_activity = 0.8 * neuron.last_activity + 0.2 * output_signal  # Exponential Moving Average

			output_activity.append(neuron.last_activity)

		return np.array(output_activity)

	def _evaluate_body_control(self):
		"""
		Evaluates how well the connectome controls the body, incorporating biological realism.
		"""
		total_control = 0.0
		total_activity = 0.0  # Track overall neuron firing
		synaptic_change = 0.0  # Track network adaptation
		input_neurons = [neuron for neuron in self.connectome if neuron.neuron_type == "input"]
		output_neurons = [neuron for neuron in self.connectome if neuron.neuron_type == "output"]

		obs, _ = self.body_env.reset() # Reset body env without obs dependency

		for _ in range(self.evaluation_steps):
			# Feed observation into input neurons
			self._update_input_neurons(obs)
			action = self._generate_body_action()  # Convert connectome activity into body actions
			obs, reward, done, _, info = self.body_env.step(action)
			truncated = info.get("TimeLimit.truncated", done)

			total_control += reward  # Movement effectiveness
			total_activity += sum(abs(neuron.V_m.mean().item()) for neuron in self.connectome)
			synaptic_change += sum(abs(weight.item()) for neuron in self.connectome for weight, _ in neuron.synapses.values())

			if done:
				self.body_env.reset()

		# Control Score: Movement effectiveness (No neuron bias)
		control_score = total_control / self.evaluation_steps

		# Energy Efficiency: Fewer active neurons are rewarded
		avg_activity = total_activity / (len(self.connectome) * self.evaluation_steps)
		energy_efficiency_score = max(0, 1.0 - avg_activity)

		# Simplicity Reward: Encourage pruning of unnecessary neurons
		min_neurons = max(1, len(input_neurons) + len(output_neurons))  # Preserve input/output
		simplicity_score = max(0, 1.0 - ((len(self.connectome) - min_neurons) / self.initial_neuron_count))

		# Synaptic Adaptation: Encourage adaptive learning
		synaptic_adaptation_score = min(1.0, synaptic_change / (len(self.connectome) * self.evaluation_steps))

		empowerment_reward =  (control_score * 0.5) + (energy_efficiency_score * 0.3) + (synaptic_adaptation_score * 0.1) + (simplicity_score * 0.1)

		return empowerment_reward, done, truncated, info
