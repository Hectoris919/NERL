from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium
import numpy as np
import multiprocessing as mp

from utils.save_load import save_connectome, load_connectome
from models.neuron import MultiCompartmentNeuron
from environments.connectome_env import ConnectomeEnv

class TimestepsTrackerCallback(BaseCallback):
	"""
	Callback to track remaining training timesteps and handle environment resets.
	"""
	def __init__(self, total_timesteps):
		super().__init__()
		self.total_timesteps = total_timesteps
		self.timesteps_used = 0

	def _on_step(self) -> bool:
		self.timesteps_used += 1
		print(self.timesteps_used)
		return True  # Continue training

	def get_remaining_timesteps(self):
		return max(0, self.total_timesteps - self.timesteps_used)

TOTAL_TIMESTEPS = 1000

def main():
	# Initialize MuJoCo body environment
	body_env = gymnasium.make("Ant-v5", render_mode="human")  # Replace with your MuJoCo-based body environment

	# Define the number of neurons
	NUM_INPUT_NEURONS = body_env.observation_space.shape[0]
	NUM_HIDDEN_NEURONS = 1000
	NUM_OUTPUT_NEURONS = body_env.action_space.shape[0]

	# Create input neurons (fixed positions)
	input_neurons = [
		MultiCompartmentNeuron(pos=np.random.rand(3), neuron_type="input", neuron_subclass="sensory")
		for _ in range(NUM_INPUT_NEURONS)
	]

	# Generate a new connectome
	PYRAMIDAL_RATIO = 0.7
	INTERNEURON_RATIO = 0.2
	MOTOR_RATIO = 0.15
	N_CPG_NEURONS = 100

	hidden_neurons = []
	for _ in range(NUM_HIDDEN_NEURONS):
		rand_val = np.random.rand()
		if rand_val < PYRAMIDAL_RATIO:
			neuron_subclass = "pyramidal"
		elif rand_val < PYRAMIDAL_RATIO + INTERNEURON_RATIO:
			neuron_subclass = "interneuron"
		elif rand_val < PYRAMIDAL_RATIO + INTERNEURON_RATIO + MOTOR_RATIO:
			neuron_subclass = "motor"
		else:
			neuron_subclass = "cpg"

		hidden_neurons.append(
			MultiCompartmentNeuron(pos=np.random.rand(3), neuron_type="hidden", neuron_subclass=neuron_subclass)
		)

	for _ in range(N_CPG_NEURONS):
		hidden_neurons.append(
			MultiCompartmentNeuron(pos=np.random.rand(3), neuron_type="hidden", neuron_subclass="cpg", freq=1.0, amp=1.0, phase=0.0)
		)

	# Create output neurons (fixed positions)
	output_neurons = [
		MultiCompartmentNeuron(pos=np.random.rand(3), neuron_type="output", neuron_subclass="motor")
		for _ in range(NUM_OUTPUT_NEURONS)
	]

	connectome = input_neurons + hidden_neurons + output_neurons

	# try:
	# 	connectome = load_connectome("trained_connectome.pkl")
	# 	print("Loaded existing connectome.")
	# except FileNotFoundError:
	# 	connectome = input_neurons + hidden_neurons + output_neurons
	# 	print("Created new connectome.")

	N_INITIAL_NEURONS = len(connectome)

	EVAL_STEPS = 500

	callback = TimestepsTrackerCallback(TOTAL_TIMESTEPS)
	env = ConnectomeEnv(connectome, body_env, n_initial_neurons=N_INITIAL_NEURONS, evaluation_steps=EVAL_STEPS)
	ppo_agent = PPO("MlpPolicy", env, n_steps=100, verbose=1)
	env.reset()
	env.render()

	while callback.get_remaining_timesteps() > 0:
		try:
			timesteps_left = callback.get_remaining_timesteps()

			ppo_agent.learn(total_timesteps=timesteps_left, callback=callback)
			connectome = env.connectome

			break  # Training completed successfully
		except RuntimeError as e:
			if "Observation space changed" in str(e):
				connectome = env.connectome

				env = ConnectomeEnv(connectome, body_env, n_initial_neurons=N_INITIAL_NEURONS, evaluation_steps=EVAL_STEPS)
				ppo_agent = PPO("MlpPolicy", env, n_steps=100, verbose=1)
				print("Number of timesteps left: ", callback.get_remaining_timesteps())
				# save_connectome(connectome, "trained_connectome.pkl")
			else:
				raise e

	# Save trained connectome
	# save_connectome(connectome, "trained_connectome.pkl")

	env.close()

if __name__ == "__main__":
	mp.freeze_support()  # Required for Windows multiprocessing
	main()