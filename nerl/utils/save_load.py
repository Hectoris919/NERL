import torch
import pickle

def save_connectome(connectome, filename="trained_connectome.pkl"):
	"""Saves the trained connectome to a file."""
	safe_connectome = []

	neuron_index_map = {neuron: i for i, neuron in enumerate(connectome)}  # Map neurons to indices

	for neuron in connectome:
		neuron_copy = neuron.__dict__.copy()

		# Store connections as (source_index, target_index, weight, compartment)
		neuron_copy["synapses"] = [
			(neuron_index_map[pre], weight, compartment)
			for pre, (weight, compartment) in neuron.synapses.items()
			if pre in neuron_index_map  # Ensure the neuron exists in the map
		]

		# Store connections as indices instead of objects
		neuron_copy["connections"] = [
			neuron_index_map[conn] for conn in neuron.connections if conn in neuron_index_map
		]

		# Convert tensors to NumPy arrays for serialization
		if "V_m" in neuron_copy:
			neuron_copy["V_m"] = neuron_copy["V_m"].detach().cpu().numpy()
		if "I_syn" in neuron_copy:
			neuron_copy["I_syn"] = neuron_copy["I_syn"].detach().cpu().numpy()

		safe_connectome.append(neuron_copy)

	torch.save(safe_connectome, filename)
	print(f"✅ Connectome saved to {filename} with connections serialized safely.")

def load_connectome(filename="trained_connectome.pkl"):
	"""Loads a trained connectome from a file."""
	with open(filename, "rb") as f:
		connectome = torch.load(f)
	for neuron in connectome:
		if not hasattr(neuron, "connections"):
			neuron.connections = []
	print(f"Connectome loaded from {filename}")
	return connectome