from .utils import pairwise_dist as pdist
import numpy as np

def measure(orig, emb, distance_matrices=None):
	"""
	Compute Sammon's stress of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
        tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
	OUTPUT:
		dict: stress
	"""
	if distance_matrices is None:
		orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
		emb_distance_matrix = pdist.pairwise_distance_matrix(emb)

	else:
		orig_distance_matrix, emb_distance_matrix = distance_matrices

	square_stress = np.square(orig_distance_matrix - emb_distance_matrix) / emb_distance_matrix
	square_stress[np.isnan(square_stress)] = 0
	sammons_stress = square_stress.sum() / emb_distance_matrix.sum()

	return {
		"sammons_stress": sammons_stress
	}