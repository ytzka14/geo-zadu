from snc.snc import SNC
import numpy as np
from .utils import knn



def measure(orig, emb, iteration=150, walk_num_ratio=0.3, alpha=0.1, k=25, clustering_strategy="dbscan", knn_info=None, return_local=False, geodesic=False):
	"""
	Compute the Steadiness and Cohesiveness of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		int: iteration: number of iterations for the SNC algorithm
		float: walk_num_ratio: ratio of the number of random walks to the number of points
		float: alpha: parameter for the SNC algorithm
		int: k: number of nearest neighbors to consider
		str: clustering_strategy: clustering strategy to use (dbscan or kmeans)
		tuple: knn_info: precomputed k-nearest neighbors of the original and embedded data (Optional)
	OUTPUT:
		dict: steadiness and cohesiveness score
	"""

	if knn_info is None:
		if geodesic:
			orig_knn_indices = knn.knn(orig, k, distance_function="geodesic")
		else:
			orig_knn_indices = knn.knn(orig, k)
		emb_knn_indices = knn.knn(emb, k)
	else:
		orig_knn_indices, emb_knn_indices = knn_info
	
	orig_snn_graph = knn.snn(orig, k, knn_indices=orig_knn_indices, directed=True)
	emb_snn_graph = knn.snn(emb, k, knn_indices=emb_knn_indices, directed=True)

	snn_knn_matrix = {
		"raw_knn": orig_knn_indices,
		"raw_snn": orig_snn_graph,
		"emb_knn": emb_knn_indices,
		"emb_snn": emb_snn_graph
	}


	snc_obj = SNC(
		orig, emb, 
		iteration=iteration, 
		walk_num_ratio=walk_num_ratio, 
		dist_strategy="snn", 
		dist_parameter={ "alpha": alpha }, 
		dist_function=None, 
		cluster_strategy=clustering_strategy, 
		# snn_knn_matrix=snn_knn_matrix
	)

	snc_obj.fit(record_vis_info=return_local)

	steadiness = snc_obj.steadiness()
	cohesiveness = snc_obj.cohesiveness()

	if return_local:
		_, _, _, points_info = snc_obj.vis_info()

		stead_local = np.array([1 - point_info["false_val"] for point_info in points_info])
		cohev_local = np.array([1 - point_info["missing_val"] for point_info in points_info])

	if return_local:
		return { ## TODO
			"steadiness": steadiness,
			"cohesiveness": cohesiveness
		}, {
			"local_steadiness": stead_local,
			"local_cohesiveness": cohev_local
		}
	else:
		return {
			"steadiness": steadiness,
			"cohesiveness": cohesiveness
		}
