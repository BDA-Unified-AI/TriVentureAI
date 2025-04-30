import numpy as np

from config import destinations, feature_names, sorted_tags_dict, tags_vector


def create_bias_weights():
    """
    Create a weights vector for bias based on the given tags and weights.
    The function initializes a weights vector to zero, then maps the weights from the weights_tags_vector to the appropriate positions in the weights_vector based on the tags present in the destinations.
    """
    weights_tags_vector = [
        [15, 15, 0.9, 15, 15, 10, 1, 5, 0.6, 0.9, 0.9, 0.8, 10, 10, 1, 15],
        [15, 15, 0.9, 15, 15, 10, 15, 1, 10, 0.6, 0.9, 0.9, 0.8, 10, 10, 15, 0.8, 15],
        [15, 0.9, 0.8, 15, 15, 1, 10, 10, 0.6, 0.9, 0.9, 0.8, 5, 5, 1, 15],
        [15, 15, 0.9, 15, 0.7, 15, 15, 15, 1, 10, 10, 1, 0.9, 0.9, 0.9, 5, 5, 15, 0.8, 15],
        [10, 10, 15, 15, 0.8, 0.9, 15, 15, 15, 1, 10, 10, 0.6, 0.5, 0.9, 0.9, 0.8, 0.7, 15, 15, 15, 15, 15],
        [0.8, 0.9, 15, 0.8, 15, 0.9, 10, 15, 0.9, 0.9, 0.9, 0.8, 15, 10, 1, 15],
        [0.9, 0.8, 5, 1, 0.9, 10, 15, 0.9, 0.9, 0.9, 0.9, 0.8, 15, 1, 1, 15],
        [0.8, 0.9, 5, 1, 15, 15, 0.9, 0.9, 0.9, 0.8, 15, 1, 15],
        [0.8, 0.7, 15, 15, 1, 10, 0.7, 0.7, 0.6, 5, 5, 15],
        [0.8, 5, 1, 15, 15, 15, 0.7, 0.7, 15],
        [0.8, 0.7, 1, 15, 15, 0.7, 0.7, 15],
        [0.8, 0.7, 1, 15, 15, 15, 0.7, 0.9, 15],
        [0.8, 0.7, 1, 15, 15, 0.7, 0.7, 15],
        [0.8, 0.7, 1, 15, 15, 15, 0.7, 0.7, 15],
        [0.8, 0.7, 1, 15, 15, 15, 1, 10, 15],
        [10, 0.9, 0.8, 1, 15, 15, 15, 0.8, 10, 15],
        [0.8, 15, 1, 15, 15, 0.8, 10, 15],
        [10, 0.8, 1, 15, 1, 0.9, 0.8, 5, 0.8],
        [0.8, 15, 1, 5, 0.9, 0.8, 0.7, 0.7],
        [0.9, 0.8, 15, 1, 15, 0.7, 0.8, 0.7, 0.7, 5, 5, 15],
        [0.8, 0.7, 1, 5, 0.9, 10, 10, 15],
        [0.8, 1, 15, 15, 1, 0.9, 0.8, 0.8, 15],
        [0.8, 1, 10, 5, 5, 15],
        [0.8, 0.7, 1, 15, 15, 0.8, 0.9, 15],
        [10, 10, 10, 1, 10, 0.8, 1, 5, 10, 10, 10, 10, 1, 0.9, 1, 1, 15],
        [0.8, 0.7, 1, 15, 15, 0.8, 0.9, 15],
        [0.8, 0.7, 1, 10, 10, 0.8, 0.9, 15],
        [10, 0.8, 0.7, 15, 15, 1, 15, 15, 0.7, 0.7, 0.6, 5, 5, 1, 15],
        [5, 0.8, 0.7, 5, 5, 1, 10, 10, 0.7, 0.7, 0.6, 5, 5, 1, 15],
        [0.8, 0.7, 15, 5, 1, 10, 10, 10, 0.8, 0.7, 0.7, 5, 5, 5, 10, 15],
        [5, 5, 10, 15, 15, 15, 15, 0.9, 0.8, 0.7, 0.7, 1, 15],
        [10, 10, 15, 15, 10, 5, 1, 15, 15, 15, 15, 0.7, 5, 5, 0.8, 1, 15],
        [10, 15, 15, 15, 10, 10, 1, 1, 1, 15, 15, 5, 5],
        [0.8, 0.7, 0.6, 0.8, 1, 1, 1, 0.9, 0.8, 0.7, 0.7, 0.6, 5, 5, 1, 15],
        [1, 0.8, 0.9, 0.7, 0.6, 1, 0.9, 0.8, 1, 1, 0.9, 0.8, 0.8, 0.7, 0.9, 5, 5, 15],
        [1, 0.8, 0.9, 0.7, 0.6, 1, 0.9, 0.8, 1, 1, 0.9, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7, 5, 5, 1, 0.7, 0.6, 15],
        [0.9, 0.7, 1, 1, 0.8, 0.7, 0.8, 0.8, 0.7, 1, 1, 1, 1, 15]
    ]
    #Create a weights vector initialized to zero
    weights_vector = np.zeros(tags_vector.shape)

    # Map weights to the appropriate positions in the weights_vector
    for i, row in enumerate(destinations["tags"][1:].values):
        tags = row.split()
        for tag, weight in zip(tags, weights_tags_vector[i]):
            index = np.where(feature_names == tag.lower())[0][0]
            weights_vector[i][index] = weight
    np.save("Datasets/Weights/weights_bias.npy", weights_vector)

def create_freq_weights():
    """
    This function creates a weights vector for frequency-based weights based on the given tags and their frequencies.
    The function initializes a weights vector to zero, then maps the weights from the sorted_tags_dict to the appropriate positions in the weights_vector based on the tags present in the destinations.
    The weights are calculated as the ratio of the tag's frequency to the maximum frequency among all tags.

    Parameters:
    tags_vector (numpy.ndarray): A 2D numpy array representing the tags vector. Each row corresponds to a destination, and each column corresponds to a tag. The value at each position is 1 if the tag is present in the destination, and 0 otherwise.
    sorted_tags_dict (dict): A dictionary where the keys are the tags and the values are their frequencies.
    feature_names (numpy.ndarray): A 1D numpy array representing the names of the features (tags).
    destinations (pandas.DataFrame): A pandas DataFrame containing the destinations data, including the tags column.

    Returns:
    numpy.ndarray: A 2D numpy array representing the weights vector for frequency-based weights. Each row corresponds to a destination, and each column corresponds to a tag. The value at each position represents the weight of the tag for that destination.
    """
    #Create a weights vector initialized to zero
    weights_vector = np.zeros(tags_vector.shape)
    max_freq = max(sorted_tags_dict.values())
    
    # Map weights to the appropriate positions in the weights_vector
    for i, row in enumerate(destinations['tags'][1:].values):
        tags = row.split()
        for tag in tags:
            index = np.where(feature_names == tag.lower())[0][0]
            weights_vector[i][index] = f"{(sorted_tags_dict[tag.replace('_', ' ')]/max_freq):.2f}"
    np.save("Datasets/Weights/weights_freq.npy", weights_vector)

create_bias_weights()
create_freq_weights()

weights_bias_vector = np.load("Datasets/Weights/weights_bias.npy")
weights_freq = np.load("Datasets/Weights/weights_freq.npy")
weighted_tags_vector = weights_bias_vector
