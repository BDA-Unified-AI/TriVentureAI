import numpy as np

from config import vectorizer
from get_default_weight import destinations, weights_bias_vector
from user_weights import get_user_weights


def get_des_accumulation(question_vector, weights_bias_vector):
    accumulation = 0
    for index in range(len(weights_bias_vector)):
        if question_vector[index]==1:
            accumulation += weights_bias_vector[index]
            
    return accumulation

def get_destinations_list(question_vector, top_k, user_id=None):
    des = destinations
    des = des[1:].reset_index(drop=True)
    """
    This function calculates the accumulated scores for each destination based on the given question vector and weights vector.
    It then selects the top 5 destinations with the highest scores and returns their names.
    Parameters:
    question_vector (numpy.ndarray): A 1D numpy array representing the question vector. Each element corresponds to a tag, and its value is 1 if the tag is present in the question, and 0 otherwise.
    weights_bias_vector (numpy.ndarray): A 2D numpy array representing the weights vector. Each row corresponds to a destination, and each column corresponds to a tag. The value at each position represents the weight of the tag for that destination.
    Returns:
    destinations_list: A list of strings representing the names of the top k destinations with the highest scores.
    """
    # Use user-specific weights if available, otherwise use default weights
    weights_vector = weights_bias_vector
    if user_id is not None:
        weights_vector = get_user_weights(user_id, weights_bias_vector)
    
    accumulation_dict = {}
    for index in range(len(weights_vector)):
        accumulation = get_des_accumulation(question_vector[0], weights_vector[index])
        accumulation_dict[str(index)] = accumulation
    
    top_keys = sorted(accumulation_dict, key=accumulation_dict.get, reverse=True)
    print(f"Top keys: {top_keys}")
    scores = [accumulation_dict[key] for key in top_keys]
    q1_score = np.percentile(scores, 25)
    destinations_list = []
    for key in top_keys:
        if accumulation_dict[key] > q1_score:
            destinations_list.append(des["name"][int(key)])
            print(f"{des['name'][int(key)]}: {accumulation_dict[key]}")
    
    return destinations_list[:top_k]

def get_question_vector(question_tags):
    """
    Generate a question vector based on the given list of question tags.

    Parameters:
    question_tags (list): A list of strings representing the tags associated with the question.
        Each tag is a word or phrase that describes a characteristic of a destination.

    Returns:
    numpy.ndarray: A 2D numpy array representing the question vector.
        The array is transformed from the input list of question tags using a vectorizer.
        Each row in the array corresponds to a tag, and its value is either 0 or 1.
        The length of each row is equal to the number of unique tags in the dataset.
    """
    question_tags = [question_tags]
    question_vector = vectorizer.transform(question_tags).toarray()
    return question_vector
