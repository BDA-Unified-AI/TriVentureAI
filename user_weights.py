import datetime
import json

import numpy as np
from database import db


def get_user_weights(user_id, default_weights):
    """
    Get the weights for the given user. If the user doesn't have weights,
    return the default weights.
    
    Parameters:
    user_id (str): The ID of the user.
    default_weights (numpy.ndarray): The default weights to use if the user doesn't have weights.
    
    Returns:
    numpy.ndarray: The weights for the user.
    """
    weights = db.get_user_weights(user_id, default_weights)
    if weights is not None:
        return weights.copy()
    else:
        return default_weights.copy()

def user_weights_exist(user_id):
    """
    Check if weights exist for the given user.
    
    Parameters:
    user_id (str): The ID of the user.
    
    Returns:
    bool: True if weights exist for the user, False otherwise.
    """
    return db.user_weights_exist(user_id)

def save_user_weights(user_id, weights):
    """
    Save the weights for the given user.
    
    Parameters:
    user_id (str): The ID of the user.
    weights (numpy.ndarray): The weights to save.
    
    Returns:
    bool: True if the weights were saved successfully, False otherwise.
    """
    return db.save_user_weights(user_id, weights)

def update_user_weights(user_id, tag_indices, new_weights, default_weights):
    """
    Update the weights for the given user at the specified tag indices.
    
    Parameters:
    user_id (str): The ID of the user.
    tag_indices (list): The indices of the tags to update.
    new_weights (list): The new weights for the tags.
    default_weights (numpy.ndarray): The default weights to use if the user doesn't have weights.
    
    Returns:
    bool: True if the weights were updated successfully, False otherwise.
    """
    weights = get_user_weights(user_id, default_weights)
    
    # Update the weights
    for i, tag_index in enumerate(tag_indices):
        for j in range(len(weights)):
            weights[j][tag_index] = new_weights[i]
    
    return save_user_weights(user_id, weights)

def update_weights_from_query(user_id, query_tags, feature_names, default_weights):
    """
    Update weights based on user query. For each tag in the query, if a destination
    has that tag with a weight > 0, increase the weight by 5.
    
    Parameters:
    user_id (str): The ID of the user.
    query_tags (list): The tags from the user's query.
    feature_names (numpy.ndarray): The names of all features (tags).
    default_weights (numpy.ndarray): The default weights to use if the user doesn't have weights.
    
    Returns:
    bool: True if the weights were updated successfully, False otherwise.
    """
    weights = get_user_weights(user_id, default_weights)
    
    # Find indices of query tags
    tag_indices = []
    for tag in query_tags:
        # Find the index of the tag in feature_names
        matches = np.where(feature_names == tag.lower())[0]
        if len(matches) > 0:
            tag_indices.append(matches[0])
    
    # Update weights for destinations that have these tags
    for tag_index in tag_indices:
        for dest_index in range(len(weights)):
            # If the destination has this tag with weight > 0, increase by 5
            if weights[dest_index][tag_index] > 0:
                weights[dest_index][tag_index] += 5
    
    return save_user_weights(user_id, weights)

def update_weights_from_feedback(user_id, destination_id, tag_id, rating, default_weights):
    """
    Update weights based on user feedback (star rating).
    
    Parameters:
    user_id (str): The ID of the user.
    destination_id (int): The ID of the destination.
    tag_id (int): The ID of the tag.
    rating (int): The star rating (1-5).
    default_weights (numpy.ndarray): The default weights to use if the user doesn't have weights.
    
    Returns:
    bool: True if the weights were updated successfully, False otherwise.
    """
    weights = get_user_weights(user_id, default_weights)
    
    # Adjust weight based on rating
    if rating == 5:
        weights[destination_id][tag_id] += 5
    elif rating == 4:
        weights[destination_id][tag_id] += 3
    elif rating == 3:
        weights[destination_id][tag_id] += 1
    elif rating == 2:
        weights[destination_id][tag_id] -= 3
    elif rating == 1:
        weights[destination_id][tag_id] -= 5
    
    return save_user_weights(user_id, weights)

def get_user_metadata():
    """
    Get the metadata for all users.
    
    Returns:
    dict: The metadata for all users.
    """
    return db.get_all_user_metadata()

def update_user_metadata(user_id, metadata):
    """
    Update the metadata for the given user.
    
    Parameters:
    user_id (str): The ID of the user.
    metadata (dict): The metadata for the user.
    
    Returns:
    bool: True if the metadata was updated successfully, False otherwise.
    """
    # Get existing metadata
    existing_metadata = db.get_user_metadata(user_id)
    
    # Update with new metadata
    existing_metadata.update(metadata)
    
    # Save updated metadata
    return db.save_user_metadata(user_id, existing_metadata)

def track_question_tags(user_id, question_tags):
    """
    Track the tags from a user's question, keeping the last 5 questions.
    
    Parameters:
    user_id (str): The ID of the user.
    question_tags (list): The tags from the user's question.
    
    Returns:
    bool: True if the tags were tracked successfully, False otherwise.
    """
    # Get existing metadata
    metadata = db.get_user_metadata(user_id)
    
    # Initialize recent_tags if it doesn't exist
    if "recent_tags" not in metadata:
        metadata["recent_tags"] = []
    
    # Add new tags to the beginning of the list
    metadata["recent_tags"].insert(0, {
        "timestamp": str(datetime.datetime.now()),
        "tags": question_tags
    })
    
    # Keep only the last 5 entries
    metadata["recent_tags"] = metadata["recent_tags"][:5]
    
    # Save updated metadata
    return db.save_user_metadata(user_id, metadata)

def get_all_users():
    """
    Get a list of all users.
    
    Returns:
    list: A list of all user IDs.
    """
    return db.get_all_users()
