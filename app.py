import os
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_predict_onnx import onnx_predictor
from user_weights import (get_all_users, get_user_metadata,
                                       get_user_weights,
                                       track_question_tags,
                                       update_user_metadata,
                                       update_user_weights,
                                       update_weights_from_feedback,
                                       update_weights_from_query)
from get_destinations import (get_destinations_list,get_question_vector)
from get_default_weight import feature_names, weights_bias_vector
from database import db

# Define request models
class WeightUpdateRequest(BaseModel):
    tag_indices: List[int]
    new_weights: List[float]
    metadata: Dict[str, Any] = {}

class FeedbackRequest(BaseModel):
    destination_id: int
    tag_id: int
    rating: int  # 1-5 stars

router = APIRouter(prefix="/model", tags=["Model"])

@router.get("/get_question_tags/{question}")
async def get_question_tags(question: str):
    # Get the prediction
    original_sentence, predicted_tags = onnx_predictor.predict(question)

    # Print the sentence and its predicted tags
    print("Sentence:", original_sentence)
    print("Predicted Tags:", predicted_tags)
    return {"question_tags": predicted_tags}

@router.get("/get_destinations_list/{question_tags}/{top_k}")
async def get_destinations_list_api(question_tags: str, top_k:str):
    # Get the prediction
    question_vector = get_question_vector(question_tags)
    destinations_list = get_destinations_list(question_vector, int(top_k))

    print("destinations_list:", destinations_list)
    return {"destinations_list": destinations_list}

@router.get("/get_destinations_list_by_question/{question}/{top_k}")
async def get_destinations_list_api(question: str, top_k: str):
    # Get the prediction
    original_sentence, question_tags = onnx_predictor.predict(question)

    # Print the sentence and its predicted tags
    print("Sentence:", original_sentence)
    print("Predicted Tags:", question_tags)
    # Get the prediction
    question_tags = " ".join(question_tags)
    question_vector = get_question_vector(question_tags)
    destinations_list = get_destinations_list(question_vector, int(top_k))

    print("destinations_list:", destinations_list)
    return {"destinations_list": destinations_list}

@router.get("/get_destinations_list_by_question/{question}/{top_k}/{user_id}")
async def get_destinations_list_with_user_api(question: str, top_k: str, user_id: str):
    """
    Get a list of destinations based on a question and user-specific weights.
    
    Parameters:
    question (str): The question to get destinations for.
    top_k (str): The number of destinations to return.
    user_id (str): The ID of the user.
    
    Returns:
    dict: A dictionary containing the list of destinations.
    """
    # Get the prediction
    original_sentence, question_tags = onnx_predictor.predict(question)

    # Print the sentence and its predicted tags
    print("Sentence:", original_sentence)
    print("Predicted Tags:", question_tags)
    
    # Track the question tags for the user
    track_question_tags(user_id, question_tags)
    
    # Update weights based on query tags
    update_weights_from_query(user_id, question_tags, feature_names, weights_bias_vector)
    
    # Get the prediction
    question_tags_str = " ".join(question_tags)
    question_vector = get_question_vector(question_tags_str)
    destinations_list = get_destinations_list(question_vector, int(top_k), user_id)

    print("destinations_list:", destinations_list)
    return {"destinations_list": destinations_list}

@router.get("/users")
async def get_users():
    """
    Get a list of all users.
    
    Returns:
    dict: A dictionary containing the list of users.
    """
    users = get_all_users()
    return {"users": users}

@router.get("/users/{user_id}")
async def get_user(user_id: str):
    """
    Get the metadata for a user.
    
    Parameters:
    user_id (str): The ID of the user.
    
    Returns:
    dict: A dictionary containing the user's metadata.
    """
    metadata = get_user_metadata()
    if user_id not in metadata:
        return {"metadata": {}}
    return {"metadata": metadata[user_id]}

@router.get("/users/{user_id}/weights")
async def get_user_weights_api(user_id: str):
    """
    Get the weights for a user.

    Parameters:
    user_id (str): The ID of the user.

    Returns:
    dict: A dictionary containing the user's weights.
    """
    weights = get_user_weights(user_id, weights_bias_vector)
    # Convert numpy array to list for JSON serialization
    weights_list = weights.tolist() if weights is not None else None
    return {"user_id": user_id, "weights": weights_list}

@router.post("/users/{user_id}/weights")
async def update_user_weights_api(user_id: str, request: WeightUpdateRequest):
    """
    Update the weights for a user.
    
    Parameters:
    user_id (str): The ID of the user.
    request (WeightUpdateRequest): The request containing the tag indices, new weights, and metadata.
    
    Returns:
    dict: A dictionary indicating whether the update was successful.
    """
    # Validate the request
    if len(request.tag_indices) != len(request.new_weights):
        raise HTTPException(status_code=400, detail="Tag indices and new weights must have the same length")
    
    # Update the weights
    success = update_user_weights(user_id, request.tag_indices, request.new_weights, weights_bias_vector)
    
    # Update the metadata
    if success and request.metadata:
        update_user_metadata(user_id, request.metadata)
    
    return {"success": success}

@router.post("/users/{user_id}/feedback")
async def record_user_feedback(user_id: str, request: FeedbackRequest):
    """
    Record user feedback on a specific tag for a specific destination.
    
    Parameters:
    user_id (str): The ID of the user.
    request (FeedbackRequest): The request containing the destination ID, tag ID, and rating.
    
    Returns:
    dict: A dictionary indicating whether the feedback was recorded successfully.
    """
    # Validate the request
    if request.rating < 1 or request.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    # Update weights based on feedback
    success = update_weights_from_feedback(
        user_id, 
        request.destination_id, 
        request.tag_id, 
        request.rating, 
        weights_bias_vector
    )
    
    return {"success": success}

@router.get("/tags")
async def get_tags():
    """
    Get a list of all tags.
    
    Returns:
    dict: A dictionary containing the list of tags.
    """
    return {"tags": feature_names.tolist()}

app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['*',]
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """
    Connect to the database when the API starts.
    """
    db.connect()

@app.on_event("shutdown")
async def shutdown_event():
    """
    Close the database connection when the API shuts down.
    """
    db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7880)))
