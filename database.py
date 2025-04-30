import json
import os
import datetime

import numpy as np
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv
load_dotenv()
class Database:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        """
        Connect to the MongoDB database.
        """
        try:
            # Connect to MongoDB
            self.client = MongoClient(os.environ.get("MONGODB_URL"))
            self.db = self.client[os.environ.get("DB_NAME", "triventure")]
            
            # Create users collection if it doesn't exist
            if "users" not in self.db.list_collection_names():
                self.db.create_collection("users")
                
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def close(self):
        """
        Close the database connection.
        """
        if self.client:
            self.client.close()
    
    def create_tables(self):
        """
        Create the necessary collections if they don't exist.
        
        Note: This method is kept for backward compatibility but is no longer used
        for creating user-specific collections. Instead, user data is stored in collections
        that are created on-demand when saving user data.
        """
        try:
            # Ensure users collection exists
            if "users" not in self.db.list_collection_names():
                self.db.create_collection("users")
            
            return True
        except Exception as e:
            print(f"Error creating collections: {e}")
            return False
    
    def create_user_table(self, user_id):
        """
        Create a collection for a specific user if it doesn't exist.
        
        Parameters:
        user_id (str): The ID of the user.
        
        Returns:
        bool: True if the collection was created successfully, False otherwise.
        """
        try:
            # User data is stored in the "user_data" collection
            if "user_data" not in self.db.list_collection_names():
                self.db.create_collection("user_data")
            
            # Add user to the users collection if not exists
            self.db.users.update_one(
                {"_id": user_id},
                {"$setOnInsert": {"created_at": datetime.datetime.now()}},
                upsert=True
            )
            
            return True
        except Exception as e:
            print(f"Error creating user collection: {e}")
            return False
    
    def save_user_weights(self, user_id, weights):
        """
        Save user weights to the database.
        
        Parameters:
        user_id (str): The ID of the user.
        weights (numpy.ndarray): The weights to save.
        
        Returns:
        bool: True if the weights were saved successfully, False otherwise.
        """
        try:
            # Create user entry if it doesn't exist
            if not self.create_user_table(user_id):
                return False
            
            # Convert numpy array to bytes
            weights_bytes = Binary(weights.tobytes())
            
            # Update or insert weights
            self.db.user_data.update_one(
                {"user_id": user_id, "data_type": "weights"},
                {"$set": {
                    "data": weights_bytes,
                    "metadata": {},
                    "updated_at": datetime.datetime.now()
                }},
                upsert=True
            )
            
            return True
        except Exception as e:
            print(f"Error saving user weights: {e}")
            return False
    
    def get_user_weights(self, user_id, default_weights):
        """
        Get user weights from the database.
        
        Parameters:
        user_id (str): The ID of the user.
        default_weights (numpy.ndarray): The default weights to use if the user doesn't have weights.
        
        Returns:
        numpy.ndarray: The weights for the user.
        """
        try:
            # Get weights from user_data collection
            result = self.db.user_data.find_one(
                {"user_id": user_id, "data_type": "weights"}
            )
            
            if result and "data" in result:
                # Convert bytes to numpy array
                weights_bytes = result["data"]
                weights = np.frombuffer(weights_bytes, dtype=default_weights.dtype)
                return weights.reshape(default_weights.shape)
            
            return default_weights
        except Exception as e:
            print(f"Error getting user weights: {e}")
            return default_weights
    
    def user_weights_exist(self, user_id):
        """
        Check if weights exist for the given user.
        
        Parameters:
        user_id (str): The ID of the user.
        
        Returns:
        bool: True if weights exist for the user, False otherwise.
        """
        try:
            # Check if weights exist in user_data collection
            result = self.db.user_data.find_one(
                {"user_id": user_id, "data_type": "weights"}
            )
            
            return result is not None
        except Exception as e:
            print(f"Error checking if user weights exist: {e}")
            return False
    
    def save_user_metadata(self, user_id, metadata):
        """
        Save user metadata to the database.
        
        Parameters:
        user_id (str): The ID of the user.
        metadata (dict): The metadata to save.
        
        Returns:
        bool: True if the metadata was saved successfully, False otherwise.
        """
        try:
            # Create user entry if it doesn't exist
            if not self.create_user_table(user_id):
                return False
            
            # Update or insert metadata
            self.db.user_data.update_one(
                {"user_id": user_id, "data_type": "metadata"},
                {"$set": {
                    "metadata": metadata,
                    "updated_at": datetime.datetime.now()
                }},
                upsert=True
            )
            
            return True
        except Exception as e:
            print(f"Error saving user metadata: {e}")
            return False
    
    def get_user_metadata(self, user_id):
        """
        Get user metadata from the database.
        
        Parameters:
        user_id (str): The ID of the user.
        
        Returns:
        dict: The metadata for the user.
        """
        try:
            # Get metadata from user_data collection
            result = self.db.user_data.find_one(
                {"user_id": user_id, "data_type": "metadata"}
            )
            
            if result and "metadata" in result:
                return result["metadata"]
            
            return {}
        except Exception as e:
            print(f"Error getting user metadata: {e}")
            return {}
    
    def get_all_user_metadata(self):
        """
        Get metadata for all users from the database.
        
        Returns:
        dict: A dictionary mapping user IDs to their metadata.
        """
        try:
            # Get metadata for all users
            results = self.db.user_data.find({"data_type": "metadata"})
            
            # Build a dictionary of user_id -> metadata
            metadata = {}
            for result in results:
                if "user_id" in result and "metadata" in result:
                    metadata[result["user_id"]] = result["metadata"]
            
            return metadata
        except Exception as e:
            print(f"Error getting all user metadata: {e}")
            return {}
    
    def get_all_users(self):
        """
        Get a list of all users from the database.
        
        Returns:
        list: A list of all user IDs.
        """
        try:
            # Get all users from users collection
            results = self.db.users.find({}, {"_id": 1})
            
            # Extract user IDs
            user_ids = [result["_id"] for result in results]
            
            return user_ids
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []


# Create a singleton instance
db = Database()
