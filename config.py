import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from constants import DEFAULT_TEXT_ANNOTATION_FILE, DEFAULT_DESTINATIONS
# Load the JSON data
with open(DEFAULT_TEXT_ANNOTATION_FILE, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Prepare sentences and labels
sentences = [item[0] for item in data["annotations"]]
labels = [item[1]['entities'] for item in data["annotations"]]
# Define tags
tags = data["classes"]
# tags = ['<pad>'] + tags

# Convert tags to indices
tag2idx = {tag: 0 for idx, tag in enumerate(tags)}
for label in labels:
    for entity in label:
        tag2idx[entity[1]] = tag2idx[entity[1]] + 1
# Sort the dictionary by values
sorted_tags_dict = dict(sorted(tag2idx.items(), key=lambda item: item[1],reverse=True))
sorted_tags = {key: value for key, value in sorted_tags_dict.items()}
sorted_tags = list(sorted_tags)

for i in range(len(sorted_tags)):
    sorted_tags[i] = sorted_tags[i].replace(" ", "_")
    
destinations = pd.read_excel(DEFAULT_DESTINATIONS)

vectorizer = CountVectorizer(max_features=10000, stop_words="english")
tags_vector = vectorizer.fit_transform(destinations["tags"].values.astype('U')).toarray()
tags_vector = tags_vector[1:]

feature_names = vectorizer.get_feature_names_out()
