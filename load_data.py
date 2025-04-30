import json
from constants import DEFAULT_TEXT_ANNOTATION_FILE

with open(DEFAULT_TEXT_ANNOTATION_FILE, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
# Prepare sentences and labels
sentences = [item[0] for item in data["annotations"]]
"""
List[str]: A list of sentences extracted from the dataset.
Each sentence corresponds to an annotation in the dataset.
"""

labels = [item[1]['entities'] for item in data["annotations"]]
"""
List[List[Tuple[str, str]]]: A list of entity labels for each sentence.
Each label is a tuple containing the entity text and its corresponding tag.
"""

# Define tags
tags = data["classes"]
"""
List[str]: A list of all possible entity tags (classes) in the dataset.
These tags will be used to label the tokens in each sentence.
"""

# Convert tags to indices
tag2idx = {tag: 0 for idx, tag in enumerate(tags)}
for label in labels:
    for entity in label:
        tag2idx[entity[1]] = tag2idx[entity[1]] + 1
# Sort the dictionary by values
sorted_tags = dict(sorted(tag2idx.items(), key=lambda item: item[1],reverse=True))
sorted_tags = {key: value for key, value in sorted_tags.items() if value != 0}

new_tag = {'<pad>': 0}

sorted_tags = {**new_tag, **sorted_tags}
