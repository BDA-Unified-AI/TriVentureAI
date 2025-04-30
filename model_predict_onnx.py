import numpy as np
import onnxruntime as ort
from DataProcessing import read_input
from load_data import sorted_tags
from transformers import AutoTokenizer
from constants import ONNX_MODEL_PATH,DEFAULT_PRETRAIN_MODEL_NAME_TOKENIZER
class Key_Ner_ONNX_Predictor:
    def __init__(self, model_path, tokenizer, tag_map):
        """
        Initialize the ONNX predictor.
        Args:
            model_path (str): Path to the ONNX model.
            tokenizer (BertTokenizer): Tokenizer to process input sentences.
            tag_map (Dict[int, str]): Mapping of indices to tags.
        """
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer
        self.tag_map = tag_map

    def predict(self, sentence):
        """
        Predict tags using the ONNX model.
        Args:
            sentence (str): Input sentence.
        Returns:
            Tuple[str, List[str]]: Original sentence and predicted tags.
        """
        sentence = read_input(sentence)
        tokens = self.tokenizer(sentence, return_tensors="np", padding=True, truncation=True)

        # Convert to int64 (ONNX requirement)
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        # Run inference
        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

        logits = outputs[0]
        predicted_tags = np.argmax(logits, axis=2)[0]

        # Convert indices to tags
        predicted_tags = [self.tag_map[idx] for idx in predicted_tags]
        print("PREDICTING", predicted_tags)
        predicted_tags = set(predicted_tags)
        predicted_tags.discard('<pad>')
        predicted_tags = [tag.replace(" ", "_") for tag in predicted_tags]

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True), predicted_tags

# Initialize ONNX-based predictor
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAIN_MODEL_NAME_TOKENIZER)
onnx_predictor = Key_Ner_ONNX_Predictor(
    model_path=ONNX_MODEL_PATH,
    tokenizer=tokenizer,
    tag_map=dict(enumerate(sorted_tags))
)
