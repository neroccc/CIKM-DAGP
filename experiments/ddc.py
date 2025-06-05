import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer once during import
MODEL_PATH = "checkpoints/deepseekcoder/checkpoint-best-f1"  # Update this to your actual model directory
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval().to(DEVICE)

def calculate_D(text: str):
    """
    Given a text input, returns a 3-element list representing the difficulty distribution:
    [prob_easy, prob_medium, prob_hard]
    """
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)
        #sorted_probs, _ = torch.sort(probs, descending=True)
        return probs.cpu().tolist()

