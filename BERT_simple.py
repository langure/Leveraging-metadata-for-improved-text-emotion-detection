import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json

EPOCHS = 10

# Create a folder for the model outputs
model_name = 'BERT_SIMPLE'
if not os.path.exists(model_name):
    os.makedirs(model_name)

# Load the pre-split datasets (limit to 1000 rows for testing)
train_data = pd.read_csv('train_initial_dataset.csv', engine='python').head(100)
test_data = pd.read_csv('test_initial_dataset.csv', engine='python').head(100)

# Convert labels to integers if not already done
train_data['valence'] = train_data['valence'].astype(int)
test_data['valence'] = test_data['valence'].astype(int)

X_train = train_data['comment'].to_list()
y_train = train_data['valence'].to_list()
X_test = test_data['comment'].to_list()
y_test = test_data['valence'].to_list()

# Load BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the datasets
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Convert the encodings to PyTorch datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are of type long
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, y_train)
test_dataset = Dataset(test_encodings, y_test)

# Define evaluation metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'./{model_name}/results',
    evaluation_strategy="epoch",  # Set to "epoch"
    save_strategy="epoch",        # Set to "epoch" to match evaluation strategy
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=f'./{model_name}/logs',
    logging_steps=10,
    load_best_model_at_end=True,  # Keep this true for loading the best model
)

# Define Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Save the model and tokenizer
model.save_pretrained(f'./{model_name}')
tokenizer.save_pretrained(f'./{model_name}')

# Save evaluation results to a JSON file
with open(f'./{model_name}/bert_simple_stats.json', 'w') as f:
    json.dump(eval_results, f, indent=4)

# Load model and tokenizer for inference
model = BertForSequenceClassification.from_pretrained(f'./{model_name}')
tokenizer = BertTokenizer.from_pretrained(f'./{model_name}')

# Inference function
def detect_valence_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    return prediction

# Example usage
example_text = "This is a great day!"
valence = detect_valence_with_bert(example_text)
print(f"Predicted valence: {valence}")