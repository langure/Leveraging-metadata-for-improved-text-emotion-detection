import os
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

EPOCHS = 10

# Create a folder for the model outputs
model_name = 'BERT_METADATA_PYTORCH'
if not os.path.exists(model_name):
    os.makedirs(model_name)

# Load the pre-split datasets (limit to 1000 rows for testing)
train_data = pd.read_csv('train_initial_dataset_with_metadata.csv', engine='python').head(100)
test_data = pd.read_csv('test_initial_dataset_with_metadata.csv', engine='python').head(100)

# Convert labels to integers if not already done
train_data['valence'] = train_data['valence'].astype(int)
test_data['valence'] = test_data['valence'].astype(int)

X_train = train_data['comment'].to_list()
y_train = train_data['valence'].to_list()
X_test = test_data['comment'].to_list()
y_test = test_data['valence'].to_list()

# Metadata columns (excluding unnecessary ones like author name, profile image, etc.)
metadata_columns = ['like_count', 'total_reply_count', 'is_public', 'day_of_week', 'morning']

# Scale the metadata features
scaler = StandardScaler()
train_metadata = scaler.fit_transform(train_data[metadata_columns])
test_metadata = scaler.transform(test_data[metadata_columns])

# Load BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the datasets
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Custom model to handle BERT output and metadata
class BertWithMetadata(nn.Module):
    def __init__(self):
        super(BertWithMetadata, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.metadata_dense = nn.Linear(len(metadata_columns), 16)  # A dense layer to handle metadata
        self.classifier = nn.Linear(768 + 16, 2)  # Combining BERT’s output with the metadata output

    def forward(self, input_ids, attention_mask, metadata=None):
        # Pass text through BERT
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Get pooled output from BERT

        if metadata is not None:
            # Pass metadata through a dense layer during training
            metadata_output = self.metadata_dense(metadata)
            # Concatenate BERT output with metadata output
            combined_output = torch.cat((pooled_output, metadata_output), dim=1)
        else:
            # If no metadata, just use BERT output for inference
            combined_output = pooled_output

        # Pass through the final classifier
        logits = self.classifier(combined_output)
        return logits

# Convert the encodings and metadata to PyTorch datasets
class DatasetWithMetadata(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, metadata):
        self.encodings = encodings
        self.labels = labels
        self.metadata = metadata

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['metadata'] = torch.tensor(self.metadata[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DatasetWithMetadata(train_encodings, y_train, train_metadata)
test_dataset = DatasetWithMetadata(test_encodings, y_test, test_metadata)

# Define evaluation metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'./{model_name}/results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=f'./{model_name}/logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

# Define custom trainer to handle metadata
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], metadata=inputs.get('metadata', None))
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize the custom model and trainer
model = BertWithMetadata()

trainer = CustomTrainer(
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
model.bert.save_pretrained(f'./{model_name}')  # Save the BERT part of the model
tokenizer.save_pretrained(f'./{model_name}')

# Save evaluation results to a JSON file
with open(f'./{model_name}/bert_metadata_stats.json', 'w') as f:
    json.dump(eval_results, f, indent=4)

# Load model and tokenizer for inference
model.bert = BertForSequenceClassification.from_pretrained(f'./{model_name}')
tokenizer = BertTokenizer.from_pretrained(f'./{model_name}')

# Inference function (text-only for prediction)
def detect_valence_with_bert_metadata(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model.bert(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    return prediction

# Example usage (text-only inference)
example_text = "This is a great day!"
valence = detect_valence_with_bert_metadata(example_text)
print(f"Predicted valence: {valence}")