import os
import pandas as pd
import pickle
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate, Input, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import load_model
import numpy as np

EPOCHS = 10

# Create a folder for the model outputs
model_name = 'BILSTM_METADATA'
if not os.path.exists(model_name):
    os.makedirs(model_name)

# Load the pre-split datasets with metadata (limit to 1000 rows for testing)
train_data = pd.read_csv('train_initial_dataset_with_metadata.csv', engine='python').head(1000)
test_data = pd.read_csv('test_initial_dataset_with_metadata.csv', engine='python').head(1000)

# Separate text and labels
X_train_text = train_data['comment']
y_train = train_data['valence']
X_test_text = test_data['comment']
y_test = test_data['valence']

# Separate metadata features (excluding the comment and valence columns)
metadata_cols = ['like_count', 'total_reply_count', 'published_at_timestamp', 'is_public', 'day_of_week', 'morning']
X_train_metadata = train_data[metadata_cols].values
X_test_metadata = test_data[metadata_cols].values

max_words = 20000
max_len = 200

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Pad sequences for uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Save the tokenizer to the model folder
with open(f'{model_name}/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Build the BiLSTM model with metadata for training
def build_bilstm_model_with_metadata(vocab_size, embedding_dim, max_len, metadata_input_shape):
    # Text input
    text_input = Input(shape=(max_len,), name='text_input')
    text_embedding = Embedding(vocab_size, embedding_dim)(text_input)
    text_lstm = Bidirectional(LSTM(128, return_sequences=False))(text_embedding)

    # Metadata input (only used during training)
    metadata_input = Input(shape=(metadata_input_shape,), name='metadata_input')  # Ensure the shape is a tuple
    metadata_dense = Dense(32, activation='relu')(metadata_input)

    # Concatenate the outputs of text LSTM and metadata
    concatenated = Concatenate()([text_lstm, metadata_dense])
    concatenated = Dropout(0.3)(concatenated)  # Optional dropout for regularization
    output = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[text_input, metadata_input], outputs=output)
    return model

vocab_size = min(max_words, len(tokenizer.word_index) + 1)
embedding_dim = 128
metadata_input_shape = X_train_metadata.shape[1]

# Build and compile the model
model = build_bilstm_model_with_metadata(vocab_size=vocab_size, embedding_dim=embedding_dim, max_len=max_len, metadata_input_shape=metadata_input_shape)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

class TQDMProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.progress_bar = tqdm(total=len(X_train_pad), desc=f'Epoch {epoch + 1}', unit='samples')

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(64)  # Update by batch size

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()

# Train the model with both text and metadata
history = model.fit(
    [X_train_pad, X_train_metadata], y_train,
    validation_data=([X_test_pad, X_test_metadata], y_test),
    epochs=EPOCHS, batch_size=64, callbacks=[early_stopping, TQDMProgressBar()]
)

# Save the trained model in the Keras format
model.save(f'{model_name}/bilstm_metadata_model.keras')

eval_metrics = model.evaluate([X_test_pad, X_test_metadata], y_test)
eval_loss, eval_accuracy = eval_metrics[0], eval_metrics[1]

y_pred = (model.predict([X_test_pad, X_test_metadata]) > 0.5).astype("int32")

classification_stats = classification_report(y_test, y_pred, output_dict=True)

stats = {
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,
    'classification_report': classification_stats
}

# Save statistics to a JSON file in the model folder
with open(f'{model_name}/bilstm_metadata_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print(f"Training and evaluation statistics saved to '{model_name}/bilstm_metadata_stats.json'")

# Load tokenizer and model for inference
with open(f'{model_name}/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model(f'{model_name}/bilstm_metadata_model.keras')

# Inference function that only takes text input, ignoring metadata
def detect_valence_with_text_only(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len)
    prediction = model.predict([text_pad, np.zeros((1, metadata_input_shape))])  # Feed zeroed metadata
    return 1 if prediction > 0.5 else 0

example_text = "This is a great day!"
valence = detect_valence_with_text_only(example_text)
print(f"Predicted valence: {valence}")