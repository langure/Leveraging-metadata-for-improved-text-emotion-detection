import os
import pandas as pd
import pickle
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import load_model

EPOCHS = 10

# Create a folder for the model outputs
model_name = 'BILSTM_SIMPLE'
if not os.path.exists(model_name):
    os.makedirs(model_name)

# Load the pre-split datasets and limit to 1000 rows
train_data = pd.read_csv('train_initial_dataset.csv', engine='python').head(1000)
test_data = pd.read_csv('test_initial_dataset.csv', engine='python').head(1000)

X_train = train_data['comment']
y_train = train_data['valence']
X_test = test_data['comment']
y_test = test_data['valence']

max_words = 20000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Save the tokenizer to the model folder
with open(f'{model_name}/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

def build_bilstm_model(vocab_size, embedding_dim, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dense(1, activation='sigmoid'))
    return model

vocab_size = min(max_words, len(tokenizer.word_index) + 1)
embedding_dim = 128

model = build_bilstm_model(vocab_size=vocab_size, embedding_dim=embedding_dim, max_len=max_len)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

class TQDMProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.progress_bar = tqdm(total=len(X_train_pad), desc=f'Epoch {epoch + 1}', unit='samples')

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(64)  # Update by batch size

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=EPOCHS, batch_size=64, callbacks=[early_stopping, TQDMProgressBar()]
)

# Save the trained model to the model folder
model.save(f'{model_name}/bilstm_valence_model.h5')

eval_metrics = model.evaluate(X_test_pad, y_test)
eval_loss, eval_accuracy = eval_metrics[0], eval_metrics[1]

y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

classification_stats = classification_report(y_test, y_pred, output_dict=True)

stats = {
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,
    'classification_report': classification_stats
}

# Save statistics to a JSON file in the model folder
with open(f'{model_name}/bilstm_valence_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print(f"Training and evaluation statistics saved to '{model_name}/bilstm_valence_stats.json'")

# Load tokenizer and model for inference
with open(f'{model_name}/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model(f'{model_name}/bilstm_valence_model.h5')

def detect_valence(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len)
    prediction = model.predict(text_pad)
    return 1 if prediction > 0.5 else 0

example_text = "This is a great day!"
valence = detect_valence(example_text)
print(f"Predicted valence: {valence}")