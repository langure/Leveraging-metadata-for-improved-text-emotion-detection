import pandas as pd
import sqlite3
import string  # To handle punctuation removal
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data from SQLite
conn = sqlite3.connect('youtube_comments.db')  # Adjust the path if needed
query = "SELECT * FROM comments"
data = pd.read_sql_query(query, conn)

# Step 1: Clean the text data (basic cleaning done)
def clean_text(text):
    # Example: lowercase, remove leading/trailing spaces, remove punctuation
    cleaned_text = text.lower().strip()
    # Remove punctuation from the text
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
    return cleaned_text

# Progress bar for cleaning text
tqdm.pandas(desc="Cleaning text data")
data['comment'] = data['comment'].progress_apply(clean_text)

# Step 2: Remove rows with empty comments
# Check for empty strings after cleaning and drop those rows
data = data[data['comment'] != '']

# Step 3: Encode the valence column (HIGH -> 1, LOW -> 0)
label_encoder = LabelEncoder()
data['valence'] = label_encoder.fit_transform(data['valence'])  # HIGH -> 1, LOW -> 0

# Step 4: Feature extraction from `published_at`
# Convert 'published_at' to datetime format
data['published_at'] = pd.to_datetime(data['published_at'], utc=True)

# Extract the day of the week (Monday=0, Sunday=6) as ordinal values
data['day_of_week'] = data['published_at'].dt.weekday  # Monday=0, Sunday=6

# Extract time of day (MORNING or NIGHT)
def classify_time_of_day(time):
    hour = time.hour
    if 7 <= hour < 19:  # 7:00 AM to 7:00 PM is MORNING
        return 1  # 1 if MORNING
    else:
        return 0  # 0 if NIGHT

# Progress bar for processing the time of day
tqdm.pandas(desc="Classifying time of day")
data['morning'] = data['published_at'].progress_apply(classify_time_of_day)

# Step 5: Prepare metadata (exclude 'author_profile_image_url', 'author_channel_id', 'author')
# Convert 'published_at' to a numeric timestamp for modeling purposes
data['published_at_timestamp'] = data['published_at'].astype(int) / 10**9  # Convert datetime to Unix timestamp

# Columns for metadata (excluding the unwanted columns)
metadata_cols = ['like_count', 'total_reply_count', 'published_at_timestamp', 'is_public', 'day_of_week', 'morning']

# Normalize metadata columns (except day_of_week and morning)
scaler = StandardScaler()
metadata_to_scale = ['like_count', 'total_reply_count', 'published_at_timestamp', 'is_public']
data[metadata_to_scale] = scaler.fit_transform(data[metadata_to_scale])

# Step 6: Create initial_dataset (only COMMENT and VALENCE)
initial_dataset = data[['comment', 'valence']]

# Step 7: Create initial_dataset_with_metadata (COMMENT, VALENCE, and other metadata)
initial_dataset_with_metadata = data[['comment', 'valence'] + metadata_cols]

# Step 8: Split the datasets into train and test, keeping label balance (VALENCE)
X_train, X_test, y_train, y_test = train_test_split(
    initial_dataset['comment'], initial_dataset['valence'], test_size=0.2, random_state=42, stratify=initial_dataset['valence'])

X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
    initial_dataset_with_metadata.drop('valence', axis=1), initial_dataset_with_metadata['valence'],
    test_size=0.2, random_state=42, stratify=initial_dataset_with_metadata['valence'])

# Save the datasets as CSV files
print("Saving datasets to CSV...")
initial_dataset.to_csv("initial_dataset.csv", index=False)
initial_dataset_with_metadata.to_csv("initial_dataset_with_metadata.csv", index=False)

# Save the train and test splits
train_df = pd.DataFrame({'comment': X_train, 'valence': y_train})
train_df.to_csv("train_initial_dataset.csv", index=False)

test_df = pd.DataFrame({'comment': X_test, 'valence': y_test})
test_df.to_csv("test_initial_dataset.csv", index=False)

train_meta_df = pd.concat([pd.DataFrame(X_train_meta), pd.DataFrame({'valence': y_train_meta})], axis=1)
train_meta_df.to_csv("train_initial_dataset_with_metadata.csv", index=False)

test_meta_df = pd.concat([pd.DataFrame(X_test_meta), pd.DataFrame({'valence': y_test_meta})], axis=1)
test_meta_df.to_csv("test_initial_dataset_with_metadata.csv", index=False)

print("All datasets saved successfully. Starting verification...")

# Step 9: Verification of CSV files with progress bar
csv_files = [
    "initial_dataset.csv",
    "initial_dataset_with_metadata.csv",
    "train_initial_dataset.csv",
    "test_initial_dataset.csv",
    "train_initial_dataset_with_metadata.csv",
    "test_initial_dataset_with_metadata.csv"
]

# Track overall verification status
all_files_verified = True

def verify_csv_file(file_path):
    global all_files_verified
    try:
        # Use 'python' engine to avoid buffer issues and proper quoting for robustness
        df = pd.read_csv(file_path, engine='python')
        for _ in tqdm(df.iterrows(), total=len(df), desc=f"Verifying {file_path}"):
            pass
        print(f"{file_path} verification successful!")
    except Exception as e:
        all_files_verified = False
        print(f"Error verifying {file_path}: {e}")

# Traverse all the CSV files
for csv_file in csv_files:
    verify_csv_file(csv_file)

# Final message
if all_files_verified:
    print("All CSV files have been verified successfully.")
else:
    print("Some CSV files failed verification. Please check the error messages.")