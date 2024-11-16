# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import nltk
nltk.download('stopwords')

train_essays = pd.read_csv("/content/train_essays.csv")
test_essays = pd.read_csv("/content/test_essays.csv")

# Explore the training data
train_essays.info()

train_essays.head()

# Check for class balance
sns.countplot(data=train_essays, x='generated')
plt.show()

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    words = text.split()  # Tokenize
    words = [word.lower() for word in words if word.isalpha()]  # Lowercase and remove non-alphabetic words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

train_essays['clean_text'] = train_essays['text'].apply(clean_text)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_essays['clean_text'], train_essays['generated'], test_size=0.2, random_state=42)

#Tokenization and Encoding for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding=True, truncation=True, max_length=128)

encoded_train = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
encoded_val = tokenizer(X_val.tolist(), padding=True, truncation=True, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)

# Create TensorDatasets
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], train_labels)
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], val_labels)

# Define the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.2f}")

# Validation loop
model.eval()
val_preds = []
val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

  # Calculate validation accuracy
val_accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Test data processing
test_inputs = tokenizer(test_essays['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Move input tensor to the same device as the model
test_inputs = {key: value.to(device) for key, value in test_inputs.items()}

# Generate predictions using your trained model
with torch.no_grad():
    outputs = model(**test_inputs)
    logits = outputs.logits

# Assuming the first column of logits corresponds to the negative class (non-AI-generated)
# and the second column corresponds to the positive class (AI-generated)
predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Move predictions back to CPU

# Create a submission DataFrame with essay IDs and corresponding predictions
submission = pd.DataFrame({
    'id': test_essays['id'],
    'generated': predictions
})

# Save the submission DataFrame to a CSV file
submission.to_csv('/content/sample_submission.csv', index=False)

# Function to preprocess and predict for user input
def predict_user_input(model, tokenizer, device):
    while True:
        user_text = input("Enter a text to check if it's AI-generated (or type 'exit' to quit): ")
        if user_text.lower() == 'exit':
            print("Exiting...")
            break

        # Preprocess the text
        clean_user_text = clean_text(user_text)

        # Tokenize the input
        user_input = tokenizer(
            [clean_user_text], padding=True, truncation=True, return_tensors='pt', max_length=128
        )
        user_input = {key: value.to(device) for key, value in user_input.items()}

        # Predict using the trained model
        model.eval()
        with torch.no_grad():
            output = model(**user_input)
            logits = output.logits
            prediction = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Probability scores

        # Display prediction result
        print(f"Prediction: {'AI-generated' if prediction[1] > 0.5 else 'Human-written'}")
        print(f"Confidence: {prediction[1] if prediction[1] > 0.5 else 1 - prediction[1]:.2f}")

        # Optional: Collect correct label from user to evaluate accuracy
        try:
            correct_label = int(input("Enter the correct label (1 for AI-generated, 0 for Human-written): "))
            if correct_label not in [0, 1]:
                print("Invalid label. Skipping accuracy check for this input.")
            else:
                global total_user_inputs, correct_user_predictions
                total_user_inputs += 1
                if correct_label == (1 if prediction[1] > 0.5 else 0):
                    correct_user_predictions += 1
                print(f"Current Accuracy on User Inputs: {correct_user_predictions / total_user_inputs:.2f}")
        except ValueError:
            print("Invalid input. Skipping accuracy check for this input.")

# Initialize counters for user input accuracy
total_user_inputs = 0
correct_user_predictions = 0

# Call the user input function
predict_user_input(model, tokenizer, device)
