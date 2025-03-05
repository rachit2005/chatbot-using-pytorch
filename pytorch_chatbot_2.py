import streamlit as st
import torch
import json
import random
import nltk
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os

# Ensure nltk dependencies are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Define the ChatBotModel class
class ChatBotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = self.dropout(self.relu(self.fc1(X)))
        X = self.dropout(self.relu(self.fc2(X)))
        X = self.fc3(X)
        return X

# Define the ChatBotAssistant class
class ChatBotAssistant:
    def __init__(self, intents_path, model_path, dimensions_path):
        self.intents_path = intents_path
        self.model_path = model_path
        self.dimensions_path = dimensions_path
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.documents = []
        self.model = None
        self.X = None
        self.y = None
        self.load_data()
        self.load_model()
    
    def tokenise_and_lemmatize(self, text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words):
        return np.array([1 if word in words else 0 for word in self.vocabulary], dtype=np.float32)

    def load_data(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            for intent in intents_data['intents']:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']
                for pattern in intent['patterns']:
                    pattern_words = self.tokenise_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
            self.vocabulary = sorted(set(self.vocabulary))
            self.prepare_data()
    
    def prepare_data(self):
        bags = []
        indices = []
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size=32, lr=0.001, epochs=1000):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.model = ChatBotModel(self.X.shape[1], len(self.intents))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = loss_fn(output, batch_y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")
        self.save_model()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        with open(self.dimensions_path, 'w') as f:
            json.dump({'input_size': len(self.vocabulary), 'output_size': len(self.intents)}, f)
    
    def load_model(self):
        if os.path.exists(self.dimensions_path) and os.path.exists(self.model_path):
            with open(self.dimensions_path, 'r') as f:
                dimensions = json.load(f)
            self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()

    
    def process_message(self, message):
        try:
            result = eval(message)
            return f"Solution: {result}"
        except:
            words = self.tokenise_and_lemmatize(message)
            bag = self.bag_of_words(words)
            bag_tensor = torch.tensor([bag], dtype=torch.float32)
            with torch.no_grad():
                preds = self.model(bag_tensor)
            pred_class_idx = torch.argmax(preds, dim=1).item()
            pred_intent = self.intents[pred_class_idx]
            return random.choice(self.intents_responses[pred_intent])

# Streamlit UI
def main():
    st.title("AI Chatbot Assistant")
    chatbot = ChatBotAssistant('edu_intents.json', 'chatbot_model.pth', 'dimensions.json')

    if st.button("Train Model"):
        chatbot.train_model()
        st.success("Model training completed!")

    user_input = st.text_input("Enter your message:")
    if st.button("Send") and user_input:
        response = chatbot.process_message(user_input)
        st.text_area("Chatbot Response:", response, height=100)

if __name__ == "__main__":
    main()
