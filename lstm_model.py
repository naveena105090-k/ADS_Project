import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Define Dataset for sequences
class UserInteractionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Define LSTM Architecture
class LSTMRecommender(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTMRecommender, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # out shape: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(df, seq_len=5):
    features = ["watch_time", "watch_percentage", "engagement_score", "liked"]
    
    # Sort just in case
    df = df.sort_values(by=["user_id", "step"])
    
    X, y = [], []
    # Group by user
    for user_id, group in df.groupby("user_id"):
        vals = group[features].values
        labels = group["language"].values
        
        # Create rolling sequences of length `seq_len`
        # Pad if a user has fewer than seq_len interactions
        for i in range(len(vals)):
            end_idx = i + 1
            start_idx = max(0, end_idx - seq_len)
            
            seq = vals[start_idx:end_idx]
            
            # Pad sequence if it's shorter than seq_len (pad with zeros at the beginning)
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), len(features)))
                seq = np.vstack([pad, seq])
                
            X.append(seq)
            # Predict the language associated with this step
            y.append(labels[i])
            
    return np.array(X), np.array(y)

def train_lstm_model(data_path="user_interactions.csv", seq_len=5):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Create label encoder for targets
    le = LabelEncoder()
    df["language_encoded"] = le.fit_transform(df["language"])
    
    # Scale features globally
    scaler = StandardScaler()
    features = ["watch_time", "watch_percentage", "engagement_score", "liked"]
    df[features] = scaler.fit_transform(df[features])
    
    print("Creating sequences...")
    X, y = create_sequences(df, seq_len=seq_len)
    
    # We already encoded in the dataframe, but create_sequences extracted strings
    # so let's transform the output array
    y_encoded = le.transform(y)
    
    # Create DataLoader
    dataset = UserInteractionDataset(X, y_encoded)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    input_size = len(features)
    hidden_size = 32
    num_classes = len(le.classes_)
    
    model = LSTMRecommender(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("Training LSTM Model...")
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
        
    print("Saving artifacts...")
    torch.save(model.state_dict(), "lstm_model.pth")
    with open("lstm_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("LSTM Training complete! Pipeline is ready for live simulator inference.")

if __name__ == "__main__":
    train_lstm_model()
