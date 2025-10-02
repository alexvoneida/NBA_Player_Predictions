import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

#prepare data
df = pd.read_parquet('player_game_logs_with_features_2024-25_TEST.parquet')

place_df = df.sort_values(['GAME_DATE'])
split_index = int(len(place_df) * 0.8)

train_df = place_df.iloc[:split_index]
test_df = place_df.iloc[split_index:]

features = ['MIN_last5', 'PTS_last5', 'REB_last5', 'AST_last5', 'USAGE_last5', 'IS_HOME', 'DAYS_REST', 'PLUS_MINUS_last5']

X_train = train_df[features].values.astype(np.float32)
y_train = train_df['PTS'].values.astype(np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test = test_df[features].values.astype(np.float32)
y_test = test_df['PTS'].values.astype(np.float32)

#convert to pytorch datasets
class NBAPlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#define datasets and dataloaders
train_ds = NBAPlayerDataset(X_train_scaled, y_train)
test_ds = NBAPlayerDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#print(f'Using device: {device}')

#define model
class NBALinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(NBALinearRegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)
    
model = NBALinearRegressionModel(input_dim=len(features))#.to(device)

#define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop

epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')