import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#prepare data
df1 = pd.read_parquet('final_database_2024-25.parquet')
df1 = df1.dropna()

place_df = df1.sort_values(['GAME_DATE'])
split_index1 = int(len(place_df) * 0.4)
split_index2 = int(len(place_df) * 0.6)

train_df = place_df.iloc[:split_index1]
train_df = pd.concat([train_df, place_df.iloc[split_index1:split_index2]])
test_df = place_df.iloc[split_index1:split_index2]

features = ['MIN_last5', 'PTS_last5', 'REB_last5', 'AST_last5', 'FG_PCT_last5', 'USAGE_last5', 'IS_HOME', 'DAYS_REST', 'PLUS_MINUS_last5', 'offensiveRating_last5', 'defensiveRating_last5', 'pace_last5', 'OPP_offensiveRating_last5', 'OPP_defensiveRating_last5', 'OPP_pace_last5']

X_train = train_df[features].values.astype(np.float32)
y_train = train_df['FG_PCT'].values.astype(np.float32)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test = test_df[features].values.astype(np.float32)
y_test = test_df['FG_PCT'].values.astype(np.float32)

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
train_ds = NBAPlayerDataset(X_train, y_train)
test_ds = NBAPlayerDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

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
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
epochs = 50
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
    
#evaluation
model.eval()
preds_list, true_list = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        preds_list.extend(preds.numpy().flatten())
        true_list.extend(y_batch.numpy().flatten())

mae = mean_absolute_error(true_list, preds_list)
rmse = np.sqrt(mean_squared_error(true_list, preds_list))
r2 = r2_score(true_list, preds_list)

print(f"PyTorch NN → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")