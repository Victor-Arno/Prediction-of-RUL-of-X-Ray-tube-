import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Outlier handling function (IQR method)
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].median(), df[column])
    return df

# Smoothing function (Savitzky-Golay filter)
def smooth_data(df, column, window_length=10, polyorder=2):
    # Get the length of the current column data
    data_length = len(df[column])
    # Ensure window_length does not exceed the data length and is odd
    window_length = min(window_length, data_length)
    if window_length % 2 == 0:
        window_length -= 1
    # Ensure window_length is at least 3, as Savitzky-Golay filter requires a minimum window_length of 3
    window_length = max(3, window_length)
    smoothed = savgol_filter(df[column], window_length=window_length, polyorder=polyorder)
    return smoothed

# 1. Data loading and preprocessing
def load_and_preprocess(file_paths):
    data = []
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            # Filter data where f3=3, f4=0, f5=0.75
            df = df[(df['f3'] == 3) & (df['f4'] == 0) & (df['f5'] == 0.75)]
            if not df.empty:
                data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # Process each tube DataFrame individually
    for i in range(len(data)):
        tube_df = data[i]
        # Handle outliers for columns f6 to f16
        for column in tube_df.columns[5:16]:  # Select f6 to f16 (index 5 to 15)
            if tube_df[column].dtype != 'object':  # Only process numerical columns
                tube_df = handle_outliers(tube_df, column)

        # Initialize RobustScaler
        scaler = RobustScaler()
        # Initialize MinMaxScaler
        minmax_scaler = MinMaxScaler()

        # Apply robust standardization to columns f6 to f16
        columns_to_scale = []
        for column in tube_df.columns[5:16]:
            if tube_df[column].dtype != 'object':
                columns_to_scale.append(column)

        if columns_to_scale:
            tube_df[columns_to_scale] = scaler.fit_transform(tube_df[columns_to_scale])

        # Apply smoothing to columns f6 to f16
        for column in tube_df.columns[5:16]:  # Select f6 to f16 (index 5 to 15)
            if tube_df[column].dtype != 'object':  # Only smooth numerical columns
                # Smooth each column
                tube_df[column + '_smoothed'] = smooth_data(tube_df, column)

        # Apply MinMax normalization to the smoothed columns
        smoothed_columns = [col for col in tube_df.columns if col.endswith('_smoothed')]
        if smoothed_columns:
            tube_df[smoothed_columns] = minmax_scaler.fit_transform(tube_df[smoothed_columns])

        data[i] = tube_df

    df = pd.concat(data, ignore_index=True)
    return df, data

# 2. Health indicator construction
def build_health_indicator(data):
    health_indicators = []
    for df in data:
        # Assume f9 is used as the current feature, adjust as needed
        current_col = 'f9'
        if current_col + '_smoothed' in df.columns:
            current_col = current_col + '_smoothed'
        min_val = df[current_col].min()
        max_val = df[current_col].max()
        # Avoid division by zero
        if max_val - min_val < 1e-6:
            df['health_indicator'] = np.zeros(len(df))
        else:
            df['health_indicator'] = (df[current_col] - min_val) / (max_val - min_val)
        health_indicators.append(df['health_indicator'].values)
    return health_indicators

# Visualize health indicators function
def visualize_health_indicators(data, failure_threshold=None):
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, df in enumerate(data, 1):
        plt.plot(df['f1'], df['health_indicator'], color=colors[i - 1], label=f'Tube {i} Health Indicator')
    if failure_threshold is not None:
        plt.axhline(y=failure_threshold, color='k', linestyle='--', label='Failure Threshold')
    plt.title('Health Indicator over Time for Each Tube')
    plt.xlabel('Time')
    plt.ylabel('Health Indicator')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. RUL label generation
def generate_rul_labels(health_indicators, failure_threshold=0.25, max_rul=200):
    rul_labels = []
    for hi in health_indicators:
        # Find the first point below the failure threshold
        below_threshold = hi < failure_threshold
        if np.any(below_threshold):
            failure_idx = np.argmax(below_threshold)
        else:  # If no point is below the threshold
            failure_idx = len(hi)
        rul = np.zeros(len(hi))
        for i in range(len(hi)):
            if i > failure_idx - max_rul:
                rul[i] = max(0, failure_idx - i)
            else:
                rul[i] = max_rul
        rul_labels.append(rul)
    return rul_labels

# 4. Feature engineering - Enhanced feature extraction
def create_features(data, health_indicators, rul_labels, window_size=5, visualize=False):
    if not data or not health_indicators or not rul_labels:
        raise ValueError("No valid data available for feature creation")
    X, y = [], []
    all_features = []
    all_times = []
    for i, (df, hi, rul) in enumerate(zip(data, health_indicators, rul_labels)):
        # Ensure all arrays have the same length
        min_length = min(len(df), len(hi), len(rul))
        df = df.iloc[:min_length]
        hi = hi[:min_length]
        rul = rul[:min_length]
        # Adjust window_size to ensure it does not exceed the data length
        adjusted_window_size = window_size
        while adjusted_window_size >= min_length:
            adjusted_window_size -= 1
            if adjusted_window_size < 1:
                print(f"Tube {i + 1} has insufficient data for feature creation. Skipping this tube.")
                break
        if adjusted_window_size < 1:
            continue
        tube_features = []
        tube_times = []
        for j in range(len(hi) - adjusted_window_size):
            # Sliding window features
            window = hi[j:j + adjusted_window_size]
            # Enhanced statistical features
            features = [
                np.mean(window),        # Mean
                np.std(window) if len(window) > 1 else 0.0,  # Standard deviation
                np.min(window),         # Minimum
                np.max(window),         # Maximum
                np.max(window) - np.min(window),  # Peak-to-peak
                np.median(window),      # Median
                np.mean(np.abs(np.diff(window))) if len(window) > 1 else 0.0,  # Average change rate
                np.std(np.diff(window)) if len(window) > 1 else 0.0            # Change rate standard deviation
            ]
            tube_features.append(features)
            # Target value (RUL at the end of the window)
            target = rul[j + adjusted_window_size]
            X.append(features)
            y.append(target)
            # Record the time at the end of the window
            tube_times.append(df['f1'].values[j + adjusted_window_size])
        all_features.append(np.array(tube_features))
        all_times.append(np.array(tube_times))
    if visualize and all_features:
        feature_names = [
            'Mean', 'Std Dev', 'Min', 'Max', 'Peak-to-Peak',
            'Median', 'Avg Change Rate', 'Change Rate Std Dev'
        ]
        for tube_idx in range(len(data)):
            # Check array dimensions
            if all_features[tube_idx].ndim < 2:
                print(f"Tube {tube_idx + 1} has insufficient data for visualization. Skipping.")
                continue
            plt.figure(figsize=(15, 10))
            for feature_idx in range(len(feature_names)):
                plt.subplot(3, 3, feature_idx + 1)
                plt.plot(all_times[tube_idx], all_features[tube_idx][:, feature_idx], label=feature_names[feature_idx])
                plt.title(f'Tube {tube_idx + 1} - {feature_names[feature_idx]}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
            plt.tight_layout()
            plt.show()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 5. Improved Lightweight Model - Avoid inplace operation issues
class LightweightRULPredictor(nn.Module):
    def __init__(self, input_size):
        super(LightweightRULPredictor, self).__init__()
        # Input layer to transform input features to 64 dimensions
        self.input_layer = nn.Linear(input_size, 64)
        # Residual Block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        # Residual Block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        # Frequency-aware layer
        self.freq_adapter = nn.Linear(64, 64)
        # Output layer to produce the final RUL prediction
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through the input layer and apply ReLU activation
        x = self.relu(self.input_layer(x))
        # Residual connection 1
        identity1 = x
        out = self.res_block1(x)
        x = out + identity1
        x = self.relu(x)
        # Residual connection 2
        identity2 = x
        out = self.res_block2(x)
        x = out + identity2
        x = self.relu(x)
        # Frequency adaptation
        x = self.relu(self.freq_adapter(x))
        # Pass through the output layer
        x = self.output_layer(x)
        return x

# 6. Main process
def main():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the Training_image folder
    image_folder = "Training_image"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Create the weights folder
    weights_folder = "model_weights"
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    # Load data from the data folder
    tube_files = [os.path.join('data', f'tube{i}.csv') for i in range(1, 6)]
    df, data = load_and_preprocess(tube_files)
    if not data:
        print("No valid data processed. Exiting.")
        return None
    # Build health indicators
    health_indicators = build_health_indicator(data)
    visualize_health_indicators(data)
    # Generate RUL labels
    rul_labels = generate_rul_labels(health_indicators, failure_threshold=0.48, max_rul=200000)
    # Create features
    X, y = create_features(data, health_indicators, rul_labels, window_size=5, visualize=False)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=3
    )
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # Initialize the improved lightweight model
    model = LightweightRULPredictor(X_train.shape[1]).to(device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add weight decay
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    # Train the model
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
        # Update learning rate
        scheduler.step(val_loss)
        # Early stopping mechanism
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            # Save the best model to the weights folder
            model_path = os.path.join(weights_folder, 'best_lightweight_rul_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model at epoch {epoch + 1} with loss {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        # Print training progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    # Load the best model from the weights folder
    model_path = os.path.join(weights_folder, 'best_lightweight_rul_model.pth')
    model.load_state_dict(torch.load(model_path))
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor).cpu().numpy().flatten()
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)
    print(f"Test Results - MAE: {mae:.2f}, R²: {r2:.4f}")
    # Visualize the training process
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Lightweight Model)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Modify the save path
    plt.savefig(os.path.join(image_folder, 'training_curve.png'))
    plt.show()
    # Visualize the prediction results
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, test_pred, alpha=0.5, label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Line')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'RUL Prediction with Lightweight Model (MAE={mae:.2f}, R²={r2:.4f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Modify the save path
    plt.savefig(os.path.join(image_folder, 'rul_prediction.png'))
    plt.show()
    # Return the trained model
    return model

# Execute the main process
if __name__ == "__main__":
    trained_model = main()