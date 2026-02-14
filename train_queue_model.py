#!/usr/bin/env python3
"""
Train a queue time prediction model for HPC jobs
"""
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("=" * 60)
print("Queue Time Prediction Model Training")
print("=" * 60)

# Load data
print("\n[1/4] Loading data...")
train_data = np.load('output/train.npz')
val_data = np.load('output/val.npz')
test_data = np.load('output/test.npz')

X_train = train_data['X']
y_train = train_data['y']
X_val = val_data['X']
y_val = val_data['y']
X_test = test_data['X']
y_test = test_data['y']

print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"  Val:   {X_val.shape[0]:,} samples")
print(f"  Test:  {X_test.shape[0]:,} samples")

# Load metadata to get feature names and target info
with open('output/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\n  Target variable: Queue wait time (seconds)")
print(f"  Engineered features: {len(metadata.get('engineered_features', []))}")
print(f"  Encoded categoricals: {len(metadata.get('encoded_categoricals', []))}")

# Train model
print("\n[2/4] Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)
print("  ✓ Training complete")

# Evaluate on validation set
print("\n[3/4] Evaluating model performance...")
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Metrics
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)

mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print("\n  Validation Set:")
print(f"    MAE:  {mae_val/60:.1f} minutes ({mae_val:.0f} seconds)")
print(f"    RMSE: {rmse_val/60:.1f} minutes ({rmse_val:.0f} seconds)")
print(f"    R²:   {r2_val:.4f}")

print("\n  Test Set:")
print(f"    MAE:  {mae_test/60:.1f} minutes ({mae_test:.0f} seconds)")
print(f"    RMSE: {rmse_test/60:.1f} minutes ({rmse_test:.0f} seconds)")
print(f"    R²:   {r2_test:.4f}")

# Show some predictions
print("\n  Sample Predictions (Test Set):")
print(f"  {'Actual (min)':>15} {'Predicted (min)':>18} {'Error (min)':>15}")
print("  " + "-" * 50)
for i in range(min(10, len(y_test))):
    actual_min = y_test[i] / 60
    pred_min = y_pred_test[i] / 60
    error_min = abs(actual_min - pred_min)
    print(f"  {actual_min:>15.1f} {pred_min:>18.1f} {error_min:>15.1f}")

# Save model
print("\n[4/4] Saving model...")
output_dir = Path('output')
model_path = output_dir / 'queue_time_model.joblib'
joblib.dump(model, model_path)
print(f"  ✓ Model saved to: {model_path}")

# Save model metadata
model_metadata = {
    "model_type": "GradientBoostingRegressor",
    "n_estimators": 100,
    "max_depth": 8,
    "target": "queue_wait_seconds",
    "n_features": X_train.shape[1],
    "n_train_samples": int(X_train.shape[0]),
    "metrics": {
        "validation": {
            "mae_seconds": float(mae_val),
            "rmse_seconds": float(rmse_val),
            "r2": float(r2_val)
        },
        "test": {
            "mae_seconds": float(mae_test),
            "rmse_seconds": float(rmse_test),
            "r2": float(r2_test)
        }
    }
}

metadata_path = output_dir / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=2)
print(f"  ✓ Model metadata saved to: {metadata_path}")

print("\n" + "=" * 60)
print("✓ Model training complete!")
print("=" * 60)
