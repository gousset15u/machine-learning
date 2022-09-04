import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import logistic_regression as lr   # <-- Your implementation

sns.set_style('darkgrid') # Seaborn plotting style 

print("=== Data set 1 ===")
data_1 = pd.read_csv('data_1.csv')
data_1.head()

plt.figure(figsize=(5, 5))
sns.scatterplot(x='x0', y='x1', hue='y', data=data_1)

# Partition data into independent (feature) and depended (target) variables
X = data_1[['x0', 'x1']].to_numpy()
y = data_1['y'].to_numpy()

# Create model
model_1 = lr.LogisticRegression(0.15,2) # <-- Should work with default constructor  

# Train model
model_1.fit(X, y)

# Calculate accuracy and cross entropy for (insample) predictions 
y_pred = model_1.predict(X)
print(f'Accuracy: {lr.binary_accuracy(y_true=y, y_pred=y_pred, threshold=0.5) :.3f}')
print(f'Cross Entropy: {lr.binary_cross_entropy(y_true=y, y_pred=y_pred) :.3f}')

# Rasterize the model's predictions over a grid
xx0, xx1 = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
yy = model_1.predict(np.stack([xx0,xx1], axis=-1).reshape(-1, 2)).reshape(xx0.shape)

# Plot prediction countours along with datapoints
_, ax = plt.subplots(figsize=(4, 4), dpi=100)
levels = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
contours = ax.contourf(xx0, xx1, yy, levels=levels, alpha=0.4, cmap='RdBu_r', vmin=0, vmax=1)
legends = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in contours.collections]
labels = [f'{a :.2f} - {b :.2f}' for a,b in zip(levels, levels[1:])]
sns.scatterplot(x='x0', y='x1', hue='y', ax=ax, data=data_1)
ax.legend(legends, labels, bbox_to_anchor=(1,1))


# Load second dataset and partition into train/test split
print("\n=== Data set 2 ===")

data_2 = pd.read_csv('data_2.csv')
data_2.head()

data_2_train = data_2.query('split == "train"')
data_2_test = data_2.query('split == "test"')

# Partition data into independent (features) and depended (targets) variables
X_train, y_train = data_2_train[['x0', 'x1']].to_numpy(), data_2_train['y'].to_numpy()
X_test, y_test = data_2_test[['x0', 'x1']].to_numpy(), data_2_test['y'].to_numpy()

new_X_train = model_1.build_dim(X_train)
new_X_test = model_1.build_dim(X_test)

# Fit model (TO TRAIN SET ONLY)
model_2 = lr.LogisticRegression(0.55,2)  # <--- Feel free to add hyperparameters
model_2.fit(new_X_train, y_train)

# Calculate accuracy and cross entropy for insample predictions 
y_pred_train = model_2.predict(new_X_train)
print('Train')
print(f'Accuracy: {lr.binary_accuracy(y_true=y_train, y_pred=y_pred_train, threshold=0.5) :.3f}')
print(f'Cross Entropy:  {lr.binary_cross_entropy(y_true=y_train, y_pred=y_pred_train) :.3f}')

# Calculate accuracy and cross entropy for out-of-sample predictions
y_pred_test = model_2.predict(new_X_test)
print('\nTest')
print(f'Accuracy: {lr.binary_accuracy(y_true=y_test, y_pred=y_pred_test, threshold=0.5) :.3f}')
print(f'Cross Entropy:  {lr.binary_cross_entropy(y_true=y_test, y_pred=y_pred_test) :.3f}')