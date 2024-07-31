import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define class names as they appear in your dataset
class_names = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet', 
    'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout'
]

# Create a mapping from class names to indices
class_to_index = {class_name: i for i, class_name in enumerate(class_names)}

# Function to parse the results file and extract predicted and actual labels
def parse_results(file_path):
    predicted_labels = []
    actual_labels = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Extract predicted and actual labels
                parts = line.split('Predicted - ')[1].split(', Actual - ')
                predicted_label = parts[0].strip()
                actual_label = parts[1].strip()
                predicted_labels.append(class_to_index[predicted_label])
                actual_labels.append(class_to_index[actual_label])
            except (IndexError, KeyError) as e:
                print(f"Skipping line due to error: {e}")
    return predicted_labels, actual_labels

# Path to your inference results file
file_path = 'inference_results.txt'
predicted_labels, actual_labels = parse_results(file_path)

# Compute the confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels, labels=list(range(len(class_names))))

# Plot confusion matrix
plt.figure(figsize=(12, 10))  # Increase figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure
plt.savefig('confusion_matrix.png')  # Save the figure as a file
plt.close()

# Print classification report
report = classification_report(actual_labels, predicted_labels, target_names=class_names, zero_division=0)
print(report)

# Save classification report to file
with open('classification_report.txt', 'w') as f:
    f.write(report)
