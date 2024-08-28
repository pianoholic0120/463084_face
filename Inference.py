import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from tqdm import tqdm
from collections import OrderedDict

# Define the number of classes
NUM_CLASSES = 44
# NUM_CLASSES = 9

# Define transformations for the inference dataset
inference_transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384x384
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the inference dataset
inference_dataset = datasets.ImageFolder(root='./classification_training_images/training_images', transform=inference_transform)
inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)

# Define the LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, hidden_size, rank, alpha):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lora_layer.weight.data.normal_(0, alpha / rank)

    def forward(self, x):
        return self.lora_layer(x)

# Define the LoRA-augmented model
class LoRAModel(nn.Module):
    def __init__(self, model, lora_layer):
        super(LoRAModel, self).__init__()
        self.model = model
        self.lora_layer = lora_layer

    def forward(self, x):
        outputs = self.model.vit(x)
        hidden_states = outputs.last_hidden_state  # Extract hidden states
        lora_out = self.lora_layer(hidden_states)
        pooled_output = hidden_states[:, 0, :] + lora_out[:, 0, :]  # Add LoRA output to the [CLS] token
        logits = self.model.classifier(pooled_output)  # Final classification layer
        return logits

# Load the pre-trained ViT model
model_name = 'google/vit-base-patch16-384'
base_model = ViTForImageClassification.from_pretrained(model_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)

# Initialize the LoRA layer
hidden_size = 768  # Hidden size of the ViT model
rank = 8  # Rank parameter for LoRA
alpha = 32  # Scaling parameter for LoRA
lora_layer = LoRALayer(hidden_size, rank, alpha)

# Wrap the base model with the LoRA layer
lora_model = LoRAModel(base_model, lora_layer)

# Load trained weights (modify 'model_checkpoint.pth' to your trained model's checkpoint path)
checkpoint = torch.load('./model_checkpoints/new_dataset_test11/checkpoint_epoch_15.pth')
# Remove the 'module.' prefix from the keys
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.'
    new_state_dict[name] = v
lora_model.load_state_dict(new_state_dict)

# Move the model to GPU and set it to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model = lora_model.to(device)
lora_model.eval()

# Define the class names (modify according to your dataset)
# class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet', 
               # 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']
class_names = ['Ancistrus', 'Apistogramma','Astyanax','Bario','Black Sea Sprat','Bryconops','Bujurquina',
            'Bunocephalus','Catla','Characidium','Charax','Copella','Corydoras','Creagrutus','Curimata','Doras','Erythrinus',
            'Gasteropelecus','Gilt Head Bream','Grass Carp','Gymnotus','Hemigrammus','Horse Mackerel','Hyphessobrycon',
            'Knodus',  'Moenkhausia','Otocinclus','Oxyropsis','Phenacogaster','Pimelodella','Prochilodus','Pygocentrus',
            'Pyrrhulina','Red Mullet','Red Sea Bream','Rineloricaria','Sea Bass','Shrimp','Sorubim','Striped Red Mullet','Tatia',
            'Tetragonopterus','Trout','Tyttocharax']

# Inference loop
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(inference_loader, desc='Inference'):
        images = images.to(device)
        labels = labels.to(device)
        outputs = lora_model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate and print accuracy
correct = sum(p == l for p, l in zip(all_preds, all_labels))
total = len(all_labels)
accuracy = correct / total
print(f'Inference Accuracy: {accuracy * 100:.2f}%')

# Print results
# for idx, (pred, label) in enumerate(zip(all_preds, all_labels)):
    # print(f"Image {idx}: Predicted - {class_names[pred]}, Actual - {class_names[label]}")

# Optional: Save predictions to a file
with open('inference_results.txt', 'w') as f:
    for idx, (pred, label) in enumerate(zip(all_preds, all_labels)):
        f.write(f"Image {idx}: Predicted - {class_names[pred]}, Actual - {class_names[label]}\n")
