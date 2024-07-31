import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
import os

# Define the save path for the model
save_path = './model_checkpoints/Full'
os.makedirs(save_path, exist_ok=True)

# Function to save the model
def save_model(epoch, model, optimizer, scheduler, save_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'num_classes': NUM_CLASSES,
        'transform': transform,  # Save the transform used for training
    }
    torch.save(state, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
    print(f'Model saved at epoch {epoch+1}.')


# Define the number of classes
NUM_CLASSES = 9

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir='./runs/Fish-dataset-Recognition-80-20_newlr')

# Define transformations for your dataset
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384x384
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your fish dataset
full_dataset = datasets.ImageFolder(root='./Fish_Dataset/Dataset', transform=transform)

# Define the proportion of the dataset to be used for training
train_size = int(0.8 * len(full_dataset))  # 80% for training
val_size = len(full_dataset) - train_size  # 20% for validation

# Randomly split the dataset into training and validation sets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ViT model
model_name = 'google/vit-base-patch16-384'
model = ViTForImageClassification.from_pretrained(model_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)

# Create a custom LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, hidden_size, rank, alpha):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lora_layer.weight.data.normal_(0, alpha / rank)

    def forward(self, x):
        return self.lora_layer(x)

# Initialize LoRA and integrate it with the model
hidden_size = 768  # Hidden size of the ViT model
rank = 4  # Rank parameter for LoRA (8 for NA ; 4 for Full)
alpha = 64  # Scaling parameter for LoRA (32 for NA ; 64 for Full)

# Create the LoRA layer
lora_layer = LoRALayer(hidden_size, rank, alpha)

# Wrap the model with the LoRA layer
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

lora_model = LoRAModel(model, lora_layer)

# Move the model to GPU and wrap it with DataParallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model = nn.DataParallel(lora_model)
lora_model.to(device)
lora_model.train()

# Define optimizer
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4,weight_decay=0.01)

# Define a warmup function and learning rate scheduler
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

# Total training steps (number of epochs * number of batches per epoch)
num_epochs = 10
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    lora_model.train()
    running_loss = 0.0
    train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training')
    for step, (images, labels) in enumerate(train_progress):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = lora_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        running_loss += loss.item()

        # Log training metrics to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch * len(train_loader) + step)

        train_progress.set_postfix(loss=running_loss/(step+1))

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Validation loop
    lora_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Validation')
    with torch.no_grad():
        for images, labels in val_progress:
            images, labels = images.to(device), labels.to(device)
            outputs = lora_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_progress.set_postfix(loss=val_loss/(total//16), accuracy=100 * correct/total)

    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {val_accuracy}%')

    # Log validation metrics to TensorBoard
    writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    # Save the model after each epoch
    save_model(epoch, lora_model, optimizer, scheduler, save_path)

# Close the TensorBoard writer
writer.close()
