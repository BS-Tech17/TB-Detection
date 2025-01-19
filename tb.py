import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.metrics import confusion_matrix

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Paths to the training and validation directories
train_dir = "xyz/test"
val_dir = "xyz/val"

# Define the transformations for the training and validation sets
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# Load data into DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# Modify the classifier layer to match binary classification (TB vs Normal)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_labels = []
        train_predictions = []

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        train_cm = confusion_matrix(train_labels, train_predictions)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print("Training Confusion Matrix:")
        print(train_cm)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for val_images, val_labels_batch in val_loader:
                val_images, val_labels_batch = val_images.to(device), val_labels_batch.to(device).float().unsqueeze(1)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels_batch).item()
                val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                val_correct += (val_predicted == val_labels_batch).sum().item()
                val_total += val_labels_batch.size(0)

                val_labels.extend(val_labels_batch.cpu().numpy())
                val_predictions.extend(val_predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total * 100
        val_cm = confusion_matrix(val_labels, val_predictions)
        print(f'Validation Loss: {val_loss:.4f}')
        print("Validation Confusion Matrix:")
        print(val_cm)

    return model

# Train and evaluate the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Save the model
torch.save(model.state_dict(), "tb_detection_pytorch_model.pth")
print("Model saved as 'tb_detection_pytorch_model.pth'")

# Load the trained model for inference
def load_model_for_inference(model_path):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Define the transformations for a single input image (same as training)
input_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict tuberculosis from an input image
def predict_tb(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = input_transforms(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    if prediction > 0.5:
        return "Tuberculosis Detected"
    else:
        return "Normal X-ray"

# Run the prediction in a loop until the user exits
def run_inference_loop(model):
    while True:
        image_path = input("Enter the path of the X-ray image (or type 'exit' to quit): ")
        if image_path.lower() == 'exit':
            print("Exiting the program.")
            break

        if not os.path.exists(image_path):
            print("Image file does not exist. Please try again.")
            continue

        result = predict_tb(image_path, model)
        print(f"Prediction: {result}")

# Load the model for inference
inference_model = load_model_for_inference("tb_detection_pytorch_model.pth")

# Start the inference loop
run_inference_loop(inference_model)
