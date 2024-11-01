import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import wandb


wandb.init(
    project="FashionMNIST",
    config={"epochs": 5, "batch_size": 128, "learning_rate": 0.001},
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

id_to_labels = {
    0: "tshirt",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot",
}
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = FashionMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 5

model.train()
for epoch in range(n_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}")

    wandb.log({"training_loss": epoch_loss})

model.eval()
all_preds = []
all_labels = []
validation_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
validation_loss /= len(test_loader)

wandb.log({"validation_loss": validation_loss})

confusion_matrix = np.zeros((10, 10), dtype=int)
for true, pred in zip(all_labels, all_preds):
    confusion_matrix[true, pred] += 1
accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
for i, acc in enumerate(accuracies):
    print(f"Dokładność klasy {id_to_labels[i]}: {acc:.2f}")
    wandb.log({f"accuracy_{id_to_labels[i]}": acc})

artifact = wandb.Artifact('fashion_mnist_script', type='code')
artifact.add_file('fashion_mnist.py')
wandb.log_artifact(artifact)

num_images = 10
data_iter = iter(test_loader)
images, labels = next(data_iter)

wandb_images = []
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap="gray")
    plt.title(f"Label: {id_to_labels[labels[i].item()]}")
    plt.axis("off")
    wandb_images.append(wandb.Image(images[i].cpu(), caption=f"True: {id_to_labels[labels[i].item()]}, Pred: {id_to_labels[all_preds[i]]}"))

plt.show()
wandb.log({"examples": wandb_images})
wandb.finish()
