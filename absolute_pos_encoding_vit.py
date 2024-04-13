import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32))
])

train_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
test_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=64, shuffle=False)

train_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
test_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        #self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, device=device)

    def forward(self, x):
        x = self.projection(x)  #(batch_size, embed_dim, num_patches_w, num_patches_h)
        x = x.permute(0, 2, 3, 1)  #(batch_size, num_patches_w, num_patches_h, embed_dim)
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3)) #(batch_size, num_patches, embed_dim)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim, max_len=512):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        #self.pos_embedding = torch.zeros(max_len, 1, embed_dim)
        self.pos_embedding = torch.zeros(max_len, 1, embed_dim, device=device)
        self.pos_embedding[:, 0, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pos_embedding[:x.size(0)]
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embedding_size, num_layers, num_heads, dropout):
        super(VisionTransformer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_size = embedding_size
        
        self.positional_embedding = PositionalEncoding(self.num_patches, embedding_size)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, norm_first=True),
            num_layers=num_layers
        )
        self.classification_head = nn.Linear(embedding_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = PatchEmbedding(self.patch_size, self.patch_size, channels, self.embedding_size)(x)
        x = x + self.positional_embedding(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))
        
        x = x.mean(0)
        x = self.dropout(x)
        x = self.classification_head(x)
        return x

#set device as mps
device = "mps" if torch.has_mps else "cpu"

model = VisionTransformer(image_size=32, patch_size=4, num_classes=10, embedding_size=128, num_layers=3, num_heads=8, dropout=0.1).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 100

#training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader_mnist):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader_mnist)
    train_acc = correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

model.eval()
correct = 0
total = 0

#testing loop
with torch.no_grad():
    for images, labels in tqdm(test_loader_mnist):
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")