import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
test_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=64, shuffle=False)

#train_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
#test_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)

device = "mps" if torch.has_mps else "cpu"


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
    def __init__(self, num_patches, embed_dim, max_len=256):
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

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio*embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio*embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multihead Attention
        residual = x
        x = self.attention(x, x, x)[0]
        x = self.dropout(x)
        x = x + residual
        x = self.norm1(x)

        # MLP
        residual = x
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embedding_size, num_layers, num_heads, dropout):
        super(VisionTransformer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        #self.positional_embedding = torch.randn(self.num_patches, embedding_size, device=device)
        self.positional_embedding = PositionalEncoding(self.num_patches, embedding_size)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, norm_first=True),
            num_layers=num_layers
        )

        '''self.transformer_encoder = nn.Sequential(*[
            EncoderBlock(embedding_size, num_heads, dropout=dropout) for _ in range(num_layers)
        ])'''

        self.classification_head = nn.Linear(embedding_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        x = PatchEmbedding(self.patch_size, self.patch_size, channels, self.embedding_size)(x)
        x = x + self.positional_embedding(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))
        
        x = x.mean(0)
        x = self.dropout(x)
        x = self.classification_head(x)
        return x


def plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training Loss of Absolute Positional Encoding Vision Tranformer on MNIST Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training Accuracy of Absolute Positional Encoding Vision Tranformer on MNIST Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    model = VisionTransformer(image_size=28, patch_size=4, num_classes=10, embedding_size=64, num_layers=4, num_heads=4, dropout=0.1).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    num_epochs = 10

    #training and validation loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

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

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            running_loss = 0.0
            for images, labels in tqdm(test_loader_mnist):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(test_loader_mnist)
        val_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    model.eval()
    correct = 0
    total = 0

    #testing loop
    with torch.no_grad():
        for images, labels in tqdm(test_loader_mnist):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies)
    


if __name__ == "__main__":
    main()