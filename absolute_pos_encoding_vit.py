import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from einops import rearrange
from einops.layers.torch import Rearrange


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
test_dataset_mnist = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=64, shuffle=False)

#train_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
#test_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)

device = "mps" if torch.has_mps else "cpu"

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
        x = nn.Dropout(0.1)(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_head=64):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(embedding_dim)

        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, embedding_dim)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, mlp_dim):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.norm = nn.LayerNorm(embedding_dim)
        self.encoder_block = nn.ModuleList([])
        for _ in range(num_layers):
            self.encoder_block.append(nn.ModuleList([
                self.attention,
                self.mlp
            ]))

    def forward(self, x):
        for attn, mlp in self.encoder_block:
            x = attn(x) + x
            x = mlp(x) + x
        x = self.norm(x)
        return x 

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embedding_dim, num_layers, num_heads, mlp_dim, channels, dropout):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_size = embedding_dim
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(image_size, patch_size, channels, embedding_dim)

        #self.positional_embedding = torch.randn(self.num_patches, embedding_dim, device=device)
        self.positional_embedding = PositionalEncoding(self.num_patches, embedding_dim)

        self.encoder_block = EncoderBlock(embedding_dim, num_heads, num_layers, mlp_dim)
        self.to_latent = nn.Identity()
        self.classification_head = nn.Linear(embedding_dim, num_classes)        

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.patch_embedding(x)
        # print("After patch embedding: ", x.shape)
        x = x + self.positional_embedding(x)
        # print("After positional embedding: ", x.shape)
        x = self.encoder_block(x)
        
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        x = self.classification_head(x)
        return x



def main():

    model = VisionTransformer(image_size=28, 
                              patch_size=4, 
                              num_classes=10, 
                              embedding_dim=128, 
                              num_layers=4, 
                              num_heads=4, 
                              mlp_dim=512,
                              channels=1,
                              dropout=0.2).to(device)
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    num_epochs = 100

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
            # print("Image shape: ", images.shape)
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