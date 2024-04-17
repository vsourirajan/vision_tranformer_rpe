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

# train_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
# test_dataset_cifar10 = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)

# train_loader_cifar10 = DataLoader(train_dataset_cifar10, batch_size=64, shuffle=True)
# test_loader_cifar10 = DataLoader(test_dataset_cifar10, batch_size=64, shuffle=False)


device = "mps" if torch.has_mps else "cpu"

#function to plot loss and accuracy
def plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training Loss of Positional Encoding Vision Tranformer on MNIST Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training Accuracy of Positional Encoding Vision Tranformer on MNIST Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

#function to calculate pairwise euclidean distance between centers of all patches
def calculate_distance_matrix(num_patches):
    distance_matrix = torch.zeros(num_patches, num_patches)
    for patch1 in range(num_patches):
        row1, col1 = divmod(patch1, math.sqrt(num_patches))
        center1 = (col1 + 0.5, row1 + 0.5)
        for patch2 in range(num_patches):
            row2, col2 = divmod(patch2, math.sqrt(num_patches))
            center2 = (col2 + 0.5, row2 + 0.5)
            distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            distance_matrix[patch1][patch2] = distance
    #shape [num_patches, num_patches] where distance_matrix[i,j] is euclidean distance between centers of patch i and patch j
    return distance_matrix


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

#general learnable function where weights are shared across all attention heads
class GeneralLearnableFunctionParallel(nn.Module):
    def __init__(self, n, d):
        super(GeneralLearnableFunctionParallel, self).__init__()
        self.n = n
        self.d = d
        self.embeddings = nn.Linear(1, d)

    def forward(self, distance_matrix):
        distance_matrix = distance_matrix.unsqueeze(-1)
        embeddings = self.embeddings(distance_matrix)
        embeddings = embeddings.view(self.n, self.n, self.d)
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, distance_matrix):
        super().__init__()
        self.dim_head = hidden_dim//num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc_q = nn.Linear(embedding_dim, self.hidden_dim)
        self.fc_k = nn.Linear(embedding_dim, self.hidden_dim)
        self.fc_v = nn.Linear(embedding_dim, self.hidden_dim)

        self.to_out = nn.Linear(hidden_dim, embedding_dim)

        self.relative_k = GeneralLearnableFunctionParallel(self.dim_head, distance_matrix)
        self.relative_v = GeneralLearnableFunctionParallel(self.dim_head, distance_matrix)
        #self.relative_k, self.relative_v = [num_patches, num_patches, dim_head]

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        #x = [batch_size, num_patches, embedding_dim]
        
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        #q,k,v = [batch_size, num_patches, inner_dim]

        q = q.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        k = k.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        #q,k = [batch_size, num_heads, num_patches, dim_head]

        QKT = torch.matmul(q, k.permute(0, 1, 3, 2))
        #QKT = [batch_size, num_heads, num_patches, num_patches]

        #apply softmax and scale
        attn1 = self.softmax(QKT) * self.scale
        #attn1 = [batch_size, num_heads, num_patches, num_patches]

        #obtain relative positional embeddings
        relative_k = self.relative_k(x)
        relative_v = self.relative_v(x)
        #relative_k, relative_v = [num_patches, embedding_dim]

        return None

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
    def __init__(self, embedding_dim, num_heads, num_layers, mlp_dim, distance_matrix):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, distance_matrix)
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
            x = attn(x, x, x) + x
            x = mlp(x) + x
        x = self.norm(x)
        return x 

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embedding_dim, num_layers, num_heads, mlp_dim, channels, dropout):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_size = embedding_dim
        self.patch_size = patch_size

        self.distance_matrix = calculate_distance_matrix(self.num_patches)

        self.patch_embedding = PatchEmbedding(image_size, patch_size, channels, embedding_dim)
        self.encoder_block = EncoderBlock(embedding_dim, num_heads, num_layers, mlp_dim, self.distance_matrix)
        self.to_latent = nn.Identity()
        self.classification_head = nn.Linear(embedding_dim, num_classes)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.patch_embedding(x)
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