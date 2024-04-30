import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import sys
from attention import MultiHeadAttentionParallel, MultiHeadAttentionIndividual
from graph import plot_loss_accuracy

device = "mps" if torch.has_mps else "cpu"

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
    def __init__(self, embedding_dim, num_heads, num_layers, mlp_dim, distance_matrix, rpe_type, subvariant):
        super(EncoderBlock, self).__init__()
        if subvariant == "1":
            self.attention = MultiHeadAttentionParallel(embedding_dim, num_heads, embedding_dim, distance_matrix, rpe_type)
        else:
            self.attention = MultiHeadAttentionIndividual(embedding_dim, num_heads, embedding_dim, distance_matrix, rpe_type)
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
    def __init__(self, image_size, patch_size, num_classes, embedding_dim, num_layers, num_heads, mlp_dim, channels, dropout, rpe_method, subvariant):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_size = embedding_dim
        self.patch_size = patch_size

        self.distance_matrix = calculate_distance_matrix(self.num_patches).to(device)

        self.patch_embedding = PatchEmbedding(image_size, patch_size, channels, embedding_dim)
        self.encoder_block = EncoderBlock(embedding_dim, num_heads, num_layers, mlp_dim, self.distance_matrix, rpe_method, subvariant)
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

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = sys.argv[1]
    rpe_method = sys.argv[2]
    subvariant = sys.argv[3]

    if dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root='./data_mnist', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data_mnist', train=False, transform=transform, download=True)

        model = VisionTransformer(image_size=28, 
                                patch_size=4, 
                                num_classes=10, 
                                embedding_dim=128, 
                                num_layers=6, 
                                num_heads=4,  
                                mlp_dim=512,
                                channels=1,
                                dropout=0.2,
                                rpe_method=rpe_method,
                                subvariant=subvariant).to(device)
    else:
        train_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, transform=transform, download=True)
        model = VisionTransformer(image_size=32, 
                            patch_size=4, 
                            num_classes=10, 
                            embedding_dim=128, 
                            num_layers=6, 
                            num_heads=4,  
                            mlp_dim=512,
                            channels=3,
                            dropout=0.2,
                            rpe_method=rpe_method,
                            subvariant=subvariant).to(device)
    '''
    # train_dataset = torchvision.datasets.MNIST(root='./data_mnist', train=True, transform=transform, download=True)
    # test_dataset = torchvision.datasets.MNIST(root='./data_mnist', train=False, transform=transform, download=True)
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VisionTransformer(image_size=32, 
                            patch_size=4, 
                            num_classes=10, 
                            embedding_dim=128, 
                            num_layers=6, 
                            num_heads=4,  
                            mlp_dim=512,
                            channels=3,
                            dropout=0.2).to(device)
    '''

    print(model)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    num_epochs = 1

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
        for images, labels in tqdm(train_loader):

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
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            running_loss = 0.0
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(test_loader)
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
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, dataset, rpe_method, subvariant)
    

if __name__ == "__main__":
    main()