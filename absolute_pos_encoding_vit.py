import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Step 1: Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32))
])

train_dataset = torchvision.datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_w, num_patches_h)
        x = x.permute(0, 2, 3, 1)  # (batch_size, num_patches_w, num_patches_h, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_size, num_layers, num_heads, dropout):
        super(VisionTransformer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        self.hidden_size = hidden_size
        
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1, "Input must have only 1 channel (grayscale)"
        
        # Patch embedding
        x = PatchEmbedding(self.patch_size, self.patch_size, C, self.hidden_size)(x)
        #flatten the 2nd and 3rd dimensions of the tensor
        x = x.view(B, self.num_patches, self.hidden_size)
        x = x + self.positional_embedding[:, :self.num_patches]
        
        x = self.transformer_encoder(x.permute(1, 0, 2))  # Transpose for transformer input
        
        x = x.mean(0)
        x = self.dropout(x)
        x = self.classification_head(x)
        return x
    
model = VisionTransformer(image_size=32, patch_size=8, num_classes=10, hidden_size=128, num_layers=3, num_heads=8, dropout=0.1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 2

#training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

model.eval()
correct = 0
total = 0

#testing loop
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")