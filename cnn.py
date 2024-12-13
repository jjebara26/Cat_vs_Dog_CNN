import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

torch.manual_seed(42)

# preprocessing
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# read images from folder
dataset = datasets.ImageFolder(root='./petimages', transform=transform)


# splitting the data into training set and evaluation set 
# 20% of the dataset as test
generator = torch.Generator().manual_seed(42)
test_set, train_set = torch.utils.data.random_split(dataset, [int(0.2 * len(dataset)), int(0.8 * len(dataset))], generator=generator)

# preparing dataloader for training set and evaluation set
trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# model hyperparameters
learning_rate = 0.0001
batch_size = 32
epoch_size = 10



# model design
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether your device has GPU
cnn = CNN().to(device) # move the model to GPU

# search in official website for CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.0001
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)


# start model training
cnn.train() # turning on train mode
for epoch in range(epoch_size): # trying 10 epochs 

    loss = 0.0 # average loss

    for i, data in enumerate(trainloader, 0):
        # get the inputs and label from dataloader
        inputs, labels = data
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zeroing the parameter gradients
        optimizer.zero_grad()
        # forward -> compute loss -> backward propogation -> optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print some statistics
        loss += loss.item() # add loss for current batch
        if i % 100 == 99:    # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')


# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval() # turning on evaluation model
with torch.no_grad(): # not training, so we don't need to calculate the gradients for our outputs
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth += labels.tolist() # convert labels to list and append to ground_truth
        # calculating outputs by running inputs through the network
        outputs = cnn(inputs)
        # the class with the highest logit is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        prediction += predicted.tolist() # convert predicted to list and append to prediction

accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction)
precision = precision_score(ground_truth, prediction)






