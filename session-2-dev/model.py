import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, num_units):
        super(MyModel, self).__init__()
        # kernel_size: convolutional kernel or filter size: 3x3
        # stride: A stride of 1 means the kernel moves one pixel at a time
        # padding: A padding of 1 means that a 1-pixel border is added to all sides of the input.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # does not change the spatial dimensions, so the input remains 64x64x16.
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # This operation effectively reduces the spatial dimensions by a factor of 2
        # 32x32x16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # the input size for the third nn.Conv2d layer after these operations will be 16x16x32.
        # Input size from the previous layer: 16x16x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input size from the previous layer: 8x8x64

        # OK self.fc1 = nn.Linear(64 * 4 * 4 * 4, num_units)
        self.fc1 = nn.Linear(64 * 8 * 8, num_units)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(num_units, num_classes)

        # self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        # output = self.log_softmax(x)
        return x



