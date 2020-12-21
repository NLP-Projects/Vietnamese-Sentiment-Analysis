# PROGRAMMING ASSIGNMENT 4: SENTIMENT ANALYSIS FOR VIETNAMESE LANGUAGE
## Sentiment Analysis for Vietnamese Language
### 1. Introduction

With the development of technology and the Internet, different types of social media such as social networks and forums have allowed people to not only share information but also to express their opinions and attitudes on products, services and other social issues. The Internet becomes a very valuable and important source of information. People nowadays use it as a reference to make their decisions on buying a product or using a service. Moreover, this kind of information also let the manufacturers and service providers receive feedback about limitations of their products and therefore should improving them to meet the customer needs better. Furthermore, it can also help authorities know the attitudes and opinions of their residents on social events so that they can make appropriate adjustments.

Since early 2000s, opinion mining and sentiment analysis have become a new and active research topic in Natural language processing and Data mining. The major tasks in this topic can be listed as follows:

#### (+) Subjective classification: 

This is the task to detect that whether a document contains personal opinions or not (only provides facts).

#### (+) Polarity classification (Sentiment classification): 

The objective of this task is to classify the opinion of a document into one of three types, which are “positive”, “negative” and “neutral”.

#### (+) Spam detection: 

The goal of this task is to detect fake reviews and reviewers.

#### (+) Rating:

Rating the documents having personal opinions from 1 star to 5 star (very negative to very positive).

### Besides these common tasks, recently there are some other important tasks:

#### (+) Aspect-based sentiment analysis: 

The goal is to identify the aspects of given target entities and the sentiment expressed for each aspect.

#### (+) Opinion mining in comparative sentences: 

This task focuses on mining opinions from comparative sentences, i.e., to identify entities to be compared and determine which entities are preferred by the author in a comparative sentence.

### 2. Task Description

This task is polarity classification, i.e., to evaluate the ability of classifying Vietnamese reviews/documents into one of three categories: “positive”, “negative”, or “neutral”.  

### 3. Data for Training and Testing
A review can be very complex with different sentiments on various objects. Therefore, we set some constraints on the dataset as follows:

- The dataset only contains reviews having personal opinions.

- The data are usually short comments, containing opinions on one object. There is no limitation on the number of the object's aspects mentioned in the comment.

- Label (positive/negative/neutral) is the overall sentiment of the whole review.

- The dataset contains only real data collected from social media, not artificially created by human.

Note: Normally, it is very difficult to rate a neutral comment because the opinions are always inclinable to be negative or positive. A review is rated to be neutral when we cannot decide whether it is positive or negative. The neutral label can be used for the situations in which a review contains both positive and negative opinions but when combining them, the comment becomes neutral.

#### Some examples of the dataset:

##### Label : Review

Pos: Đẳng cấp Philips, máy đẹp, pin bền. Đóng và giao hàng rất chuyên nghiệp.

Pos: Tốt Giá vừa túi tiền đẹp và sang.

Pos: Rẻ hơn Samsung J1 nhưng cấu hình lại tốt hơn.

Pos: lướt web nhanh,chụp hình rõ nét, âm thanh ngoài trung bình rất xứng đáng với giá bán hiện giờ. pin đang trãi nghiệm (do mới sạc lần đầu).

 
Neg: Lâu lâu bị lỗi, màn hình cảm ứng không nhạy, chất lượng camera kém.

Neg: pin nhanh tụt, chỉ được xài 1 ngày.

Neg: Máy hay đơ màn hình, màn hình không nhạy dưới bên phải các phím M, N.

Neg: Mình trả cách đây gần 1 tháng rồi.


Neu: Pin khá hơn tí thì tốt nhỉ.

Neu: Đẹp thật, tiếc là ram và pin chưa ngon.

Neu: Vậy là không hỗ trợ thẻ nhớ. Một điểm hơi lăn tăn.

# Fundamentals of Deep Learning

http://perso.ens-lyon.fr/jacques.jayez/Cours/Implicite/Fundamentals_of_Deep_Learning.pdf

http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf


# Long Short-Term Memory with PyTorch
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/


# Sentiment analysis using LSTM - PyTorch

https://www.kaggle.com/arunmohan003/sentiment-analysis-using-lstm-pytorch

An example:

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


# References

https://viblo.asia/p/phan-tich-phan-hoi-khach-hang-hieu-qua-voi-machine-learningvietnamese-sentiment-analysis-Eb85opXOK2G

https://arxiv.org/ftp/arxiv/papers/1412/1412.8010.pdf

Thư mục Reference.

