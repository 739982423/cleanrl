import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1.生成数据集

def get_input(load_data_length=5000):
    input_stream = []
    with open("./tweet_load.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            line += 1
            if int(row[1]) > 15000:
                row[1] = 15000
            if line - 2 == load_data_length:
                break
            input_stream.append([int(row[1])])

    return input_stream

def create_data_seq(data_raw, seq):
    data_feat,data_target = [], []
    for index in range(len(data_raw) - seq - 1):
        # 构建特征集
        data_feat.append(data_raw[index: index + seq])
        # 构建target集
        data_target.append(data_raw[1 + index: 1 + index + seq])
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    return data_feat, data_target


def train_test(data_feat, data_target, validate_size, train_size, seq):
    train_size = int(len(data_feat) * train_size)
    validate_size = int(len(data_feat) * validate_size)

    valX = torch.from_numpy(data_feat[:validate_size].reshape(-1, seq, 1)).type(torch.Tensor)
    trainX = torch.from_numpy(data_feat[validate_size: train_size + validate_size].reshape(-1, seq, 1)).type(torch.Tensor)
    testX  = torch.from_numpy(data_feat[train_size + validate_size:].reshape(-1, seq, 1)).type(torch.Tensor)

    valY = torch.from_numpy(data_target[:validate_size].reshape(-1, seq, 1)).type(torch.Tensor)
    trainY = torch.from_numpy(data_feat[validate_size: train_size + validate_size].reshape(-1, seq, 1)).type(torch.Tensor)
    testY  = torch.from_numpy(data_target[train_size + validate_size:].reshape(-1, seq, 1)).type(torch.Tensor)
    return trainX, trainY, valX, valY, testX, testY

data = torch.Tensor(get_input())

seq = 10
scaler = MinMaxScaler(feature_range=(-1., 1.))
normalized_data = scaler.fit_transform(data)
print(normalized_data.shape)
normalized_data = normalized_data.reshape(-1)
print(normalized_data.shape)
data_feat, data_target = create_data_seq(normalized_data, seq)
print(data_feat.shape, data_target.shape)
trainX, trainY, valX, valY, testX, testY = train_test(data_feat, data_target, seq = seq, validate_size = 0.1, train_size = 0.6)
print("train shape", trainX.shape, trainY.shape)
print("validate shape", valX.shape, valY.shape)
print("test shape", testX.shape, testY.shape)

batch_size = 10
num_epochs = 150

train = torch.utils.data.TensorDataset(trainX, trainY)
test = torch.utils.data.TensorDataset(testX, testY)
train_loader = torch.utils.data.DataLoader(dataset = train, 
                                           batch_size = batch_size, 
                                           shuffle = False)

test_loader = torch.utils.data.DataLoader(dataset = test, 
                                          batch_size = batch_size, 
                                          shuffle = False)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out) 
        return out   



# 5. 定义超参数

input_dim = 1
hidden_dim = 64
num_layers = 2 
output_dim = 1

for loop in range(1):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    print(model)

    loss_fn = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    hist_trainset = np.zeros(num_epochs)
    hist_valset = np.zeros(num_epochs)
    seq_dim = seq

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()
        
        # Forward pass
        y_train_pred = model(trainX)

        loss = loss_fn(y_train_pred, trainY)
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist_trainset[t] = loss.item()

        y_val_pred = model(valX)
        loss_validate = loss_fn(y_val_pred, valY)

        hist_valset[t] = loss_validate.item()
        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()




    # make predictions
    y_test_pred = model(testX)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy()[:,-1,0].reshape(-1,1))
    y_train = scaler.inverse_transform(trainY.detach().numpy()[:,-1,0].reshape(-1,1))
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy()[:,-1,0].reshape(-1,1))
    y_test = scaler.inverse_transform(testY.detach().numpy()[:,-1,0].reshape(-1,1))




    # 保存loss曲线数据
    with open("train_val_loss.csv", mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "epoch", "loss", "type"])
        for epoch, data in enumerate(hist_trainset):
            writer.writerow([epoch, data, "train"])
        for epoch, data in enumerate(hist_valset):
            writer.writerow([epoch, data, "validate"])
        



# # loss变化曲线
# plt.figure()
# plt.plot([i for i in range(len(hist_trainset))], hist_trainset, 'o-.',label = "train set", markerfacecolor="white")
# plt.plot([i for i in range(len(hist_valset))], hist_valset, '-.',label = "validate set")
# plt.legend()
# plt.show()

# # 将训练集和测试集画在一张图上：
show_length = 300
plt.figure()
plt.plot([i for i in range(len(y_train_pred))], y_train_pred, c="orange", label="train_predict")
plt.plot([i for i in range(len(y_train))], y_train, c="b", alpha=0.25, label="train_real")

plt.plot([i for i in range(len(y_train_pred) + len(y_test_pred) - show_length, len(y_train_pred) + len(y_test_pred))], y_test_pred[-show_length:], c="g", label="test_predict")
plt.plot([i for i in range(len(y_train_pred) + len(y_test_pred) - show_length, len(y_train_pred) + len(y_test_pred))], y_test[-show_length:], c="b", alpha=0.25, label="test_real")
plt.legend()
plt.show()

# # 将训练集和测试集画在两张图上：
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot([i for i in range(len(y_train_pred))], y_train_pred, label="train_predict")
# plt.plot([i for i in range(len(y_train))], y_train, label="train_real")
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot([i for i in range(len(y_test_pred))], y_test_pred, label="test_predict")
# plt.plot([i for i in range(len(y_test))], y_test, label="test_real")
# plt.legend()
# plt.show()
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(y_train, y_train_pred))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(y_test, y_test_pred))
# print('Test Score: %.2f RMSE' % (testScore))