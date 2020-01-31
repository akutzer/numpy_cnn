import numpy as np
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style

from CNN.modules import *

np.random.seed(2)
style.use("seaborn-darkgrid")


### LOAD DATA ###
train_f = gzip.GzipFile("FashionMNIST/fashion-mnist_train.npy.gz", "r")
train_data = np.load(train_f)
train_f.close()
test_f = gzip.GzipFile("FashionMNIST/fashion-mnist_test.npy.gz", "r")
test_data = np.load(test_f)
test_f.close()
np.random.shuffle(train_data), np.random.shuffle(test_data)
X_train, y_train = train_data[:,1:].reshape(-1, 1, 28, 28), train_data[:,0]
X_test, y_test = test_data[:,1:].reshape(-1, 1, 28, 28), test_data[:,0]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


### CREATE MODEL-CLASS ###
class Net():
    def __init__(self):
        # INPUT : (1, 28, 28)
        self.conv1 = Conv(1, 32, 3, padding=1)
        pool1 = MaxPool(2, 2)
        relu1 = LeakyReLU(1e-4)
        # SHAPE : (32, 14, 14)
        self.conv2 = Conv(32, 32, 3, padding=1)
        pool2 = MaxPool(2, 2)
        relu2 = LeakyReLU(1e-4)
        # SHAPE : (32, 7, 7)
        self.linear3 = Linear(32*7*7, 128)
        relu3 = LeakyReLU(1e-4)
        self.linear4 = Linear(128, 10)
        softmax4 = Softmax(axis=1)

        self.model = (self.conv1, pool1, relu1,
                      self.conv2, pool2, relu2,
                      self.linear3, relu3,
                      self.linear4, softmax4)

        self.criterion = CrossEntropyLoss()

    def forward(self, x, y=None):
        for layer in self.model:
            x = layer(x)
        if isinstance(y, np.ndarray):
            loss = self.criterion(x, y)
            return x, loss
        else:
            return x

    def backward(self):
        grad = self.criterion.gradient()
        for layer in reversed(self.model):
            grad = layer.gradient(grad)
        return grad

    def parameters(self):
        layers = []
        for layer in self.model:
            if hasattr(layer, "weight"):
                layers.append(layer)
        return layers


### DEFINE TRAIN FUNCTION ###
def train(X_batch, y_batch, model, optimizer=None, validate=False):
    x, loss = model.forward(X_batch, y_batch)
    if not validate:
        grad = model.backward()
        optimizer.update()

    correct = [1 if x==y else 0 for x, y in zip(np.argmax(x, axis=1), y_batch)]
    acc = sum(correct)/X_batch.shape[0]

    return loss, acc


### DEFINE TRAININGS LOOP ###
def train_loop(X_train, y_train, X_test, y_test,
               model, optimizer, epochs, batch_size):

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    ### LIVE PLOTTING ###
    fig, axs = plt.subplots(1, 2, figsize=(8,4), sharex=True)
    fig.tight_layout()

    def plot_acc_loss(j):
        axs[0].clear(), axs[1].clear()
        axs[0].plot(train_losses, label="Train Loss")
        axs[0].plot(test_losses, label="Test Loss")
        axs[0].legend()
        axs[1].plot(train_accs, label="Train Accs")
        axs[1].plot(test_accs, label="Test Accs")
        axs[1].legend()

    ani = FuncAnimation(fig, plot_acc_loss, interval=1000)

    for e in range(epochs):
        print(f"\n\t####### EPOCH : {e+1} #######")
        for batch in tqdm(range(0, X_train.shape[0], batch_size)):
            X_train_batch = X_train[batch:batch + batch_size]
            y_train_batch = y_train[batch:batch + batch_size]
            X_train_batch = (X_train_batch/255-.5)/.5 # normalize between [-1, 1]

            train_loss, train_acc = train(X_train_batch, y_train_batch, \
                                          model, optimizer)

            test_idx = np.random.randint(0, X_test.shape[0]-batch_size)

            X_test_batch = X_test[test_idx:test_idx + 4*batch_size]
            y_test_batch = y_test[test_idx:test_idx + 4*batch_size]
            X_test_batch = (X_test_batch/255-.5)/.5 # normalize between [-1, 1]

            test_loss, test_acc = train(X_test_batch, y_test_batch,\
                                        model, validate=True)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            plt.pause(.001)

    return model, train_losses, train_accs, test_losses, test_accs


EPOCHS, BATCH_SIZE, LEARNING_RATE = 1, 128, 1e-4
MODEL = Net()
OPTIMIZER = Adam(MODEL.parameters(), LEARNING_RATE)

model, train_losses, train_accs, test_losses, test_accs =\
        train_loop(X_train, y_train, X_test, y_test,
                   MODEL, OPTIMIZER, EPOCHS, BATCH_SIZE)


print(f"Train Loss: {sum(train_losses[-10:])/10}\tTrain Accuracy: {sum(train_accs[-10:])/10}\
\nTest Loss: {sum(test_losses[-10:])/10}\tTrain Accuracy: {sum(test_accs[-10:])/10}")
plt.show()
