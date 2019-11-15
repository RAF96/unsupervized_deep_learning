import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

def sample_data():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))

data = sample_data()
train_and_validate, test = train_test_split(data, test_size=0.2)
train, validate = train_test_split(train_and_validate, test_size=0.1)
theta = torch.zeros(100, requires_grad=True)

def compute_loss(theta, batch):
    vec = torch.exp(theta) / torch.sum(torch.exp(theta))
    probs = torch.gather(vec, dim=0, index=batch)
    loss = torch.sum(torch.log(probs) * -1) / probs.shape[0]
    return loss

NUM_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 1024

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

optimizer = torch.optim.Adam([theta], lr=LR)
train_losses = []
validate_losses = []

for epoch in range(NUM_EPOCHS):
    for idx, train_batch in enumerate(train_loader):
        loss = compute_loss(theta, train_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    val_loss = compute_loss(theta, torch.tensor(validate))
    validate_losses.append(val_loss.item())


test_loss = compute_loss(theta, torch.tensor(test))
print(f"Final train loss: {train_losses[-1]}")
print(f"Final valid loss: {validate_losses[-1]}")
print(f"Final test loss: {test_loss}")

plt.plot(train_losses, label="train")
plt.plot(np.linspace(0, len(train_losses), len(validate_losses)), validate_losses, label="validate")
plt.title("Negative log likelihood")
plt.show()


plt.hist(data, range(100))
plt.title("Histogram of data")
plt.show()

vec = torch.exp(theta) / torch.sum(torch.exp(theta))
plt.plot(vec.detach().numpy())
plt.title("Probability of the model")
plt.show()