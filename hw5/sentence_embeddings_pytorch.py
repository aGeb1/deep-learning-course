from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dataset = load_dataset('ag_news')
pretrained_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

train_embeddings = pretrained_model.encode(dataset['train']['text'])
train_dataset = TensorDataset(torch.tensor(train_embeddings), torch.tensor(dataset['train']['label']))
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_embeddings = pretrained_model.encode(dataset['test']['text'])
test_dataset = TensorDataset(torch.tensor(test_embeddings), torch.tensor(dataset['test']['label']))
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = nn.Linear(384,4)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.02)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
