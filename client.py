import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EmbeddingClassifierDecoder

class Client:
    def __init__(self, client_id, train_data, device):
        self.id = client_id
        self.train_data = train_data
        self.device = device
        self.personal_state = None  # for personalized model

    def train(self, model, epochs=5, batch_size=32, lr=0.01):
        model = model.to(self.device)
        model.train()
        loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x, mode='classifier')
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

        return model.cpu().state_dict()

    def compute_embedding(self, model):
        model.eval()
        model = model.to(self.device)
        sample = next(iter(DataLoader(self.train_data, batch_size=1)))[0].to(self.device)
        with torch.no_grad():
            emb = model(sample, mode='embed')[0].cpu()
        return emb
