import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class Server:
    def __init__(self, global_model, clients, device, num_clusters=3):
        self.model = global_model
        self.clients = clients
        self.device = device
        self.num_clusters = num_clusters
        self.cluster_embeddings = None
        self.membership = None

    def aggregate(self, client_weights):
        N, K = len(self.clients), self.num_clusters
        W = self.membership if self.membership is not None else np.ones((N, K)) / K

        cluster_weights = []
        for k in range(K):
            avg = {}
            denom = W[:, k].sum()
            for name in client_weights[0]:
                agg = sum(W[i, k] * client_weights[i][name] for i in range(N))
                avg[name] = agg / denom
            cluster_weights.append(avg)

        global_avg = {}
        for name in cluster_weights[0]:
            global_avg[name] = sum(cluster_weights[k][name] for k in range(K)) / K
        self.model.load_state_dict(global_avg)

        personalized = []
        for i in range(N):
            state = {}
            for name in global_avg:
                w = sum(W[i, k] * cluster_weights[k][name] for k in range(K))
                state[name] = w
            personalized.append(state)

        return personalized

    def evaluate(self, test_data, batch_size=64):
        self.model.to(self.device).eval()
        loader = DataLoader(test_data, batch_size=batch_size)
        correct = total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x, mode='classifier')
                pred_labels = torch.argmax(F.softmax(preds, dim=1), dim=1)
                correct += (pred_labels == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        print(f"[Server] Global Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def update_clusters(self):
        # Collect embeddings from clients
        embeddings = []
        for client in self.clients:
            emb = client.compute_embedding(self.model.to(client.device))
            embeddings.append(emb.numpy())
        embeddings = np.stack(embeddings)

        if self.cluster_embeddings is None:
            # Initialize using KMeans++ at first round
            kmeans = KMeans(n_clusters=self.num_clusters, init="k-means++")
            kmeans.fit(embeddings)
            self.cluster_embeddings = kmeans.cluster_centers_
        
        # Compute membership weights (soft attention)
        sim = self._cosine_similarity(embeddings, self.cluster_embeddings)
        soft = self._softmax(sim)
        self.membership = soft

        # Recompute cluster centers
        self.cluster_embeddings = (soft.T @ embeddings) / soft.sum(axis=0)[:, None]


    def _cosine_similarity(self, A, B):
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        return A @ B.T

    def _softmax(self, sim, temp=0.5):
        exp_sim = np.exp(sim / temp)
        return exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
