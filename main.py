# main.py
import torch
from model import EmbeddingClassifierDecoder
from data.loader import load_cifar10, partition_dataset
from client import Client
from server import Server
from torch.utils.data import DataLoader
import torch.nn.functional as F

def evaluate_personalized_models(clients, test_data, device):
    total_correct, total_samples = 0, 0
    for client in clients:
        model = EmbeddingClassifierDecoder(num_classes=10).to(device)
        model.load_state_dict(client.personal_state)
        model.eval()

        loader = DataLoader(test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x, mode='classifier')
                pred_labels = torch.argmax(F.softmax(preds, dim=1), dim=1)
                correct += (pred_labels == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"[Client {client.id}] Personalized Accuracy: {acc*100:.2f}%")
        total_correct += correct
        total_samples += total

    avg_acc = total_correct / total_samples
    print(f"[Server] Avg Personalized Accuracy: {avg_acc*100:.2f}%")
    return avg_acc

def main():
    num_clients = 5
    num_clusters = 3
    rounds = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = load_cifar10()
    client_datasets = partition_dataset(train_data, num_clients, alpha=0.5)

    clients = [Client(i, data, device) for i, data in enumerate(client_datasets)]

    global_model = EmbeddingClassifierDecoder(num_classes=10)
    server = Server(global_model, clients, device, num_clusters)

    for r in range(rounds):
        print(f"\n--- Round {r+1} ---")
        client_weights = []
        for client in clients:
            model = EmbeddingClassifierDecoder(num_classes=10)
            if client.personal_state is not None:
                model.load_state_dict(client.personal_state)
            else:
                model.load_state_dict(global_model.state_dict())
            updated_weights = client.train(model)
            client_weights.append(updated_weights)

        personalized_models = server.aggregate(client_weights)
        for client, state in zip(clients, personalized_models):
            client.personal_state = state

        server.update_clusters()
        print(f"[Round {r+1}] Membership matrix (shape {server.membership.shape}):")
        print(server.membership)

        evaluate_personalized_models(clients, test_data, device)

    print("Training complete.")

if __name__ == "__main__":
    main()
