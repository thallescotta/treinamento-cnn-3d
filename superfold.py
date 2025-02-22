import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import json
import time
from datetime import datetime
from torchvision.models.video import r3d_18, R3D_18_Weights
import logging
from imblearn.over_sampling import RandomOverSampler

# -----------------------------------------------------------------------------
# Configuração do Logging
# -----------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_log_{timestamp}.txt"
results_filename = f"/home/thalles.fontainha/3dCnnOtimizada/final_results_{timestamp}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
def log_message(message):
    logging.info(message)

# -----------------------------------------------------------------------------
# FUNÇÃO CRIA MODELO
# -----------------------------------------------------------------------------
def create_model():
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(device)

# -----------------------------------------------------------------------------
# CLASSE DO DATASET
# -----------------------------------------------------------------------------
class KneeMRIDataset(torch.utils.data.Dataset):
    def __init__(self, normal_path, abnormal_path, transform=None):
        self.normal_data = np.load(normal_path, mmap_mode="r")
        self.abnormal_data = np.load(abnormal_path, mmap_mode="r")
        self.transform = transform
        self.total_normal = len(self.normal_data)
        self.total_abnormal = len(self.abnormal_data)
        self.labels = np.concatenate((np.zeros(self.total_normal), np.ones(self.total_abnormal)))
        
    def __len__(self):
        return self.total_normal + self.total_abnormal

    def __getitem__(self, idx):
        if idx < self.total_normal:
            volume = self.normal_data[idx]
            label = 0
        else:
            volume = self.abnormal_data[idx - self.total_normal]
            label = 1
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            volume = self.transform(volume)
        label = torch.tensor(label, dtype=torch.float32)
        return volume, label

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/thalles.fontainha/3dCnnOtimizada/best_model_fold_2.pth"
batch_size = 8
learning_rate = 0.0001
max_epochs = 1000
early_stop_patience = 40

normal_path = "/home/thalles.fontainha/dataset/OAI-MRI-3DDESS/normal-3DESS-128-64.npy"
abnormal_path = "/home/thalles.fontainha/dataset/OAI-MRI-3DDESS/abnormal-3DESS-128-64.npy"
transform = None

dataset = KneeMRIDataset(normal_path, abnormal_path, transform=transform)
log_message(f"Dataset carregado com {len(dataset)} amostras.")

# -----------------------------------------------------------------------------
# DIVISÃO: 85% SUPER FOLD (TREINO+VALIDAÇÃO), 15% TESTE
# -----------------------------------------------------------------------------
total_size = len(dataset)
super_fold_size = int(0.85 * total_size)
test_size = total_size - super_fold_size

super_fold_dataset, test_dataset = random_split(dataset, [super_fold_size, test_size])
log_message(f"Super Fold: {super_fold_size}, Teste: {test_size}")

# Oversampling no Super Fold
ros = RandomOverSampler(random_state=42)
super_fold_indices = np.arange(len(super_fold_dataset)).reshape(-1, 1)
super_fold_labels = [super_fold_dataset[i][1].item() for i in range(len(super_fold_dataset))]
resampled_indices, _ = ros.fit_resample(super_fold_indices, super_fold_labels)
resampled_indices = resampled_indices.flatten()

sampler = SubsetRandomSampler(resampled_indices)
super_fold_loader = DataLoader(super_fold_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -----------------------------------------------------------------------------
# Carregar o modelo com parâmetros do Fold 2
# -----------------------------------------------------------------------------
model = create_model()
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
log_message(f"Modelo carregado a partir de: {model_path}")

# -----------------------------------------------------------------------------
# Configuração do treinamento
# -----------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Early stopping config
best_loss = float('inf')
early_stop_counter = 0

log_message("Iniciando treinamento no Super Fold com Early Stopping...")

# Histórico para salvar métricas
history = []

# Treinamento com avaliação no conjunto de teste por época
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in super_fold_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.repeat(1, 3, 1, 1, 1)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(super_fold_loader)

    # Avaliação no teste
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.repeat(1, 3, 1, 1, 1)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    log_message(f"Época {epoch+1} | Loss: {avg_loss:.4f} | Acurácia: {accuracy*100:.2f}% | AUC: {auc:.4f} | F1: {f1:.4f}")

    history.append({
        "epoch": epoch + 1,
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc_roc": auc,
        "f1_score": f1
    })

    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model_superfold.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            log_message(f"Early stopping na época {epoch+1}")
            break

with open(results_filename, "w") as f:
    json.dump(history, f, indent=4)

log_message(f"Treinamento concluído! Resultados salvos em {results_filename}")
