import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
import os
from datetime import datetime
import logging
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler

# Importando r3d_18 e seus pesos "DEFAULT"
from torchvision.models.video import r3d_18, R3D_18_Weights

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
NUM_EPOCHS = 1000           # Número total de épocas para treinamento
EARLY_STOP_PATIENCE = 40    # Paciência para Early Stopping (épocas sem melhoria) - visto que 100 é muito e demora d+
BATCH_SIZE = 8              # Tamanho do batch
NUM_WORKERS = 4             # Número de workers para DataLoader
N_SPLITS = 5                # Número de folds para Cross-Validation
LEARNING_RATE = 0.0001      # Taxa de aprendizado
WEIGHT_DECAY = 1e-5         # Decaimento de peso para regularização
POS_WEIGHT_VALUE = 2.0      # Peso para a classe positiva (para lidar com desbalanceamento)

# =============================================================================
# CLASSES DE DATA AUGMENTATION 3D
# =============================================================================
class Compose3D:
    """Encadeia várias transformações 3D."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomFlip3D:
    """Flip aleatório em dimensões 3D (por ex., altura e largura)."""
    def __init__(self, p=0.5, dims=(2, 3)):
        self.p = p
        self.dims = dims  # Ex.: flip nas dimensões "altura" (2) e "largura" (3)

    def __call__(self, volume):
        for dim in self.dims:
            if np.random.rand() < self.p:
                volume = torch.flip(volume, dims=[dim])
        return volume

class RandomResizedCrop3D:
    """
    Faz um crop 3D aleatório e depois redimensiona para o tamanho alvo.
    size: (D, H, W) final
    scale: fator de escala aleatório (0.8 a 1.0, por exemplo)
    """
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, volume):
        # volume: (C, D, H, W)
        c, d, h, w = volume.shape
        scale_factor = np.random.uniform(*self.scale)
        new_d = int(d * scale_factor)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        # Garantir que não ultrapasse o tamanho original
        new_d = min(new_d, d)
        new_h = min(new_h, h)
        new_w = min(new_w, w)
        # Escolher ponto de início aleatoriamente
        start_d = np.random.randint(0, d - new_d + 1) if d - new_d > 0 else 0
        start_h = np.random.randint(0, h - new_h + 1) if h - new_h > 0 else 0
        start_w = np.random.randint(0, w - new_w + 1) if w - new_w > 0 else 0

        cropped = volume[:, start_d:start_d+new_d, start_h:start_h+new_h, start_w:start_w+new_w]
        # Redimensionar para o tamanho alvo usando interpolação trilinear
        cropped = cropped.unsqueeze(0)  # Adicionar dimensão de batch
        resized = nn.functional.interpolate(cropped, size=self.size, mode='trilinear', align_corners=False)
        return resized.squeeze(0)

# =============================================================================
# FOCAL LOSS PARA CLASSIFICAÇÃO BINÁRIA
# =============================================================================
class FocalLoss(nn.Module):
    """
    Implementa a Focal Loss para lidar com desbalanceamento de classes.
    alpha: fator de balanceamento
    gamma: fator de foco (controla quanto penaliza erros de classes raras)
    pos_weight: usado internamente no BCEWithLogitsLoss
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # inputs são logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)  # probabilidade estimada
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
start_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = f"training_progress_{start_time_str}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_message(message):
    print(message)
    logging.info(message)

# =============================================================================
# CONFIGURAÇÃO DO DISPOSITIVO
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_count = torch.cuda.device_count()
    log_message(f"Número de GPUs detectadas: {gpu_count}")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        log_message(f"GPU {i + 1}: {gpu_name} com {gpu_memory:.2f} GB de memória")

# =============================================================================
# CLASSE DO DATASET (CARREGA NORMAL E ANORMAL)
# =============================================================================
class KneeMRIDataset(Dataset):
    """
    Carrega dois .npy:
    - normal_data -> label 0
    - abnormal_data -> label 1
    """
    def __init__(self, normal_path, abnormal_path, transform=None):
        self.normal_data = np.load(normal_path, mmap_mode="r")
        self.abnormal_data = np.load(abnormal_path, mmap_mode="r")
        self.total_normal = len(self.normal_data)
        self.total_abnormal = len(self.abnormal_data)
        self.labels = np.concatenate((np.zeros(self.total_normal), np.ones(self.total_abnormal)))
        self.transform = transform

    def __len__(self):
        return self.total_normal + self.total_abnormal

    def __getitem__(self, idx):
        if idx < self.total_normal:
            volume = self.normal_data[idx]
            label = 0
        else:
            volume = self.abnormal_data[idx - self.total_normal]
            label = 1

        # Normalização por volume individual
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)

        if self.transform:
            volume = self.transform(volume)

        label = torch.tensor(label, dtype=torch.float32)
        return volume, label

# =============================================================================
# TRANSFORMAÇÕES 3D PARA DATA AUGMENTATION
# =============================================================================
transform = Compose3D([
    RandomFlip3D(p=0.5, dims=(2, 3)),  # Flip aleatório nas dimensões "altura" e "largura"
    RandomResizedCrop3D(size=(128, 128, 64), scale=(0.8, 1.0))
])

# =============================================================================
# CARREGAMENTO DO DATASET
# =============================================================================
normal_path = "/home/thalles.fontainha/dataset/OAI-MRI-3DDESS/normal-3DESS-128-64.npy"
abnormal_path = "/home/thalles.fontainha/dataset/OAI-MRI-3DDESS/abnormal-3DESS-128-64.npy"
dataset = KneeMRIDataset(normal_path, abnormal_path, transform=transform)

# =============================================================================
# FUNÇÃO CRIA MODELO (C/ R3D_18 E DATAPARALLEL)
# =============================================================================
def create_model():
    """
    Cria o modelo r3d_18 com pesos pré-treinados (Kinetics-400),
    substitui a última camada e, se houver mais de uma GPU, utiliza DataParallel.
    """
    # Carrega modelo com pesos "DEFAULT" (Kinetics-400)
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    # Ajusta a última camada para saída de 1 neurônio (classificação binária)
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Se houver mais de 1 GPU, aplica DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model.to(device)

# =============================================================================
# FUNÇÃO DE TREINAMENTO (1 FOLD)
# =============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stop_patience, fold):
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Replicar canal para ter 3 canais (compatível com r3d_18)
            inputs = inputs.repeat(1, 3, 1, 1, 1)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Replicar canal para 3 canais
                inputs = inputs.repeat(1, 3, 1, 1, 1)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Métricas
        accuracy = np.mean(np.array(all_labels) == np.array(all_preds)) * 100
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        f1 = f1_score(all_labels, all_preds)

        log_message(
            f"Fold {fold+1} | Época {epoch + 1}/{num_epochs} "
            f"| Treino: {train_loss:.4f} | Val: {val_loss:.4f} "
            f"| Acurácia: {accuracy:.2f}% | AUC: {auc:.4f} | F1: {f1:.4f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_path = f"best_model_fold_{fold + 1}.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                log_message(f"Fold {fold+1} - Early stopping at epoch {epoch + 1}")
                break

    return {
        'test_loss': val_loss,
        'test_accuracy': accuracy,
        'test_auc': auc,
        'test_f1': f1
    }

# =============================================================================
# FUNÇÃO DE TREINAMENTO COM CROSS-VALIDATION (K-FOLD)
# =============================================================================
def train_with_cross_validation(dataset, num_epochs, early_stop_patience, n_splits):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        log_message(f"\nFold {fold + 1}/{n_splits}")

        # Oversampling nos índices de treino para balanceamento de classes
        ros = RandomOverSampler(random_state=42)
        train_ids_array = np.array(train_ids).reshape(-1, 1)
        train_labels_list = [dataset[i][1].item() for i in train_ids]
        resampled_train_ids, _ = ros.fit_resample(train_ids_array, train_labels_list)
        resampled_train_ids = resampled_train_ids.flatten().tolist()

        train_subsampler = SubsetRandomSampler(resampled_train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler,
                                num_workers=NUM_WORKERS, pin_memory=True)

        # Criar modelo e configurar otimização
        model = create_model()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        pos_weight = torch.tensor([POS_WEIGHT_VALUE]).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean', pos_weight=pos_weight)

        # Treinamento do fold atual
        fold_results = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, num_epochs,
            early_stop_patience, fold
        )
        results.append(fold_results)

    return results

# =============================================================================
# EXECUTANDO O CROSS-VALIDATION
# =============================================================================
results = train_with_cross_validation(
    dataset,
    num_epochs=NUM_EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
    n_splits=N_SPLITS
)

log_message(f"Resultados do Cross-Validation: {results}")

# Exibir resultados finais no terminal
print("\nResultados Finais:")
print("{:<15} {:<15} {:<15} {:<15} {:<15}".format(
    "Fold", "Perda (Teste)", "Acurácia", "F1-Score", "AUC-ROC"
))
for i, res in enumerate(results):
    print(f"Fold {i + 1}: {res['test_loss']:.4f} | {res['test_accuracy']:.2f}% | {res['test_f1']:.4f} | {res['test_auc']:.4f}")
