# 🤖 Treinamento de Modelos CNN 3D

Este repositório contém scripts e análises detalhadas para o treinamento de Redes Neurais Convolucionais (CNN), explorando diferentes hiperparâmetros como taxa de aprendizado e tamanho de batch. O objetivo é estudar o impacto dessas variações no desempenho do modelo.

---

## 🔍 Projeto: Treinamento e Avaliação de Redes Neurais Convolucionais 3D (CNN 3D)

Este projeto implementa e avalia um modelo de Rede Neural Convolucional 3D (CNN 3D) para classificação de imagens médicas tridimensionais. Ele utiliza o dataset público **OAI-MRI-3DDESS** e explora diversos aspectos do treinamento de redes neurais, como ajuste de hiperparâmetros e visualização de resultados.

## 🛠️ Requisitos do Sistema

- Python 3.8+
- GPU com suporte a CUDA (opcional, mas recomendado)

## 🌐 Estrutura do Repositório

1. [**app-boa-avaliacao-190125.ipynb**](https://github.com/thallescotta/treinamento-cnn-3d/blob/main/app-boa-avaliacao-190125.ipynb)
   - Primeiro modelo criado do início ao fim baseado no DenseNet121.

2. [**doze.ipynb**](https://github.com/thallescotta/treinamento-cnn-3d/blob/main/doze.ipynb)
   - Entendendo sobre Épocas.

3. [**2benchmark-modelos-cnn-3d.ipynb**](https://github.com/thallescotta/treinamento-cnn-3d/blob/main/2benchmark-modelos-cnn-3d.ipynb)
   - Benchmark dos modelos DenseNet121, ResNet50, EfficientNet-B0, MobileNetV2 e VGG16.

4. [**Simplified-UNet.ipynb**](https://github.com/thallescotta/treinamento-cnn-3d/blob/main/Simplified-UNet.ipynb)
   - Modelo UNet usado com alguns prints iniciais.


## 🔼 Dataset Utilizado

### OAI-MRI-3DDESS
Dataset público obtido da iniciativa Osteoarthritis Initiative (OAI), contendo imagens médicas tridimensionais da articulação do joelho. As imagens foram classificadas como normais ou anormais, com base nos graus de Kellgren-Lawrence.

### Informações do Dataset:
- **Fonte**: [Kaggle - OAI-MRI-3DDESS](https://www.kaggle.com/datasets/mohamedberrimi/oaimri3ddess/data)
- **Formato**: `.npy`
- **Tamanhos**: Total 
  - `abnormal-3DESS-128-64.npy`: Imagens anormais 5.52 GB
  - `normal-3DESS-128-64.npy`: Imagens normais 6.96 GB

## 🚀 Tecnologias Utilizadas

- **Linguagem**: Python
- **Bibliotecas Principais**:
  - PyTorch
  - NumPy
  - Matplotlib
  - scikit-learn

## 📊 Informações de Hardware e GPU

Este projeto foi executado em uma máquina com duas GPUs NVIDIA GeForce RTX 2080 Ti. Abaixo estão os detalhes capturados pelo comando `nvidia-smi`:

```bash
Fri Jan 17 11:30:12 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off | 00000000:01:00.0 Off |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 2080 Ti     Off | 00000000:02:00.0 Off |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

- **Driver NVIDIA**: 535.183.01
- **Versão do CUDA**: 12.2
- **Memória Total por GPU**: 11.2 GiB
- **Utilização durante o Treinamento**:
  - GPU 0: 48% de utilização e 5684 MiB de memória usada.
  - GPU 1: 83% de utilização e 5696 MiB de memória usada.


## ✨ Sobre o Autor
- **Thalles Fontainha**
  - Cientista da computação e especialista em sistemas de comunicações de satélites pela **StarOne - Claro**.
  - Doutorando em Instrumentação e Óptica Aplicada pelo **CEFET-RJ** ([PPGIO](http://www.dippg.cefet-rj.br/ppgio/)).
