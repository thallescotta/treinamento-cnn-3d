import os
import gc
import subprocess
import tensorflow as tf


def liberar_memoria_ram():
    """Libera a memória RAM não utilizada."""
    print("Liberando memória RAM...")
    gc.collect()
    try:
        os.system("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
        print("Memória RAM liberada com sucesso.")
    except Exception as e:
        print(f"Erro ao liberar memória RAM: {e}")


def liberar_memoria_gpu():
    """Libera a memória de todas as GPUs NVIDIA."""
    print("Liberando memória das GPUs...")
    try:
        # Obtém os IDs dos processos usando as GPUs
        processo_gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
        processos = [int(pid) for pid in processo_gpu.decode('utf-8').split('\n') if pid.strip().isdigit()]
        
        # Mata cada processo que está usando as GPUs
        for pid in processos:
            os.system(f'kill -9 {pid}')
        print("Memória das GPUs liberada com sucesso.")
    except Exception as e:
        print(f"Erro ao liberar memória das GPUs: {e}")


def configurar_tf_para_gpus():
    """Configura TensorFlow para usar GPUs com memória controlada."""
    print("Configurando TensorFlow para GPUs...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            memory_limit_per_gpu = 6144  # Limite de 6 GB por GPU
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_per_gpu)]
                )
            print(f"Configuração de memória aplicada: {memory_limit_per_gpu} MB por GPU.")
        except RuntimeError as e:
            print(f"Erro ao configurar as GPUs: {e}")
    else:
        print("Nenhuma GPU foi encontrada. Verifique a instalação do TensorFlow.")


if __name__ == "__main__":
    print("Iniciando processo de liberação de memória...")
    liberar_memoria_ram()
    liberar_memoria_gpu()
    configurar_tf_para_gpus()
    print("Processo concluído.")
