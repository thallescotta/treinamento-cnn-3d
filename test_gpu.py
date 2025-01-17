import tensorflow as tf
print("TensorFlow compilado com suporte CUDA:", tf.test.is_built_with_cuda())
print("Suporte cuDNN ativo:", tf.test.is_built_with_gpu_support())
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
