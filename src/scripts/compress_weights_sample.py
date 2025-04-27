import torch
import zlib
import io
import os

# Caminho do arquivo original
original_path = f"/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/acc_weights_pruned_10_alexnet_cifar10.pt"
compressed_path = f"/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/acc_weights_pruned_10_alexnet_cifar10.pt.zlib"

# Lê o conteúdo original e comprime
with open(original_path, 'rb') as f:
    original_data = f.read()

compressed_data = zlib.compress(original_data)

# Salva o arquivo comprimido
with open(compressed_path, 'wb') as f:
    f.write(compressed_data)

# Remove o original se quiser economizar espaço
#os.remove(original_path)

# Opcional: mostrar tamanhos
print("Tamanho original:", len(original_data))
print("Tamanho comprimido (zlib):", len(compressed_data))
