import pandas as pd
import glob
import os
import re

# Directory containing the CSV files
directory = '/media/marronedantas/HD4TB/Projects/gap-pruning/backlog'

# Optional substring filter (set to None if no filter required)
substring_filter = 'alexnet_flowers102'

# Human-friendly sort function
def human_sort(files):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)

# Create a list to store results
results = []

# Get sorted list of files ending with 'output.csv'
file_list = human_sort(glob.glob(os.path.join(directory, '*output.csv')))

# Apply substring filter if provided
if substring_filter:
    file_list = [f for f in file_list if substring_filter in os.path.basename(f)]

# Iterate through sorted and filtered files
for file_path in file_list:
    df = pd.read_csv(file_path)
    
    max_train_acc = df['train_acc'].max()
    max_val_acc = df['val_acc'].max()
    min_train_loss = df['# train_loss'].min()
    min_val_loss = df['val_loss'].min()
    
    results.append({
        'file_path': file_path,
        'train_acc': max_train_acc,
        'val_acc': max_val_acc,
        'train_loss': min_train_loss,
        'val_loss': min_val_loss
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f'{substring_filter}_summary_metrics_resnet.csv', index=False)

print("Summary metrics saved to summary_metrics.csv")