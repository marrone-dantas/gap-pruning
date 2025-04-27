import os
import pandas as pd

def find_filenames_with_words(words, directory):
    """
    Returns a list of filenames from the given directory that contain all specified words in the filename, regardless of order.

    Args:
        words (list): List of words to look for in filenames.
        directory (str): Path to the directory to search.

    Returns:
        list: Filenames that contain all the given words.
    """
    matched_files = []

    for filename in os.listdir(directory):
        lowercase_name = filename.lower()
        if all(word.lower() in lowercase_name for word in words):
            matched_files.append(filename)

    return matched_files


words_to_search = ["resnet50.ra_in1k", "cifar100_", "eval"]
directory_path = "/media/marronedantas/HD4TB/Projects/gap-pruning/backlog/gprt_summary/backlog"
result = find_filenames_with_words(words_to_search, directory_path)

output = []

for file_path in result:
    
    data = pd.read_csv(directory_path+'/'+file_path)
    result_data = pd.DataFrame(data.tail(2)['Predicted Label']).T
    result_data.columns = ['loss','acc']
    result_data['file_name'] = file_path
    result_data['prune_rate'] = result_data['file_name'].str.extract(r'pruned_(\d+)')
    output.append(result_data)

output_df = pd.concat(output, axis=0, ignore_index=True).reset_index(drop=True)
output_df = output_df.fillna(0)
output_df['prune_rate'] = output_df['prune_rate'].astype(float)
output_df = output_df.sort_values(by='prune_rate')
file_to_save = output_df.iloc[0]['file_name'].replace('.csv','_summary.csv')
output_df.to_csv(file_to_save, index=False)