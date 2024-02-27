import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_processed_dataset(input_dir='./miniimagenet', output_dir='./ProcessedMiniImagenet', test_size=0.2):
    """
    Splits the MiniImagenet dataset into train and test sets and organizes them into a new directory structure with a progress bar.
    
    Parameters:
    - input_dir: Path to the original MiniImagenet dataset.
    - output_dir: Path to the output directory for the processed dataset.
    - test_size: Fraction of the dataset to be used as test set.
    """
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    train_dir = os.path.join(output_dir, 'Train')
    test_dir = os.path.join(output_dir, 'Test')
    
    # Create train and test directories
    for dir_path in [train_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for class_name in classes:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
    
    # Split and copy images with progress bar
    for class_name in tqdm(classes, desc="Processing Classes"):
        class_dir = os.path.join(input_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        
        # Copy images to their respective directories with progress bar for each class
        for img in tqdm(train_imgs, desc=f"Copying Train Images for {class_name}", leave=False):
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))
        for img in tqdm(test_imgs, desc=f"Copying Test Images for {class_name}", leave=False):
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))
            
    print("Dataset successfully split and copied to:", output_dir)

# Example usage
create_processed_dataset(test_size=0.2, input_dir='./food-101', output_dir='./ProcessedFood')
