import os
import shutil

from tqdm import tqdm

source_dir_imgs = ""
source_dir_masks = ""
target_dir = ""

os.makedirs(target_dir, exist_ok=True)


def copy_and_rename_files(source_dir, target_dir, suffix):
    for filename in tqdm(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}{suffix}{ext}"
            target_path = os.path.join(target_dir, new_filename)
            shutil.copy2(source_path, target_path)


copy_and_rename_files(source_dir_imgs, target_dir, "_img")
copy_and_rename_files(source_dir_masks, target_dir, "_masks")
