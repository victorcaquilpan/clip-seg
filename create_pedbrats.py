import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training_folder", type=str, default = '100p_train', help="Training root")
args = parser.parse_args()

# Define the directories
main_dir = '../../data/20_pedbrain/'
output_train_file = 'dataset/dataset_list/PAOT_BRAIN_train.txt'
output_val_file = 'dataset/dataset_list/PAOT_BRAIN_val.txt'
output_test_file = 'dataset/dataset_list/PAOT_BRAIN_test.txt'

# Get list of files
main_train_folder = os.path.join(main_dir,args.training_folder)
main_test_folder = os.path.join(main_dir,'test')


for phase in ['train', 'test']:

    if phase == 'train':
        img_files = sorted(os.listdir(os.path.join(main_train_folder,'images')))
        label_files = sorted(os.listdir(os.path.join(main_train_folder,'labels')))
    elif phase == 'test':
        img_files = sorted(os.listdir(os.path.join(main_test_folder,'images')))
        label_files = sorted(os.listdir(os.path.join(main_test_folder,'labels')))

    # Open the output file
    if phase == 'train':
        with open(output_train_file, 'w') as f:
            # Iterate over the image files and find matching label files
            for img_file in img_files:
                # Construct the full path for image and label files
                img_path = os.path.join(main_train_folder,'images', img_file).replace("../../data/","")
                label_path = os.path.join(main_train_folder,'labels', img_file.replace('img', 'label'))
                
                # Check if the label file exists
                if os.path.exists(label_path):
                    # Write the paths to the text file
                    f.write(f"{img_path}\t{label_path.replace('../../data/','')}\n")
                else:
                    print(f"Warning: No matching label for {img_file}")
        with open(output_val_file, 'w') as f:
            # Iterate over the image files and find matching label files
            for img_file in img_files[0:1]:
                # Construct the full path for image and label files
                img_path = os.path.join(main_train_folder,'images', img_file).replace("../../data/","")
                label_path = os.path.join(main_train_folder,'labels', img_file.replace('img', 'label'))
                
                # Check if the label file exists
                if os.path.exists(label_path):
                    # Write the paths to the text file
                    f.write(f"{img_path}\t{label_path.replace('../../data/','')}\n")
                else:
                    print(f"Warning: No matching label for {img_file}")
    elif phase == 'test':
        with open(output_test_file, 'w') as f:
            # Iterate over the image files and find matching label files
            for img_file in img_files:
                # Construct the full path for image and label files
                img_path = os.path.join(main_test_folder,'images', img_file).replace("../../data/","")
                label_path = os.path.join(main_test_folder,'labels', img_file.replace('img', 'label'))
                
                # Check if the label file exists
                if os.path.exists(label_path):
                    # Write the paths to the text file
                    f.write(f"{img_path}\t{label_path.replace('../../data/','')}\n")
                else:
                    print(f"Warning: No matching label for {img_file}")

