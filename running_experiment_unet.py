import subprocess
import os

EXPECTED_EPOCHS = 1000
def run_script(script_path, *args):
    result = subprocess.run(['python', script_path] + list(args), capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

for experiment in ["100p_train"]:

    # Create the text input files
    script_path = 'create_pedbrats.py'
    args = ['--training_folder', experiment]  
    run_script(script_path, *args)

    # Run the label transfer script
    script_path = 'label_transfer.py'
    args = ['--phase', 'train']  
    run_script(script_path, *args)

    # Run the training
    script_path = 'train.py'
    args = ["--data_root_path", "../../data/",
                 "--num_workers", "8",
                 "--device", "0",
                 "--num_samples", "1",
                 "--uniform_sample",
                 "--backbone", "unet",
                 "--dataset_list", "PAOT_BRAIN",
                 "--datasetkey", "20",
                 "--batch_size", "2",
                 "--max_epoch", str(EXPECTED_EPOCHS + 1),
                 "--local_rank", "0",
                 "--log_name", "testing-unetv2",
                 "--report_wandb",
                 "--save_epoch_freq", "10",
                 "--pretrain", "./pretrained_weights/unet.pth",
                 "--word_embedding", "./pretrained_weights/txt_encoding.pth"]  
    run_script(script_path, *args)

    # Run the label transfer script
    script_path = 'label_transfer.py'
    args = ['--phase', 'test']  
    run_script(script_path, *args)

    # Run the inference
    script_path = 'test.py'
    args = [
            "--data_root_path", "../../data/",
            "--num_workers", "8",
            "--device", "0",
            "--num_samples", "1",
            "--dataset_list", "PAOT_BRAIN",
            "--batch_size", "1",
            "--epoch", "1000",
            "--backbone", "unet",
            "--log_name", "testing-unetv2",
            "--store_result",
            "--resume", f"./out/testing-unetv2/epoch_1000.pth"
            ]
    run_script(script_path, *args)



