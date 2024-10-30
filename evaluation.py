import os
import numpy as np
import nibabel as nib

# DICE coefficient function
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    if np.sum(y_true) + np.sum(y_pred) == 0:
        return 1  # If both are empty, it's a perfect match
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# Function to load NIfTI files
def load_nifti_file(filepath):
    return nib.load(filepath).get_fdata()

# Compute DICE scores for labels and predictions
def compute_dice_scores(main_folder):

    folders = os.listdir(main_folder)
    #label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    #prediction_files = sorted([f for f in os.listdir(predictions_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    dice_scores_individual = {1: [], 2: [], 3: []}
    dice_scores_evaluation = {'ET': [], 'TC': [], 'WT': []}

    for folder in folders:
        files = os.listdir(os.path.join(main_folder,folder))
        label_name = [f for f in files if 'gt' in f][0]
        pred_name = [f for f in files if 'result_v1' in f][0]
        
        label_path = os.path.join(main_folder, folder,label_name)
        prediction_path = os.path.join(main_folder, folder,pred_name)

        labels = load_nifti_file(label_path)
        predictions = load_nifti_file(prediction_path)
        
        # Individual class DICE scores (Label 1 → Pred 33, Label 2 → Pred 34, Label 3 → Pred 35)
        for label, pred_label in zip([1, 2, 3], [33, 34, 35]):
            label_mask = (labels == label)
            prediction_mask = (predictions == pred_label)
            dice_score = dice_coefficient(label_mask, prediction_mask)
            dice_scores_individual[label].append(dice_score)
        
        # Evaluation masks for labels
        ET_mask_label = (labels == 3)  # Label 3 corresponds to ET
        CT_mask_label = (labels == 1) | (labels == 3)  # Labels 1 and 3 correspond to CT
        WT_mask_label = (labels == 1) | (labels == 2) | (labels == 3)  # Labels 1, 2, and 3 correspond to WT

        # Evaluation masks for predictions
        ET_mask_pred = (predictions == 35)  # Prediction 35 corresponds to ET
        CT_mask_pred = (predictions == 33) | (predictions == 35)  # Predictions 33 and 35 correspond to CT
        WT_mask_pred = (predictions == 33) | (predictions == 34) | (predictions == 35)  # Predictions 33, 34, and 35 correspond to WT
        
        # Compute DICE scores for evaluation classes
        dice_ET = dice_coefficient(ET_mask_label, ET_mask_pred)
        dice_TC = dice_coefficient(CT_mask_label, CT_mask_pred)
        dice_WT = dice_coefficient(WT_mask_label, WT_mask_pred)
        
        dice_scores_evaluation['ET'].append(dice_ET)
        dice_scores_evaluation['TC'].append(dice_TC)
        dice_scores_evaluation['WT'].append(dice_WT)

    # Compute average DICE scores
    avg_dice_scores_individual = {label: np.mean(scores) for label, scores in dice_scores_individual.items()}
    avg_dice_scores_evaluation = {label: np.mean(scores) for label, scores in dice_scores_evaluation.items()}

    return avg_dice_scores_individual, avg_dice_scores_evaluation

# Main function to run the DICE computation
def main():

    main_dir = "out/100p_train/test_healthp_50/output/20_pedbrain/"

    # Compute DICE scores
    avg_dice_scores_individual, avg_dice_scores_evaluation = compute_dice_scores(main_dir)
    
    # Print average DICE scores for individual tumor classes
    print("Average DICE scores for individual classes:")
    for label, score in avg_dice_scores_individual.items():
        print(f"  Class {label}: {score:.4f}")
    
    # Print average DICE scores for evaluation classes
    print("\nAverage DICE scores for evaluation classes:")
    for label, score in avg_dice_scores_evaluation.items():
        print(f"  {label}: {score:.4f}")

if __name__ == "__main__":
    main()
