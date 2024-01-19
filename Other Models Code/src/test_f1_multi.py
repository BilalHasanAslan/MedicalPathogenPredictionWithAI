import torch
from custom_dataset import CustomDataset as cd
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import main_f1_multi as mn

'''
This code encapsulates the testing/validation pipeline used in our paper.
Given all model permuations, this code provides behaviour to test all 240 of them.
This code also contains the Default Validation algorithm
'''
model_names = ["convnext_tiny", "convnext_small", "convnext_base", "swin_v2_t", "swin_v2_s", "swin_v2_b", "densenet121", 
               "densenet169", "densenet201", "resnet18", "resnet50", "resnet152", "efficientnet_v2_s", 
               "efficientnet_v2_m", "efficientnet_v2_l"]
pretrained_methods = ["pretrained"]
da_methods = ["no_da", "RandomCrop", "RandAug"]
datasets = ["Neck"]

#Directory code to test all 240 models, provided they exist
MODEL_PATHS = ["../trained_models/" + f"{model}_{pretrained}_{da}_{dataset}.pt" 
              for model in model_names 
              for pretrained in pretrained_methods
              for da in da_methods 
              for dataset in datasets]

#Directory code to test any one model - change this to test a given model:
MODEL_PATHS_INDIVIDUAL = ["../trained_models/convnext_base_pretrained_no_da_Neck.pt"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Testing pipeline
def testModels():
    #Write results to text file
    with open('test_out.txt', 'a') as f:
        #Reconstruct models and populate data loaders via the parameter information embedded in the model names
        for model_path in MODEL_PATHS:
            model_name = model_path[17:]
            if model_path[-7:] == "lbow.pt":
                dir_path = "../dataset/Elbow"
            else:
                dir_path = "../dataset/Neck"
            #dir_path = "../dataset/Neck"
            if "not_pretrained" in model_path:
                tl_method = "not_pretrained"
            else:
                tl_method = "pre_trained"
            test_dataset = cd(f"{dir_path}/images", f"{dir_path}/test.csv")
            test_dataloader = DataLoader(test_dataset, batch_size=32, drop_last=True)
            classes = len(test_dataset.classes)

            if "not" in model_name:
                model_identifier = model_name[:model_name.find("not")-1]
            else:
                model_identifier = model_name[:model_name.find("pre")-1]
            model_identifier = model_identifier[1:]
            model = mn.create_model(model_identifier, classes, tl_method)
            model.load_state_dict(torch.load(model_path))

            model.eval()
            all_labels = []
            all_preds = []
            total_predictions = 0
            running_corrects = 0

            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                probabilities = torch.sigmoid(outputs)
                preds = (probabilities > 0.5).float()

                '''Default validation implementation
                for pred in preds:
                    if (pred[:-1] == 0).all():
                        pred[-1] = 1
                    else:
                        pred[-1] = 0'''
                        
                total_predictions += len(preds)

                batch_corrects = torch.sum(preds == labels.data, dim=1) == classes
                running_corrects += batch_corrects.sum().item()

                all_labels.append(labels.cpu().detach().numpy())
                all_preds.append(preds.cpu().detach().numpy())

            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            test_acc = running_corrects / total_predictions

            confusion_matrix = {"TP": [], "TN": [], "FP": [], "FN": []}
            
            overall_TP = 0
            overall_TN = 0
            overall_FP = 0
            overall_FN = 0

            precision_per_class = []
            tpr_per_class = []
            fpr_per_class = []

            for c in range(classes):
                TP = np.sum(np.logical_and(all_preds == c, all_labels == c))
                TN = np.sum(np.logical_and(all_preds != c, all_labels != c))
                FP = np.sum(np.logical_and(all_preds == c, all_labels != c))
                FN = np.sum(np.logical_and(all_preds != c, all_labels == c))

                # Precision calculation
                if TP + FP > 0:
                    precision = TP / (TP + FP)
                else:
                    precision = 0
                precision_per_class.append(precision)

                # TPR calculation
                if TP + FN > 0:
                    TPR = TP / (TP + FN)
                else:
                    TPR = 0
                tpr_per_class.append(TPR)

                # FPR calculation
                if FP + TN > 0:
                    FPR = FP / (FP + TN)
                else:
                    FPR = 0
                fpr_per_class.append(FPR)

            # Calculating macro averages
            macro_avg_precision = np.mean(precision_per_class)
            macro_avg_TPR = np.mean(tpr_per_class)
            macro_avg_FPR = np.mean(fpr_per_class)
            print(model_path)
            print(f"Macro Average Precision: {macro_avg_precision:.4f}")
            print(f"Macro Average TPR: {macro_avg_TPR:.4f}")
            print(f"Macro Average FPR: {macro_avg_FPR:.4f}")
            

            f.write(f'Model: {model_name}\n')
            '''for label, class_name in enumerate(test_dataset.classes):
                f.write(f'Label: {class_name}\n')
                f.write(f'TP: {confusion_matrix["TP"][label][0]} ({confusion_matrix["TP"][label][1]:.2f}%)\n')
                f.write(f'FP: {confusion_matrix["FP"][label][0]} ({confusion_matrix["FP"][label][1]:.2f}%)\n')
                f.write(f'FN: {confusion_matrix["FN"][label][0]} ({confusion_matrix["FN"][label][1]:.2f}%)\n')
                f.write(f'TN: {confusion_matrix["TN"][label][0]} ({confusion_matrix["TN"][label][1]:.2f}%)\n')'''


            overall_total = overall_TP + overall_TN + overall_FP + overall_FN
            try:
                f.write(f'\nOverall Metrics for the Model:\n')
                f.write(f'TP: {overall_TP} ({overall_TP/overall_total*100:.2f}%)\n')
                f.write(f'TN: {overall_TN} ({overall_TN/overall_total*100:.2f}%)\n')
                f.write(f'FP: {overall_FP} ({overall_FP/overall_total*100:.2f}%)\n')
                f.write(f'FN: {overall_FN} ({overall_FN/overall_total*100:.2f}%)\n')
            except:
                pass
            hamming = hamming_loss(all_labels, all_preds) * 100
            f1 = f1_score(all_labels, all_preds, average='macro') * 100
            '''auc = roc_auc_score(all_labels, all_preds, average='micro') * 100
            precision = precision_score(all_labels, all_preds, average='micro') * 100
            recall = recall_score(all_labels, all_preds, average='micro') * 100

            f.write(f'Hamming Loss: {hamming:.2f}%\n')'''
            f.write(f'F1: {f1:.2f}%\n')
            '''f.write(f'AUC: {auc:.2f}%\n')
            f.write(f'Precision: {precision:.2f}%\n')
            f.write(f'Recall: {recall:.2f}%\n\n')'''

            per_label_acc = np.mean(all_labels == all_preds, axis=0) * 100
            overall_acc = np.mean(per_label_acc)

            f.write(f'Test Accuracy: {test_acc*100:.2f}%\n')
            '''f.write('Per Label Accuracy:\n')
            for i, x in enumerate(per_label_acc):
                f.write(f'{test_dataset.classes[i]}: {x:.2f}%\n')'''
            f.write(f'\nAvg. Per Label Accuracy: {overall_acc:.2f}%\n')
            f.write('-'*30 + '\n')

if __name__ == "__main__":
    testModels()