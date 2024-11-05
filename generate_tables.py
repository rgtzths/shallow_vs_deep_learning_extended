
import pathlib
import re



results_dir =  pathlib.Path("results")

slicing_datasets = ["Slicing5G", "NetSlice5G"]
ids_datasets = ["UNSW", "IOT_DNL"]#,  "TON_IOT"]
models = [("DNN", "DNN"), ("LOG", "LR"), ("KNN", "k-NN"), ("SVM", "SVM"), ("NB", "GaussianNB"), ("DT","DT") , ("RF", "RF"), ("ABC", "AdaBoost"), ("GBC", "Gradient Boosting")]

extracted_data = {}

for folder in results_dir.glob("*"):
    if folder.is_dir():
        for file in folder.glob(f"*.md"):
            md_file = open(file)
            extracted_data[str(folder).split("/")[1]] = {"file":md_file}
            lines = md_file.readlines()
            # Skip the header row
            lines = lines[2:]
            for line in lines:
                # Extract model name, train time, infer time, ACC, F1, MCC
                model_name, train_time, infer_time, acc, f1, mcc = re.split(r'\|', line.strip())[1:-1]
                # Remove extra spaces and potential ± signs
                metrics = [float(metric.strip().split('±')[0]) 
                           for metric in [train_time, infer_time]] + [float(metric) for metric in [acc, f1, mcc]]
                           
                # Add one for models with perfect score (due to ± sign removal)
                #metrics.extend([1] * (3 - len([m for m in metrics if m < 1])))

                extracted_data[str(folder).split("/")[1]][model_name.strip()] = metrics

print("Slicing accuracy")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        text += f" & {extracted_data[dataset][model][-1]:.2f}"
    text += "\\\\"
    print(text)

print("IDS accuracy")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        #print(extracted_data[dataset][model][-1])
        text += f" & {extracted_data[dataset][model][-1]:.2f}"
    text += "\\\\"
    print(text)

print("Slicing training")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        if model == "DNN":
            text += f" & {extracted_data[dataset][model][0]}(-)"
        else:
            acc = (1- extracted_data[dataset][model][0]/ extracted_data[dataset]["DNN"][0])*100
            text += f" & {extracted_data[dataset][model][0]}({acc:.0f}\\%)"
    text += "\\\\"
    print(text)

print("Slicing prediction")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        if model == "DNN":
            text += f" & {extracted_data[dataset][model][1]}(-)"
        else:
            acc = (1- extracted_data[dataset][model][1]/ extracted_data[dataset]["DNN"][1])*100
            text += f" & {extracted_data[dataset][model][1]}({acc:.0f}\\%)"
    text += "\\\\"
    print(text)

print("ids training")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        if model == "DNN":
            text += f" & {extracted_data[dataset][model][0]}(-)"
        else:
            acc = (1- extracted_data[dataset][model][0]/ extracted_data[dataset]["DNN"][0])*100
            text += f" & {extracted_data[dataset][model][0]}({acc:.0f}\\%)"
    text += "\\\\"
    print(text)

print("ids prediction")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        if model == "DNN":
            text += f" & {extracted_data[dataset][model][1]}(-)"
        else:
            acc = (1- extracted_data[dataset][model][1]/ extracted_data[dataset]["DNN"][1])*100
            text += f" & {extracted_data[dataset][model][1]}({acc:.0f}\\%)"
    text += "\\\\"
    print(text)