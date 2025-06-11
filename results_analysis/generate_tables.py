
import pathlib
import re



results_dir =  pathlib.Path("results")

slicing_datasets = ["Slicing5G", "NetSlice5G", "KPI_KQI", "QoS_QoE", "UNAC"]
ids_datasets = ["UNSW", "IOT_DNL", "Botnet_IOT", "IoTID20", "RT_IOT"]
models = [("DNN", "DNN"), ("LOG", "LR"), ("KNN", "k-NN"), ("SVM", "SVM"), ("NB", "GaussianNB"), ("DT","DT") , ("RF", "RF"), ("ABC", "AdaBoost"), ("GBC", "Gradient Boosting")]
models = [("DNN", "DNN"), ("KNN", "k-NN"), ("NB", "GaussianNB"), ("DT","DT") , ("RF", "RF"), ("ABC", "AdaBoost"), ("GBC", "Grad. Boost.")]

extracted_data = {}
def to_pretty_sci(n, precision=2):
    s = f"{n:.{precision}e}"
    mantissa, exp = s.split("e")
    exp = int(exp)  # Convert exponent to int
    return f"{mantissa} \\times 10^{'{'+str(exp)+'}'}"

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
                model_name, train_time, train_energy, infer_time, infer_energy, roc_auc, f1, mcc = re.split(r'\|', line.strip())[1:-1]
                # Remove extra spaces and potential ± signs
                metrics = [float(metric.strip().split('±')[0]) 
                           for metric in [train_time, train_energy, infer_time, infer_energy]] + [float(metric) for metric in [roc_auc, f1, mcc]]
                           
                # Add one for models with perfect score (due to ± sign removal)
                #metrics.extend([1] * (3 - len([m for m in metrics if m < 1])))

                extracted_data[str(folder).split("/")[1]][model_name.strip()] = metrics

print("Slicing mcc")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        text += f" & {extracted_data[dataset][model][-1]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nIDS mcc")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        #print(extracted_data[dataset][model][-1])
        text += f" & {extracted_data[dataset][model][-1]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nSlicing f1")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        text += f" & {extracted_data[dataset][model][-2]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nIDS f1")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        #print(extracted_data[dataset][model][-1])
        text += f" & {extracted_data[dataset][model][-2]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nSlicing roc_auc")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        text += f" & {extracted_data[dataset][model][-3]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nIDS roc_auc")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        #print(extracted_data[dataset][model][-1])
        text += f" & {extracted_data[dataset][model][-3]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nSlicing training time")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        if model == "DNN":
            if extracted_data[dataset][model][0] > 1000 or extracted_data[dataset][model][0] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][0],2)}$(-)"
            else:
                text += f" & {extracted_data[dataset][model][0]:.2f}(-)"
        else:
            acc = (1- extracted_data[dataset][model][0]/ extracted_data[dataset]["DNN"][0])*100
            if extracted_data[dataset][model][0] > 1000 or extracted_data[dataset][model][0] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][0],2)}$({acc:.0f}\\%)"
            else:
                text += f" & {extracted_data[dataset][model][0]:.2f}({acc:.0f}\\%)"

    text += "\\\\"
    print(text)

print("\n\nSlicing prediction time")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        if model == "DNN":
            if extracted_data[dataset][model][2] > 1000 or extracted_data[dataset][model][2] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][2],2)}$(-)"
            else:
                text += f" & {extracted_data[dataset][model][2]:.2f}(-)"
        else:
            acc = (1- extracted_data[dataset][model][2]/ extracted_data[dataset]["DNN"][2])*100
            if extracted_data[dataset][model][2] > 1000 or extracted_data[dataset][model][2] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][2],2)}$({acc:.0f}\\%)"
            else:
                text += f" & {extracted_data[dataset][model][2]:.2f}({acc:.0f}\\%)"

    text += "\\\\"
    print(text)

print("\n\nSlicing training energy")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        if extracted_data[dataset][model][1] > 1000 or extracted_data[dataset][model][1] < 0.01: 
            text += f" & ${to_pretty_sci(extracted_data[dataset][model][1],2)}$"
        else:
            text += f" & {extracted_data[dataset][model][1]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nSlicing prediction energy")
for model, name in models:
    text = f"{name}"
    for dataset in slicing_datasets:
        if extracted_data[dataset][model][3] > 1000 or extracted_data[dataset][model][3] < 0.01:
            text += f" & ${to_pretty_sci(extracted_data[dataset][model][3],2)}$"
        else:
            text += f" & {extracted_data[dataset][model][3]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nids training time")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        if model == "DNN":
            if extracted_data[dataset][model][0] > 1000 or extracted_data[dataset][model][0] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][0],2)}$(-)"
            else:
                text += f" & {extracted_data[dataset][model][0]:.2f}(-)"
        else:
            acc = (1- extracted_data[dataset][model][0]/ extracted_data[dataset]["DNN"][0])*100
            if extracted_data[dataset][model][0] > 1000 or extracted_data[dataset][model][0] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][0],2)}$({acc:.0f}\\%)"
            else:
                text += f" & {extracted_data[dataset][model][0]:.2f}({acc:.0f}\\%)"
                
    text += "\\\\"
    print(text)

print("\n\nids prediction time")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        if model == "DNN":
            if extracted_data[dataset][model][2] > 1000 or extracted_data[dataset][model][2] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][2],2)}$(-)"
            else:
                text += f" & {extracted_data[dataset][model][2]:.2f}(-)"
        else:
            acc = (1- extracted_data[dataset][model][2]/ extracted_data[dataset]["DNN"][2])*100
            if extracted_data[dataset][model][2] > 1000 or extracted_data[dataset][model][2] < 0.01: 
                text += f" & ${to_pretty_sci(extracted_data[dataset][model][2],2)}$({acc:.0f}\\%)"
            else:
                text += f" & {extracted_data[dataset][model][2]:.2f}({acc:.0f}\\%)"

    text += "\\\\"
    print(text)

print("\n\nids training energy")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        if extracted_data[dataset][model][1] > 1000 or extracted_data[dataset][model][1] < 0.01: 
            text += f" & ${to_pretty_sci(extracted_data[dataset][model][1],2)}$"
        else:
            text += f" & {extracted_data[dataset][model][1]:.2f}"
    text += "\\\\"
    print(text)

print("\n\nids prediction energy")
for model, name in models:
    text = f"{name}"
    for dataset in ids_datasets:
        if extracted_data[dataset][model][3] > 1000 or extracted_data[dataset][model][3] < 0.01:
            text += f" & ${to_pretty_sci(extracted_data[dataset][model][3],2)}$"
        else:
            text += f" & {extracted_data[dataset][model][3]:.2f}"
    text += "\\\\"
    print(text)