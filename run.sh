#Dataset preprocessing
#python preprocess.py

#Select best network for datasets missing a network
#python dnn_selection.py -d QoS_QoE
#python dnn_selection.py -d IoTID20
#python dnn_selection.py -d Botnet_IOT

#Running the training/prediction comparison
python train.py -d QoS_QoE
python train.py -d IoTID20
python train.py -d Botnet_IOT
python train.py -d NetSlice5G
python train.py -d Slicing5G
python train.py -d IOT_DNL
python train.py -d UNSW
python train.py -d KPI_KQI
python train.py -d RT_IOT
python train.py -d UNAC

#Explaining
python xai_analysis.py

#Generate results

#Generate the Decision tree diagram for Slicing5G
python results_analysis/plot_decision_tree.py

#Generate learning curves for the DNNs
python plot_learning_curves.py

#Generate the tables presented in the paper
python generate_tables.py > results/tables.txt

#Generate the correlation plot
python plot_explanations.py

#Generate the PI & PDV individual plots
python plot_bars.py

#Generate the decision tree
python plot_decision_tree.py