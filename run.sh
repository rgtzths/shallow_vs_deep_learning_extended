#Dataset preprocessing
python preprocess.py

#Select best network for datasets missing a network
python dnn_selection.py

#Running the training/prediction comparison
python train.py

#Explaining
python xai_analysis.py
