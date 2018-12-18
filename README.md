# KBautotag-dl

This project is to do multioutput/multilabel classification, based on Apple KB hierachical multilabel dataset, using deep learning 
algorithms on tensorflow framework.

1. run "sh dl_train.sh" to generate training data, and train models, model accuracy result will be save in file "dl_model_compare.csv".
2. run "python dl_prediction.py" to do prediction
3. saved model in directory of dl_model, for example: dl_model>level2_clean>cnn