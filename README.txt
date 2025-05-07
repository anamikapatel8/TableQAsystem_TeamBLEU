#Train the model over the FeTaQA training data
python3 train.py 

#Test the saved model on FeTaQA test data
python3 model_evaluate.py

#Trained model is saved as models/t5_fetaqa