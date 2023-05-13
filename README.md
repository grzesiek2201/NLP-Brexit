Predicting attitude towards Brexit based on article headlines.

Arguments:
-model: str, The model type: multibayes, linsvc, mlp, sgd.
-file_predict: str, File to perdict (has target values), filename of .txt extension in ./data folder.
		Prints the most important statistical metrics.
-file_classify: str, File to classify, filename of .txt extension in ./data folder. Has no target values.
-file_train: str, File to train on, filename of .txt extension in ./data folder. Model is chosen by -model
		argument. After learning it overwrites previous model of this type. To classify test
		dataset, use -file_classify.
-cross_val: bool, If True, after training the model with -file_train argument, it will also display 
			cross-validation results.


Example usage:
python main.py -file_train TextBrexit2_text.txt -file_predict test.txt -model multibayes -cross_val True