<h3>Predicting attitude towards Brexit based on article headlines.</h3>


Arguments:

<code>-model</code>: str, The model type: multibayes, linsvc, mlp, sgd.

<code>-file_predict</code>: str, File to perdict (has target values), filename of .txt extension in ./data folder. Prints the most important statistical metrics.
		
<code>-file_classify</code>: str, File to classify, filename of .txt extension in ./data folder. Has no target values.

<code>-file_train</code>: str, File to train on, filename of .txt extension in ./data folder. Model is chosen by -model argument. After learning it overwrites previous model of this type. To classify test dataset, use -file_classify.

<code>-cross_val</code>: bool, If True, after training the model with -file_train argument, it will also display cross-validation results.


Example usage:

<code>python main.py -file_train TextBrexit2_text.txt -file_predict test.txt -model multibayes -cross_val True</code>
