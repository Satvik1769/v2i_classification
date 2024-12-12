New! Keyboard shortcuts … Drive keyboard shortcuts have been updated to give you first-letters navigation
# Task

## Task Description: Image Classification

The task is an **Image Classification** task. The dataset consists of images of various vehicles along with their respective categories. `Please note that the dataset may exhibit imbalances, potentially incorrect annotations and other issues`. Your objective is to train a deep-learning based architecture for the classification of the vehicles in this dataset. You are given the train and validation sets, your model will be evaluated on a separate test set to obtain your final performance.

### Task information

- Time given: 24 hrs
- Dataset: contained as zip in _vehicle_dataset_ directory

NOTE: Please do not upload the code or dataset to Github, Kaggle or any other platform on the internet.

#### Requirements

1. Your model should be developed using Python and a deep learning library such as TensorFlow or PyTorch.
2. You should `analyze, clean` (preferably using [fastdup](https://visual-layer.readme.io/docs/getting-started) or [cleanvision](https://cleanvision.readthedocs.io/en/latest/)) and pre-process the data appropriately.
3. You should experiment with different model architectures, hyper-parameters, and training strategies to find the best model for the task.
4. You should evaluate the performance of your model on the validation set using accuracy, precision, recall, F1-score and mAP metrics. Also create learning curves and confusion matrix to evaluate model performance.
5. You have to export the model into ONNX format and run the verification script to ensure the model runs properly.
6. You should provide clear and concise report of your approach using the following [format](#report-format).

### Deliverables

1. Python scripts or notebooks containing your entire code for the task (including data analysis, data cleaning, model training, model exporting, inference etc.).
2. Trained model in ONNX format (along with information about any preprocessing steps during inference).
3. A classes.txt file with the names of the classes appearing in the order as predicted by the model.
4. A report summarizing your approach, in the following [format](#report-format).
5. A table or plot showing the performance of your model on the test set, including relevant accuracy, precision, recall, F1-score metrics.
6. A README file containing clear instructions for how to run your code and reproduce your results.

### Evaluation Criteria

1. The precision of your final model on the test set.
2. The quality of your code and documentation.
3. The rigor of your approach to the task, including the models you tried and the hyper-parameters you tuned.
4. Your ability to explain and justify your decisions and choices during the model training process.

#### Report Format

1. Abstract: Summary of the report discussing method and results in brief (less than 150 words).
2. Data Analysis and Cleaning: Explain the observations about the dataset during the data analysis step, and the data cleaning steps you carried out and why.
3. Data Preprocessing: Explain the data preprocessing steps you used and the reason behind using each one.
4. Model Architecture: Details about the model architecture you are using. Also why did you select this particular architecture? What is the benefit of this architecture for the current task?
5. Training and Experimentation: Explain you training setting and methodology, along with loss function, optimizer, hyper-parameters and other important details. What modifications have you applied to the pipeline/training/model to improve performance.
6. Results and Key findings: Explain the model performance, with metrics, learning curves, and confusion matrix. Also explain key takeaways from the training and experimentation process.
7. Future work: If given sufficient time, what are the methods you will like to implement that you think will improve the performance.

**NOTE**: The task does not specificaly require a GPU, but if you want to use one and don't have access to one, you can use below mentioned free resources:

- Google Colab
- Kaggle
- Gradient by Paperspace
- Sagemeaker by Amazon
- Codesphere

or any other you know of.
