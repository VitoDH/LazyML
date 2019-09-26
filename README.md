# LazyML Documentation

**Author: Vito Liu**



## Quick Start

### Step 0. Install the required python packages

```
# Suggest installing Anaconda first
win + R
type "cmd", enter
cd path   # the path storing "requirements.txt"
pip install -r requirements.txt
```



### Step * Run without notebook

If you want to directly run the python file and don't want to go into details, here is a quick way to go.

1. Place the training set "train_demo.csv" and test set "test_demo.csv" in "LazyML/data/"
2. Use console to run python file

```shell
cd ../LazyML/src
python daily.py         # daily scrubbing work for engineers
python simulation.py    # simulation for the past data
```

1. The output will be displayed in the "../LazyML/output/".

For daily.py, you can see "embedding plot", "summary_plot", "dependence_plot", "force_plot" "confusion_matrix" in "../LazyML/output/img/" and "metrics.csv", "actionability_output.csv" in "../LazyML/output/prediction/".

For simulation.py, you can see "simulation_plot" in "../LazyML/output/img/" and "Simulation.csv" in "../LazyML/output/prediction/".



The exact meaning for each file is listed LazyML follows:

|        File Name         |                            Usage                             |
| :----------------------: | :----------------------------------------------------------: |
|      embedding plot      |       Visualize the prediction and feature in 2D space       |
|       summary_plot       |                 Check the feature importance                 |
|     dependence_plot      |            Check the impact of a specific feature            |
|        force_plot        | Check how the features impact the output of a feedback item  |
|     confusion_matrix     |       Display the exact performance of the prediction        |
|        metric.csv        | Display f1, recall, precision,accuracy, auc for the test set |
| actionability_output.csv | Contain ID, score and binary prediction for each item sorted by score |
|     simulation_plot      |   Plot the performance of the model within around 10 days    |
|      simulation.csv      |  Record the performance of the model within around 10 days   |





If you prefer to use notebook to run the codes step by step, please see the instructions below.

### Step 1.  Import file and set up an instance

```python
import importlib
import LML
importlib.reload(LazyML)
from LML import LazyML

# set up an instance
lml_ = AutoScrub()
```



### Step 2. Import all the relevant files except data

```python
# Get the Encoder Dictionary for the categorical variable
lml_.get_EncoderDict()
# Get the tfidf vectorizer for turning word into vector
lml_.get_vectorizer()
```



### Step 3. Import data and preprocessing

```python
# Load the data queried from database
train_raw = lml_.get_raw_data("train_demo.csv")
test_raw = lml_.get_raw_data("test_demo.csv")

# Perform preprocessing on train and test
train,train_display = lml_.data_preprocessing(train_raw)
test,test_display = lml_.data_preprocessing(test_raw)
# If you want to save the data after preprocessing, please specified the following parameters
train,train_display = lml_.data_preprocessing(train_raw,save=True,output_data="train.csv",output_data_display="train_display.csv")
test,test_display = lml_.data_preprocessing(test_raw,save=True,output_data="test.csv",output_data_display="test_display.csv")

# After you save it, you don't have to be re-run preprocessing again
train,train_display = lml_.get_data("train.csv","train_display.csv")
test,test_display = lml_.get_data("test.csv","test_display.csv")

# Train-Validation Split
train,val,train_display,val_display = lml_.train_val_split(train,test_size = test.shape[0],random_state = 53)

# BayesianOptimization
opt_params = lml_.optimize(train,val)

# Final Training
gbm = lml_.training(train,val,opt_params)

# Testing and predicitng
test_X,test_y = lml_.data_split(test)
test_pred,test_pred_b = lml_.predict(test_X,gbm,opt_params['threshold'])

# Evaluation: return metrics and confusion matrix
lml.performance(test_y,test_pred_b)

# Real-world Scenario, can't evaluate and directly output the prediction
test_nolabel = test.drop(columns=['label'])

```



### Step 4. Load and Save the model

```python
# Save the model
lml_.SaveModel(gbm,"lgbm")

# Load the model
gbm = lml_.LoadModel("lgbm")
```



### Step 5. Visualization

```python
# Output the force plot, summary plot and dependence plot to the local files
lml_.Visualization(gbm,test_X,test_display,test_y)

#        You can do each part seperately
# Calculate the shap values
explainer,shap_values = lml_.CalShap(gbm,test_X)

# Get the embedding plots for prediction and top 3(default) features
lml_.output_embedding_plot(shap_values,test_pred,test_y)
# Get the summary plot
lml_.output_summary_plot(shap_values,test_X)
# Get the dependence plot for top 20(default) features
lml_.output_dependence_plot(shap_values,test_X,test_display)
# Get the force plots for all items
lml_.output_force_plot(explainer,shap_values,test_display)
```



### Step 6. Simulation (back testing)

```python
# Return a dictionary containing the records of all the metrics in all the epochs
monitor = lml_.Simulation()
```











## 1. Get data, parameters and necessary files

```python
def get_params(self,params_path)

```

Get the parameters from the json file. Automatically run when initializing an instance.

```markdown
Parameters
-------
param path:  string, path storing the params
        
Returns
-------
params: Dictionary, storing the path and the name of data

```



```python
def get_EncoderDict(self)

```

Load the EncoderDict for the categorical variables



```python
def get_raw_data(self,raw_data_name=None)

```

Load the raw data queried by Kusto

```markdown
Parameters
-------
raw_data_name: string, file name of the raw data
        
Returns
-------
raw_data: DataFrame, the data queried from the database

```



```python
def get_data(self,data_path=None,verbose = False):

```

Load the data and data_display(without transformation of categorical features) after the preprocessing of the raw data

```python
Parameters
-------
data_name:  string, the file name of the data
verbose: print out the number of data,features, minimum and maxmimum of timeslice
        
Returns
-------
data: DataFrame, directly load the postprocessed data from local files
data_display: DataFrame，directly load the postprocessed data_display from local files

```



```python
def get_corpus(self)

```

Get the corpus for building the tfidf_vectorizer and save the vectorizer



```python
def get_vectorizer(self)

```

Load the tfidf_vectorizer that has been trained



## 2. Preprocessing

```python
def data_preprocessing(self,input_data,save=False,output_data=None,output_data_display=None)

```

Perform the pipeline for the preprocessing of the data from Kusto

```markdown
Parameters
-------
input_data: DataFrame, the data waiting to be cleaned
save: bool, whether to save the postprocessed data in local file
output_data: string, the filename for the output data after cleaning (save == True)
output_data: string, the filename for the output data after cleaning(display mode: categorical features not encoded) (save == True)
        
Returns
-------
data: DataFrame, the postprocessed data after cleaning
data_display: DataFrame, the postprocessed data_display after cleaning

```



```python
def filter_na(self,data=None,na_threshold = 0.95)  

```

Filter out the features with at leastt 95% missing values

```markdown
Parameters
-------
data: DataFrame, the data used for filtering NAN, can be the training or the testing
na_threshold: float, if # missing / # records > 0.95, remove this feature

```



```python
def binary_feat(self,data)

```

Transforming the string into 0/1 for binary variables

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned

```



```python
def drop_after_exam(self,data)

```

Remove some duplicated and imbalanced features after manual examination

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned

```



```python
def text_preprocessing(self,data)

```

Clean the text from user and turn it into tfidf vector

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned
Returns
-------
data: DataFrame, the raw data to be cleaned (text have been cleaned)

```



```python
def continuous_feat(self,data)

```

Take log transformation to stablize the continuous features

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned

```



```python
def Time_Process(self,data)

```

Generate feature from TimeSlice including "IsWeekday" and "DaysUsedOfBuild"

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned

```



```python
def cate_feat(self,data)

```

Encode the categorical features by default encoder dictionary EncoderDict

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned
data_display: DataFrame, the postprocessed data_display after cleaning

```



```python
def target2label(self,data)

```

Turn target into the label for classification model

```markdown
Parameters
-------
data: DataFrame, the raw data to be cleaned

```







## 3. Training, Optimization and Prediction



```python
def optimize(self,train,val)

```

Use Bayesian Optimization to optimize the parameters for the light gbm

```markdown
Parameters
-------
train: DataFrame, the training data
val: DataFrame, the validation data
        
Returns
-------
opt_params： Dictionary, storing the optimized paramters

```



```python
def data_split(self,data)

```

Split the data into design matrix X and label y

```markdown
Parameters
-------
data: DataFrame, the data to be split
        
Returns
-------
X： DataFrame, the design matrix for training or prediction
y: Array, the ground truth label

```



```python
def train_val_split(self,train,train_display,test_size)

```

Split the training data into new training and validation data for tuning the parameters

```markdown
Parameters
-------
train: DataFrame, the training data to be split
train_display: DataFrame, training data before encoding for categorical variables
test_size: int, the number of records in test set
        
Returns
-------
new_train： DataFrame, the new training data
new_val: DataFrame, the new validation data
new_train_display: DataFrame, the new training data before encoding for categorical variables
new_val_display: DataFrame, the new validation data before encoding for categorical variables

```





```python
def training(self,train,val,params,return_val=False,verbose=False)

```

Train the data with Light GBM

```markdown
Parameters
-------
train: DataFrame, the training data
val: DataFrame, the validation data
params: Dictionary, storing the paramters for Light GBM
return_val: bool, default False, need to set it to be True while optimization
verbose: bool, default False, used for printing out information while training
        
Returns
-------
tmp_gbm： Light GBM model
val_X： DataFrame, design matrix X of validation
val_y: Array, label y of validation

```



```python
def predict(self,X,clf,cutoff)

```

Make prediction by Light GBM

```markdown
Parameters
-------
X: DataFrame, the design matrix
clf: Light GBM model, the classifier
cutoff: float, the thresholid to turn the probability into binary prediction
        
Returns
-------
pred： Array, the probability of actionability
pred_b: Array, binary prediction of actionability

```



## 4. Load & Save Model

```python
def SaveModel(self,clf,name)

```

Save the model to the local files

```markdown
Parameters
-------
clf: Light GBM model, the classifier
name: string, the name of the model
        
Returns
-------
Save the model to the local files

```



```python
def LoadModel(self,name)

```

Load the model to the local files

```mariadb
Parameters
-------
name: string, the name of the model
        
Returns
-------
clf: Light GBM model, the classifier

```









## 5. Visualization

```python
def performance(y,pred)

```

combine the metrics output and the confusion matrix

```markdown
Parameters
-------
y: array, ground truth label
pred: array, prediction
        
Returns
-------
print out the metrics and plot the confusion matrix

```





```python
def CalShap(self,clf,X)

```

Calculate the shap values for the input data given the classification model

```markdown
Parameters
-------
clf: Light GBM model
X: DataFrame, the design matrix of the input
        
Returns
-------
explainer: tree explainer module
shap_values: 2-d array, shap values per sample per feature

```



```python
def output_force_plot(self,explainer,shap_values,X_display)

```

Output the force plots of each feedback item

```markdown
Parameters
-------
explainer: tree explainer module
shap_values: 2-d array, storing the shap values
X_display: DataFrame, storing the display version of the input data 
        
Returns
-------
Save the force plot to the local files

```



```python
def output_summary_plot(self,shap_values,X,top=10)

```

Output the summary plots of feature importance

```markdown
Parameters
-------
shap_values: 2-d array, storing the shap values
X: DataFrame, storing the design matrix of the input data 
top: int, top n important features 

Returns
-------
Save the summary plot to the local files

```



```python
def output_dependence_plot(self,shap_values,X,X_display,top = 20)

```

Output the dependence plots of each important feature

```markdown
Parameters
-------
shap_values: 2-d array, storing the shap values
X: DataFrame, storing the design matrix of the input data 
X_display: DataFrame, storing the display version of the input data 
top: int, top n important features 
        
Returns
-------
Save the dependence plot to the local files

```

**  There is a bug in shap. If categorical feature contain missing value, it might not display the x-axis with actual string values correctly. However, we can't change the missing values into other type since lgbm handles NAN in a specified way. So for plot displayed without actual string values, please refer to the EncoderDict.



```python
def plot_embedding(self,embedding_values,y,title)

```

plot the embedding plot for the prediction or the ground truth

```markdown
Parameters
-------
embedding_values: array, n x 2 , storing the 2d information of the data
y: array, can be log odds of prediction  or the ground truth
title: string, title of the plot
        
Returns
-------
Embedding plot of prediction or ground truth

```



```python
def output_embedding_plot(self,shap_values,pred,y=None,top=3)

```

Output the embedding plot for prediction and the top 3 features

```markdown
Parameters
-------
shap_values: 2-d array, storing the shap values
pred: array, the prediction
y: array, the ground truth label, only used in testing but not real-world scenario
top: int, the number of features to be plotted
        
Returns
-------
Embedding plot for prediction and the top 3 features

```





```python
def Visualization(self,clf,X,X_display,y=None)

```

Output the all the plots relevant to SHAP in one shot

```markdown
Parameters
-------
clf: Light GBM model
X: DataFrame, the design matrix of the input
X_display: DataFrame, storing the display version of the input data 
y: array, the ground truth label, only used in testing but not real-world scenario

Returns
-------
Save the plot to the local files

```



## 6. Longitudinal Simulation

```python
def Simulation(self,train_window=10000,test_window=1000)

```

Perform backtesting to evaluate the model

```markdown
Parameters
-------
train_window: int, default 10000, amount of data selected for training
test_window: int, default 1000, amount of data selected for testing
        
Returns
-------
monitor: Dictionary, storing all the metrics in each epoch

```





## 

### Utility Function

```python
def bool2int(x)

```

Turn boolean into 0/1

```markdown
Parameters
-------
x: bool
        
Returns
-------
: int, 1: True, 0: False

```





```python
def IsWeekday(timestamp)

```

Check the timestamp to see whether it belongs to weekday

```markdown
Parameters
-------
timestamp: datetime, the timestamp of the feedback
        
Returns
-------
: int, 1: IsWeekday   0: Not Weekday

```





```python
def metrics_oneforall(label,pred)

```

Calculate the metrics accuracy,auc,f1,recall and precision

```markdown
Parameters
-------
label: array, the ground truth label
pred: array, the prediction
    
Returns
-------
accuracy,f1,auc,precision,recall: float, metrics for evaluation

```





```python
def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues)

```

plot to the confusion matrix to see the exact performance of the model

```markdown
Parameters
-------
y_true: array, the ground truth label
y_pred: array, the prediction
classes: list, assign the exact definitions of 0(non-actionable) and 1(actionable)
normalize: bool, whether to normalize the matrix
title: string, title of the plot
cmap: plt.cm, color mapping in matplotlib
    
Returns
-------
ax: plot, the confusion matrix

```



```python
def gen_train_val_test(data,cur_index,train_window,test_window)

```

Generate training, validation, testing data for simulation

```markdown
Parameters
-------
data: DataFrame, the data after preprocessing
cur_index: int, current indedx, the beginning index of training set
train_window: int, amount of data to used for training
test_window: int, amount of data to used for testing
        
Returns
-------
train,val,test: DataFrame, training,validation,test set
next_cur_index: int, updated current index
train_end_index: int, the ending index of training set

```



```python
def legend_to_top(ncol=4, x0=0.5, y0=1.2)

```

Set the legend on the top

```markdown
Parameters
-------
ncol: int, number of classes in legend
x0: float, horizontal place to adjust for the legend
y0: float, vertical place to adjust for the legend

```





