import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import nltk
import pickle
import logging
import json
import difflib
import time
import lightgbm as lgb
import spacy,en_core_web_sm
import shap

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,make_scorer
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from shap.plots import colors

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from bayes_opt import BayesianOptimization
from sklearn.metrics.pairwise import cosine_similarity
from distutils.version import LooseVersion
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')



def bool2int(x):
    """
    Turn boolean into 0/1
        
    Parameters
    -------
    x: bool
        
    Returns
    -------
    : int, 1: True, 0: False
    """ 
    if x == True or x == "TRUE" or x == 1:
        return 1
    elif x == False or x== "FALSE" or x == 0:
        return 0
    else:
        return 0
        


        
def IsWeekday(timestamp):
    """
    Check the timestamp to see whether it belongs to weekday
        
    Parameters
    -------
    timestamp: datetime, the timestamp of the sample
        
    Returns
    -------
    : int, 1: IsWeekday   0: Not Weekday
    """ 
    day = timestamp.weekday()
    if 0 <= day <= 4:
        return 1
    else:
        return 0



def metrics_oneforall(label,pred):
    """
    Calculate the metrics accuracy,auc,f1,recall and precision
        
    Parameters
    -------
    label: array, the ground truth label
    pred: array, the prediction
    
    Returns
    -------
    accuracy,f1,auc,precision,recall: float, metrics for evaluation
    """ 
    # Calculate the accuracy score
    accuracy = metrics.accuracy_score(label, pred)
    print("The accuracy is: %f" % accuracy)
    
    # Calculate the f1 score
    f1 = metrics.f1_score(label, pred)
    print("The f1 is: %f" % f1)
    
    # Calculate the auc score
    auc = metrics.roc_auc_score(label, pred)
    print("The auc is: %f" % auc)
    
    # Calculate the precision score
    precision = metrics.precision_score(label,pred)
    print("The precision is: %f" % precision)
    
    
    # Calculate the recall score / true positive rate / sensitivity
    recall = metrics.recall_score(label,pred)
    print("The recall is: %f" % recall)
    
    tn,fp,fn,tp = confusion_matrix(label,pred).ravel()
    # Calculate the true negative rate / specificity
    true_negative_rate = tn / (tn + fp)
    print("The true negative rate is: %f" % true_negative_rate)
    
    # Calculate the false positive rate 
    false_positive_rate = fp / (fp + tn)
    print("The false positive rate is: %f" % false_positive_rate)
    
    # Calculate the false negative rate
    false_negative_rate = fn / (fn + tp)
    print("The false negative rate is: %f" % false_negative_rate)
    
    return accuracy,f1,auc,precision,recall,true_negative_rate,false_positive_rate,false_negative_rate
 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    plot to the confusion matrix to see the exact performance of the model
        
    Parameters
    -------
    y_true: array, the ground truth label
    y_pred: array, the prediction
    classes: list, assign the exact definitions of 0(negative) and 1(positive)
    normalize: bool, whether to normalize the matrix
    title: string, title of the plot
    cmap: plt.cm, color mapping in matplotlib
    
    Returns
    -------
    ax: plot, the confusion matrix
    """ 
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def gen_train_val_test(data,cur_index,train_window,test_window):
    """
    Generate training, validation, testing data for simulation
        
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
    """ 
    train_end_index = cur_index + train_window
    train = data.iloc[cur_index:train_end_index,:]
    train,val = train_test_split(train,test_size = test_window,random_state = 53)
    
    test_end_index = train_end_index + test_window
    test = data.iloc[train_end_index:test_end_index,:]
    next_cur_index = cur_index + test_window
    
    return train,val,test,next_cur_index,train_end_index

def legend_to_top(ncol=4, x0=0.5, y0=1.2):
    """
    Set the legend on the top
        
    Parameters
    -------
    ncol: int, number of classes in legend
    x0: float, horizontal place to adjust for the legend
    y0: float, vertical place to adjust for the legend
        
    """ 
    plt.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(x0, y0))



class LazyML(object):
    def __init__(self):
        """
        :param data:  dimension of data: N*D， with N samples and D features
        :param corpus: Corpus of sample text
        Initialize 
        """   

        self.corpus = None

        self.num_data = None
        self.num_feature = None
        self.model = None
        self.clf = None
        self.threshold = None
        self.opt_params = None

        self.categories = dict()
        self.delete_columns = None
        
        self.tfidf_vectorizer = None
        self.tfidf_features = None
        self.EncoderDict = None
        
        self.X = None
        self.y = None

        self.pred = None
        self.pred_adj = None
        

        
        self.feat_names = None
        
        self.params,self.params_path = self.get_params()  
        self.img_path = self.params['output_path'] + "/img/"
        self.pred_path = self.params['output_path'] + "/prediction/"
        if not os.path.exists(self.img_path):
            os.mkdir(self.img_path)
        if not os.path.exists(self.pred_path):
            os.mkdir(self.pred_path)

        #self.get_EncoderDict()
        
        self.cat_list = [''] # categorical variable list
		self.bin_list = [''] # binary variable list
		self.cont_list = [''] # continuous variable list
        self.num = '0123456789'
        stop_words = list(set(stopwords.words('english')))
        manual_stopwords = ['']  # manually defined stopwords
        stop_words.extend(manual_stopwords)
        self.stop_words = set(stop_words)
    
    ##################################################################
    #  ------>      Part 1.  Get data, parameters and necessary files  
    ##################################################################

    
    def get_params(self,params_path = "./config.json"):
        """
        Initialize the params for an isntance
        
        Parameters
        -------
        param path:  string, path storing the params
        
        Returns
        -------
        params: Dictionary, storing the path and the name of data
        """  
        with open(params_path,'r') as params_json:
            params = json.load(params_json)
        return params,params_path
    
    
    
    def get_EncoderDict(self):
        """
        Load the EncoderDict for the categorical variables
        """  
        file = open(self.params['data_path'] + "EncoderDict.pkl", 'rb')
        self.EncoderDict = pickle.load(file)
        file.close()
    
    def get_raw_data(self,raw_data_name=None):
        """
        Load the raw data queried by Kusto
        
        Parameters
        -------
        raw_data_name: string, file name of the raw data
        
        Returns
        -------
        raw_data: DataFrame, the data queried from the database
        """ 
        data_path = self.params['data_path'] 
        if raw_data_name is None:
            raw_data_name = self.params['raw_name']
        
        raw_data = pd.read_csv(data_path + raw_data_name)
        num_data,num_feature = raw_data.shape
        print("Capturing %d records with %d features from raw data..." % (num_data,num_feature))
        
        return raw_data
            
    def get_data(self,data_name=None,data_display_name=None,verbose = False):
        """
        Load the data and data_display after the preprocessing of the raw data
        
        Parameters
        -------
        data_name:  string, the file name of the data
        verbose: print out the number of data,features, minimum and maxmimum of timeslice
        
        Returns
        -------
        data: DataFrame, directly load the postprocessed data from local files
        data_display: DataFrame，directly load the postprocessed data_display from local files
        """
        
        data_path = self.params['data_path'] 
        if data_name is None:
            data_name = self.params['data_name']
            data_display_name = self.params['data_display_name']
        
        data = pd.read_csv(data_path + data_name,index_col=0)
        data_display = pd.read_csv(data_path + data_display_name,index_col=0)
        num_data,num_feature = data.shape
        print("Capturing %d records with %d features from cleaned data..." % (num_data,num_feature))
        
        
        if verbose == True:
            print("The number of data is %d " % num_data)
            print("The number of raw features is %d" % num_feature)
            print("The earliest date of review is %s " % min(data['TimeSlice']))
            print("The latest date of review is %s " %max(data['TimeSlice']))
            print("The missing values in each column are : ")
            print(data.isna().sum())
        
        return data,data_display
        
    def get_corpus(self):
        """
        Get the corpus for building the tfidf_vectorizer and save the vectorizer
        """
        corpus_path = self.params['data_path'] + self.params['corpus_name']
        corpus =  pd.read_csv(corpus_path,index_col=False,header=None)
        corpus.set_axis(['Id','State','TimeSlice','OriginalText','TranslatedText'],axis = 'columns',inplace=True)
        
        self.corpus = self.text_preprocessing(corpus)
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english",max_features=100)
        tfidf_corpus = self.tfidf_vectorizer.fit_transform(self.corpus.loc[:,'cleaned_text'])
        self.tfidf_features = self.tfidf_vectorizer.get_feature_names()
        
        # Save the tfidf_vectorizer to the path of model
        file = open(self.params['model_path'] + "tfidf_vectorizer.pkl", 'wb')
        pickle.dump(self.tfidf_vectorizer,file)
        file.close()
        
        print("The size of the corpus is: %d" % corpus.shape[0])
          
    def get_vectorizer(self):
        """
        Load the tfidf_vectorizer that has been trained
        """
        # Load the tfidf_vectorizer that has been trained from the path of model
        file = open(self.params['model_path'] + "tfidf_vectorizer.pkl",'rb') 
        self.tfidf_vectorizer = pickle.load(file)
        self.tfidf_features = self.tfidf_vectorizer.get_feature_names()
        file.close()
    
    

    ##################################################################
    #  ------>      Part 2.  Preprocessing 
    ##################################################################
    
    #          Pipeline
    def data_preprocessing(self,input_data,save=False,output_data=None,output_data_display=None):
        """
        Perform the pipeline for the preprocessing of the data from Database
        
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
        """
        data = input_data.copy()
        
        self.filter_na(data)
        self.binary_feat(data)
        self.drop_after_exam(data)
        data = self.text_preprocessing(data).copy()
        self.continuous_feat(data)
        self.Time_Process(data)
        
        if 'Target' in data.columns:
            self.target2label(data)
        data_display = data.copy()
        self.cate_feat(data,data_display)
        
        # round the continuous feature to 2 decimals in display dataframe
        continuous_feat = self.tfidf_features + ['# continuous features']
        decimals = pd.Series([2] * len(continuous_feat),index = continuous_feat)
        data_display = data_display.round(decimals).copy()
        
        data = data.drop(columns = 'TimeSlice').copy()

        if save:
            data.to_csv(self.params['data_path'] + output_data)
            data_display.to_csv(self.params['data_path'] + output_data_display)
        return data,data_display
        
    # Step 1: filter NA    
    def filter_na(self,data,na_threshold = 0.95):
        """
        Filter out the features with at least 95% missing values (Here I hard-code the columns derived from a larger dataset)
        
        Parameters
        -------
        data: DataFrame, the data used for filtering NAN, can be the training or the testing
        na_threshold: float, if # missing / # records > 0.95, remove this feature
        """
        
        if data is None:
            data = self.raw_data.copy()
            na_dict = dict(data.isna().sum())
            del_columns = []
            for key,value in na_dict.items():
                if value/self.num_data > na_threshold:
                    del_columns.append(key)

            self.del_columns = del_columns
            data.drop(columns = del_columns,inplace=True)
            
        else:
            data.drop(columns = self.del_columns,inplace=True)
        
        # For smaller dataset, directly use the result from Cosmos data analysis, 95% of these features are missing 
        # hard-code the columns to be deleted
        print("Filtering NAN features..")

          
    # Step 2: Binary feature
    def binary_feat(self,data):
        """
        Transforming the string into 0/1 for binary variables
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        """
        print("Transforming binary features...")
		for var in self.bin_list:
			data[var] = data[var].apply(lambda x: 0 if pd.isna(x) else 1)
       
        
        
    
    # Step 3: Remove some duplicated and imbalanced features
    def drop_after_exam(self,data):
        """
        Remove some duplicated and imbalanced features after manual examination
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        """
        print("Removing duplicated and imbalanced features..")
        rm_after_exam = ['feature names of the duplicated and imbalance features']
        data.drop(columns=rm_after_exam,inplace=True)
        
    
    # Step 4: Text to Vector
    def text_preprocessing(self,data):
        """
        Clean the text from user and turn it into tfidf vector
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        
        Returns
        -------
        data: DataFrame, the raw data to be cleaned
        """
        print("Text Preprocessing..")
        nlp = en_core_web_sm.load(disable = ['parser','ner'])
        
        
        
        data = data[(~data['Text'].duplicated()) | (data['Text'].isna())].copy()
        data.loc[:,'Text'].fillna('',inplace=True)
        data.loc[:,'Text'] = data.Text.apply(lambda x: x.lower())
        data.loc[:,'Text'] = data.Text.apply(lambda x: ''.join([c for c in x if c not in punctuation]))
        data.loc[:,'Text'] = data.Text.apply(lambda x: ''.join([c for c in x if c not in self.num]))
        data.loc[:,'cleaned_text'] = data.Text.apply(lambda x: word_tokenize(x))
        data.loc[:,'cleaned_text'] = data.cleaned_text.apply(lambda x: ' '.join([w for w in x if w not in self.stop_words]))
        data.loc[:,'cleaned_text'] = data.cleaned_text.apply(lambda x: nlp(x))
        data.loc[:,'cleaned_text'] = data.cleaned_text.apply(lambda x: [token.lemma_ for token in x])
        data.loc[:,'cleaned_text'] = data.cleaned_text.apply(lambda x: ' '.join(x))
        data = data.drop(columns='Text').copy()
        
        data_corpus = pd.DataFrame(self.tfidf_vectorizer.transform(data.loc[:,'cleaned_text']).todense(),columns=self.tfidf_features)
        data_corpus.index = data.index.copy()
        data = pd.concat([data,data_corpus],axis=1).copy()

        return data
    
    # Step 5: Continuous Variables
    def continuous_feat(self,data):
        """
        Take log transformation to stablize the continuous features
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        """
        print("Transforming continuous features..")
        for var in self.cont_list:
			data['Log('+var+')'] = np.log(pd.to_numeric(data[var]) + 1)
			data.drop(columns=var,inplace=True)
        
    # Step 6: Features relevant to time
    def Time_Process(self,data):
        """
        Generate feature from TimeSlice including "IsWeekday" 
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        """
        print("Transforming time-related features..")
        data['TimeSlice'] = pd.to_datetime(data['TimeSlice'],errors='coerce')
        data.sort_values(by = 'TimeSlice',inplace=True)
        data.reset_index(inplace=True)
        data.drop(columns='index',inplace=True)
        data['IsWeekday'] = data['TimeSlice'].apply(lambda x: IsWeekday(x))

    
    # Step 7: Categorical Variables
    def cate_feat(self,data,data_display):
        """
        Encode the categorical features by default encoder dictionary EncoderDict
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        data_display: DataFrame, the postprocessed data_display after cleaning
        """
        
        print("Transforming categorical features..")
        
        for cat in self.cat_list:
            lb_ = self.EncoderDict[cat]
            #null_mask = pd.isnull(data[cat])
            unnull_mask = ~pd.isnull(data[cat])
            #data.loc[null_mask,cat] = "Unknown"
            #data_display.loc[null_mask,cat] = "Unknown"
            #data.loc[:,cat] = lb_.transform(data[cat])
            data.loc[unnull_mask,cat] = lb_.transform(data.loc[unnull_mask,cat])
            data.loc[:,cat] = data.loc[:,cat].astype('category')

        

    # Step 8: Turn State into Label
    def target2label(self,data):
        """
        Turn state into the label for classification model
        
        Parameters
        -------
        data: DataFrame, the raw data to be cleaned
        """
     
        # Transforming state into label
        print("All the states in the dataset: ")
        print(data.groupby('Target').count().iloc[:,0])
        print("\n")
        
        data['label'] = data['Target'].apply(lambda x: 0 if x in [''] else 1)
        data.drop(columns = ['Target'],inplace=True)
        
        print("Distribution of the labels: ")
        print(data.groupby('label').count().iloc[:,0])
        print("\n")
        
    
    ##################################################################
    #  ------>      Part 3.  Training and Predicting
    ##################################################################           
    
    
    def optimize(self,train,val):
        """
        Use Bayesian Optimization to optimize the parameters for the light gbm
        
        Parameters
        -------
        train: DataFrame, the training data
        val: DataFrame, the validation data
        
        Returns
        -------
        opt_params： Dictionary, storing the optimized paramters
        """
        pbounds = {
        'learning_rate': (0.01,0.1),
        'num_leaves': (24,45),  # we should let it be smaller than 2^(max_depth)
        'max_depth': (5,8.99),  # -1 means no limit
        'min_child_samples': (10,20),  # Minimum number of data need in a child(min_data_in_leaf)
        'feature_fraction': (0.1,0.9),
        'bagging_fraction': (0.8,1),
        'min_split_gain': (0.001,0.1),
        'min_child_weight': (5,50),
        'reg_alpha': (0,5),  # L1 regularization term on weights
        'reg_lambda': (0,3),  # L2 regularization term on weights
        'threshold': (0.1,0.8)
        }     
        
        
        def get_F1(learning_rate,num_leaves,max_depth,min_child_samples,feature_fraction,bagging_fraction,min_split_gain,
          min_child_weight,reg_alpha,reg_lambda,threshold):
            """
            Return a F1 score on validation set after training
        
            Parameters
            -------
            All the parameters are from Light GBM
        
            Returns
            -------
            f1： float, the f1 score of validation set
            """
            tmp_params  = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric':'auc',
            'nthread': 8,
            'verbose': 0
            }
        
            tmp_params['learning_rate'] = learning_rate
            tmp_params['num_leaves'] = int(num_leaves)  # we should let it be smaller than 2^(max_depth)
            tmp_params['max_depth'] = int(max_depth)  # -1 means no limit
            tmp_params['min_child_samples'] = int(min_child_samples)  # Minimum number of data need in a child(min_data_in_leaf)
            tmp_params['feature_fraction'] = feature_fraction
            tmp_params['bagging_fraction'] = bagging_fraction
            tmp_params['min_split_gain'] = min_split_gain
            tmp_params['min_child_weight'] = min_child_weight
            tmp_params['reg_alpha'] = reg_alpha
            tmp_params['reg_lambda']= reg_lambda
            
            evals_result = {}
            tmp_gbm,val_X,val_y = self.training(train,val,tmp_params,return_val=True)
            
            pred_val,pred_val_b = self.predict(val_X,tmp_gbm,threshold)

            f1 = metrics.fbeta_score(val_y, pred_val_b,beta=2)
        
            return f1
        
        
        
        start = time.time()
        print("Optimization...")
        optimizer = BayesianOptimization(f=get_F1,pbounds=pbounds,random_state=53,verbose=0)
        optimizer.maximize(init_points=10,n_iter=10)
        opt_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric':('f1','auc','binary_error'),
            'learning_rate': optimizer.max['params']['learning_rate'],
            'bagging_fraction': optimizer.max['params']['bagging_fraction'],  # we should let it be smaller than 2^(max_depth)
            'feature_fraction': optimizer.max['params']['feature_fraction'],  # we should let it be smaller than 2^(max_depth)
            'max_depth': int(optimizer.max['params']['max_depth']),  # -1 means no limit
            'min_child_samples': int(optimizer.max['params']['min_child_samples']),  # Minimum number of data need in a child(min_data_in_leaf)
            'min_child_weight': optimizer.max['params']['min_child_weight'],  # Minimum number of data need in a child(min_data_in_leaf)
            'min_split_gain': optimizer.max['params']['min_split_gain'],  # Minimum number of data need in a child(min_data_in_leaf)
            'num_leaves': int(optimizer.max['params']['num_leaves']),  # we should let it be smaller than 2^(max_depth)
            'threshold': optimizer.max['params']['threshold'],
            'reg_alpha': optimizer.max['params']['reg_alpha'],  # L1 regularization term on weights
            'reg_lambda': optimizer.max['params']['reg_lambda'],  # L2 regularization term on weights
            'nthread': 8,
            'verbose': 0,
        }
        print("Time Elapased: %f" % (time.time() - start))
        self.opt_params = opt_params
        return opt_params
    
    def data_split(self,data):
        """
        Split the data into design matrix X and label y
        
        Parameters
        -------
        data: DataFrame, the data to be split
        
        Returns
        -------
        X： DataFrame, the design matrix for training or prediction
        y: Array, the ground truth label
        """
        if 'Id' in data.columns:
            X = data.drop(columns='Id').copy()
        if 'label' in data.columns:
            X = X.drop(columns='label').copy()
            y = data['label']
            return X,y
        else:
            return X
    
    def train_val_test_split(self,data,data_display,test_size):
        """
        Split the training data into new training and validation data for tuning the parameters
        
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
        """
        train,test = train_test_split(data,test_size = test_size,shuffle=False)
        train_display = data_display.loc[train.index,:].copy()
        test_display = data_display.loc[test.index,:].copy()
        
        new_train,new_val = train_test_split(train,test_size = test_size,random_state = 53)
        new_train_display = train_display.loc[new_train.index,:].copy()
        new_val_display = train_display.loc[new_val.index,:].copy()
        
        return new_train,new_val,test,new_train_display,new_val_display,test_display
    
    
    
    
    def training(self,train,val,params,return_val=False,verbose=False):
        """
        Train the data with Light GBM
        
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
        """
        
        print("Training...")
        start = time.time()
        evals_result = {}
            
        train_X,train_y = self.data_split(train)
        
        val_X,val_y = self.data_split(val)
        
        lgb_train = lgb.Dataset(train_X,label=train_y)
        lgb_val = lgb.Dataset(val_X,label=val_y)
            
        tmp_gbm = lgb.train(params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train','val'],
            #feature_name=list(train_X.columns),
            categorical_feature=self.cat_list,
            evals_result=evals_result,
            early_stopping_rounds= 200,verbose_eval=False)
        print("Time Elapased: %f" % (time.time() - start))
        if return_val:
            return tmp_gbm,val_X,val_y
        else:
            return tmp_gbm
    
    def predict(self,X,clf,cutoff):
        """
        Make prediction by Light GBM
        
        Parameters
        -------
        X: DataFrame, the design matrix
        clf: Light GBM model, the classifier
        cutoff: float, the thresholid to turn the probability into binary prediction
        
        Returns
        -------
        pred： Array, the probability of actionability
        pred_b: Array, binary prediction of actionability
        """
        print("Predicting...")
        # actionability
        pred = clf.predict(X,num_iteration=clf.best_iteration)
        # binary prediction
        pred_b = np.where(pred > cutoff,1,0)
        return pred,pred_b
    
    
    


    ##################################################################
    #  ------>      Part 4.  Save & Load Model
    ##################################################################     
    def SaveModel(self,clf,name):
        """
        Save the model to the local files
        
        Parameters
        -------
        clf: Light GBM model, the classifier
        name: string, the name of these model
        
        Returns
        -------
        Save the model to the local files
        """
        
        file = open(self.params['model_path'] + name + ".pkl", 'wb')
        pickle.dump(clf,file)
        file.close()
        
        print(name + " has been saved.")
        
    def LoadModel(self,name):
        """
        Load the model to the local files
        
        Parameters
        -------
        name: string, the name of the model
        
        Returns
        -------
        clf: Light GBM model, the classifier
        """
        file = open(self.params['model_path'] + name + ".pkl",'rb')
        clf = pickle.load(file)
        file.close()
        print("The model has been loaded.")
        
        return clf
        

    ##################################################################
    #  ------>      Part 5.  Visualization (SHAP)
    ##################################################################     
    
    def performance(self,y,pred):
        """
        combine the metrics output and the confusion matrix
        
        Parameters
        -------
        y: array, ground truth label
        pred: array, prediction
        
        Returns
        -------
        print out the metrics and plot the confusion matrix
        """ 
        print("The performance on the data is:")
        accuracy,f1,auc,precision,recall,true_negative_rate,false_positive_rate,false_negative_rate = metrics_oneforall(y,pred)
        metrics = pd.DataFrame([[accuracy,f1,auc,precision,recall,true_negative_rate,false_positive_rate,false_negative_rate]],columns=['accuracy','f1','auc','precision','recall','true_negative_rate','false_positive_rate','false_negative_rate'],index=['metrics'])
        metrics.to_csv(self.pred_path + 'metrics.csv',index=True)
        
        if not os.path.exists(self.img_path + "summary_plot/"):
            os.mkdir(self.img_path + "summary_plot/") 
        plot_confusion_matrix(y,pred,classes = np.array(["Negative","Positive"]))
        plt.savefig(self.img_path + "summary_plot/"+"ConfusionMatrix.png",bbox_inches='tight')
        plt.close()
    
    def CalShap(self,clf,X):
        """
        Calculate the shap values for the input data given the classification model
        
        Parameters
        -------
        clf: Light GBM model
        X: DataFrame, the design matrix of the input
        
        Returns
        -------
        explainer: tree explainer module
        shap_values: 2-d array, shap values per sample per feature
        """
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        self.feat_names = list(X.columns)
        return explainer,shap_values
        
    def output_force_plot(self,explainer,shap_values,X_display):
        """
        Output the force plots of each sample
        
        Parameters
        -------
        explainer: tree explainer module
        shap_values: 2-d array, storing the shap values
        X_display: DataFrame, storing the display version of the input data 
        
        Returns
        -------
        Save the force plot to the local files
        """
        print("Output the force plot(Warning: This might take 0.5-1 hour)...")
        if not os.path.exists(self.img_path + "force_plot/"):
            os.mkdir(self.img_path + "force_plot/")
        Id = X_display['Id']
        X_display = X_display.drop(columns = ["Id",'TimeSlice']).copy()
        if 'label' in X_display.columns:
            X_display = X_display.drop(columns = "label").copy()
        
        for i in range(len(X_display)):
            tmp_id = Id.iloc[i]
            shap.force_plot(explainer.expected_value, shap_values[i,:], X_display.iloc[i,:],link='logit',matplotlib=True,show=False)
            plt.savefig(self.img_path + "force_plot/" + tmp_id + ".png",bbox_inches='tight')
            plt.close()
    
    def output_summary_plot(self,shap_values,X,top=10):
        """
        Output the summary plots of feature importance
        
        Parameters
        -------
        shap_values: 2-d array, storing the shap values
        X: DataFrame, storing the design matrix of the input data 
        top: int, top n important features 
        
        Returns
        -------
        Save the summary plot to the local files
        """
        print("Output the summary plot...")
        if not os.path.exists(self.img_path + "summary_plot/"):
            os.mkdir(self.img_path + "summary_plot/")
        shap.summary_plot(shap_values, X,max_display=top,show=False)
        plt.savefig(self.img_path + "summary_plot/" + "summary_plot.png",bbox_inches='tight')
        plt.close()
        shap.summary_plot(shap_values, X,max_display=top,show=False,plot_type="bar")
        plt.savefig(self.img_path + "summary_plot/" + "summary_plot_bar.png",bbox_inches='tight')
        plt.close()
    
    def output_dependence_plot(self,shap_values,X,X_display,top = 20):
        """
        Output the dependence plots of each important feature
        
        Parameters
        -------
        shap_values: 2-d array, storing the shap values
        X: DataFrame, storing the design matrix of the input data 
        X_display: DataFrame, storing the display version of the input data 
        top: int, top n important features 
        
        Returns
        -------
        Save the dependence plot to the local files
        """
        print("Output the dependence plot...")
        if not os.path.exists(self.img_path + "dependence_plot/"):
            os.mkdir(self.img_path + "dependence_plot/")
        top_n_feat = [self.feat_names[i] for i in np.argsort(-np.abs(shap_values).mean(0))[:top]]
        
        X_display = X_display.drop(columns = ["Id",'TimeSlice']).copy()
        if 'label' in X_display.columns:
            X_display = X_display.drop(columns = "label").copy()
        
        for i,feat in enumerate(top_n_feat):
            shap.dependence_plot(feat, shap_values, X,display_features=X_display,show=False)
            plt.savefig(self.img_path + "dependence_plot/" + str(i + 1) + "_" + feat + ".png",bbox_inches='tight')
            plt.close()
            print(feat + " Finished")
    
    def plot_embedding(self,embedding_values,y,title):
        """
        plot the embedding plot for the prediction or the ground truth
        
        Parameters
        -------
        embedding_values: array, n x 2 , storing the 2d information of the data
        y: array, can be log odds of prediction  or the ground truth
        title: string, title of the plot
        
        Returns
        -------
        Embedding plot of prediction or ground truth
        """
        
        plt.scatter(embedding_values[:,0], embedding_values[:,1], c=y,
        cmap=colors.red_blue, alpha=1.0, linewidth=0)
        
        plt.axis("off")
        #pl.title(feature_names[ind])
        cb = plt.colorbar()
        cb.set_label(title, size=13)
        cb.outline.set_visible(False)
        plt.gcf().set_size_inches(7.5, 5)
        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 10)
        cb.set_alpha(1)

    
    def output_embedding_plot(self,shap_values,pred,y=None,top=3):
        """
        Output the embedding plot for prediction and the top 3 features
        
        Parameters
        -------
        shap_values: 2-d array, storing the shap values
        pred: array, the prediction
        y: array, the ground truth label, only used in testing but not real-world scenario
        top: int, the number of features to be plotted
        
        Returns
        -------
        Embedding plot for prediction and the top 3 features
        """
        print("Output the embedding plot...")
        if not os.path.exists(self.img_path + "embedding_plot/"):
            os.mkdir(self.img_path + "embedding_plot/")
        
        # output the log odds embedding plot
        log_odds = np.log(pred / (1 - pred))
        pca = PCA(2)
        embedding_values = pca.fit_transform(shap_values)
        self.plot_embedding(embedding_values,log_odds,"2D Log Odds")
        plt.savefig(self.img_path + "embedding_plot/" + "Test Log Odds.png")
        plt.close()
        if y is not None:
            self.plot_embedding(embedding_values,y,"2D Ground Truth")
            plt.savefig(self.img_path + "embedding_plot/" + "Test Ground Truth.png")
            plt.close()
        
        
        # output the top 3 embedding plots
        top_n_feat = [self.feat_names[i] for i in np.argsort(-np.abs(shap_values).mean(0))[:top]]
        for i in range(top):
            shap.embedding_plot("rank("+str(i)+")",shap_values,self.feat_names,method="pca", alpha=1.0, show=False)
            plt.savefig(self.img_path + "embedding_plot/" + str(i+1) + "_" + top_n_feat[i] + ".png")
            plt.close()
        
        
        
    
    def Visualization(self,clf,X,X_display,y=None):
        """
        Output the all the plots relevant to SHAP in one shot
        
        Parameters
        -------
        clf: Light GBM model
        X: DataFrame, the design matrix of the input
        X_display: DataFrame, storing the display version of the input data 
        y: array, the ground truth label, only used in testing but not real-world scenario
        
        Returns
        -------
        Save the plot to the local files
        """
        print("Start Visualization...")
        explainer,shap_values = self.CalShap(clf,X)
        
        pred = clf.predict(X,num_iteration=clf.best_iteration)
        self.output_embedding_plot(shap_values,pred,y)
        self.output_summary_plot(shap_values,X)
        self.output_dependence_plot(shap_values,X,X_display)
        self.output_force_plot(explainer,shap_values,X_display)
    

    
    ##################################################################
    #  ------>      Part 6.  Longitudinal Simulation
    ##################################################################
    def Simulation(self,train_window=10000,test_window=1000):
        """
        Perform backtesting to evaluate the model
        
        Parameters
        -------
        train_window: int, default 10000, amount of data selected for training
        test_window: int, default 1000, amount of data selected for testing
        
        Returns
        -------
        monitor: Dictionary, storing all the metrics in each epoch
        """
        #data,data_display = self.get_data()
        raw = self.get_raw_data()
        data,data_display = self.data_preprocessing(raw)
        current_index = 0
        len_data = data.shape[0]
        
        monitor = {}
        monitor['time'] = []
        monitor['accuracy'] = []
        monitor['auc'] = []
        monitor['f1'] = []
        monitor['recall'] = []
        monitor['precision'] = []
        monitor['true_negative_rate'] = []
        monitor['false_positive_rate'] = []
        monitor['false_negative_rate'] = []
        
        monitor['pos_num'] = []
        monitor['pos_percent'] = []
        
        while True:
            print("Current Index is : %d" % current_index)
            train,val,test,current_index,train_end_index = gen_train_val_test(data,current_index,train_window,test_window)
            if train_end_index > len_data:
                print("The beginning of test exceeds length of dataset!")
                break
            

            test_X,test_y = self.data_split(test)
            pos_num = np.sum(test_y == 1)
            pos_percent = pos_num / len(test_y)
            print("Percentage of positive sample in test set: %f" % pos_percent)
    
            if len(set(test_y)) != 2:
                print("Ground truth test doesn't have 2 labels in this round.Skipped!")
                continue
    
            opt_params = self.optimize(train,val)
    
            # Training
            gbm = self.training(train,val,opt_params)
            
            best_cutoff = opt_params['threshold']
            print("Best cutoff: %f" % best_cutoff)
            
            # Prediction
            test_pred,test_pred_b = self.predict(test_X,gbm,best_cutoff)
    
            # Evaluation
            print("Evaluation metric on test set for this round:")
            accuracy,f1,auc,precision,recall,true_negative_rate,false_positive_rate,false_negative_rate = metrics_oneforall(test_y,test_pred_b)
            test_begin_time = data_display['TimeSlice'].iloc[train_end_index]
            monitor['time'].append(test_begin_time)
            monitor['accuracy'].append(accuracy)
            monitor['f1'].append(f1)
            monitor['auc'].append(auc)
            monitor['precision'].append(precision)
            monitor['recall'].append(recall)
            monitor['true_negative_rate'].append(true_negative_rate)
            monitor['false_positive_rate'].append(false_positive_rate)
            monitor['false_negative_rate'].append(false_negative_rate)
            monitor['pos_percent'].append(pos_percent)
            monitor['pos_num'].append(pos_num)
            plot_confusion_matrix(test_y,test_pred_b,classes = np.array(["Negative","Positive"]))
            print("\n")
            
        if not os.path.exists(self.img_path + "simulation_plot/"):
            os.mkdir(self.img_path + "simulation_plot/")    
        monitor_df =  pd.DataFrame(monitor).round(3)
        monitor_df.to_csv(self.pred_path + "Simulation.csv",index=False)
        monitor_df.plot(x='time',y=['f1','recall','precision','true_negative_rate','false_positive_rate','false_negative_rate','pos_percent'],kind='bar')
        legend_to_top()
        plt.savefig(self.img_path + "simulation_plot/"+"Simulation.png",bbox_inches='tight')
        plt.close()    

        return monitor
    
   
    
        
    
 