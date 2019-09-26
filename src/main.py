import LML
from LML import LazyML

def main():
    # set up an instance
    lml_ = LazyML()

    # Get the Encoder Dictionary for the categorical variable
    lml_.get_EncoderDict()
    # Get the tfidf vectorizer for turning word into vector
    lml_.get_vectorizer()
    
    # Load the ocv data queried from Kusto
    raw_data = lml_.get_raw_data(lml_.params['data_path'] + lml_.params['raw_name'])
   
    
    # Perform preprocessing on train and test
    data,data_display = lml_.data_preprocessing(raw_data)
    
    
    # Train-Validation Split
    train,val,test,train_display,val_display,test_display = lml_.train_val_test_split(data,data_display,test_size = lml_.params['test_size'])

    # Bayesian Optimization
    opt_params = lml_.optimize(train,val)
    

    # Final Training
    gbm = lml_.training(train,val,opt_params)

    # Testing and predicitng
    #test_X,test_y = lml_.data_split(test)   ###### When using in real-world data, mute this line
    test_X= lml_.data_split(test)
    test_pred,test_pred_b = lml_.predict(test_X,gbm,opt_params['threshold'])

    # Evaluation: return metrics and confusion matrix
    #lml_.performance(test_y,test_pred_b)    ###### When using in real-world data, mute this line

    # Real-world Scenario, can't evaluate and directly output the actionability score
    if 'label' in test.columns:
        test = test.drop(columns = "label").copy()
    
    lml_.output_action_set(test,gbm,opt_params['threshold'])
    
    # Save the model
    lml_.SaveModel(gbm,"lgbm")

    # Load the model
    gbm = lml_.LoadModel("lgbm")
    
    # print out the best cutoff
    best_cutoff = opt_params['threshold']
    print("Best cutoff: %f" % best_cutoff)
    
    # Output the force plot, summary plot and dependence plot to the local files
    #lml_.Visualization(gbm,test_X,test_display,test_y) ###### When using in real-world data,remove ",test_y"
    lml_.Visualization(gbm,test_X,test_display) ###### When using in real-world data,remove ",test_y"
    

if __name__ == "__main__":
    main()
	

