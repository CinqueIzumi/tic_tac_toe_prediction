##### this code cannot be run as a jupyter notebook
##### save this code to a file and run it using "anaconda prompt> streamlit run streamlit_demo.py"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import streamlit as st

@st.cache  # this function is executed for the 1st user request; for subsequent user requests, the cached function result is used
# please note that the code below is identical to the code of hands-on 4
def train_model():
    print("function train_model() is called (to verify the caching behavior of Streamlit)")

    # read the training set
    df = pd.read_csv('data/tic-tac-toe-endgame.csv')

    # Convert the values to numerical 
    df['conv_class'] = df['V10'].apply(replace_class_with_num)
    df['TL'] = df['V1'].apply(replace_value_with_num)
    df['TM'] = df['V2'].apply(replace_value_with_num)
    df['TR'] = df['V3'].apply(replace_value_with_num)
    df['ML'] = df['V4'].apply(replace_value_with_num)
    df['MM'] = df['V5'].apply(replace_value_with_num)
    df['MR'] = df['V6'].apply(replace_value_with_num)
    df['BL'] = df['V7'].apply(replace_value_with_num)
    df['BM'] = df['V8'].apply(replace_value_with_num)
    df['BR'] = df['V9'].apply(replace_value_with_num)
    
    # Split the data into a Training (70%) and a Test set (30%)
    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['conv_class'], random_state=42)
    
    featuresModelRF = ['MM', 'BM', 'MR', 'ML', 'TM']
    forest = RandomForestClassifier(n_estimators=200, random_state = 123)
    forest.fit(df_train[featuresModelRF], df_train['conv_class'])

    return forest

def replace_class_with_num(given_class):
    if(given_class == 'positive'):
        return 1
    else:
        return 0
    
def replace_value_with_num(given_value):
    if(given_value == 'x'):
        return 1
    elif(given_value == 'o'):
        return 0
    else:
        return -1
     
def compute_auroc(truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(truth, prediction, pos_label=1)
    return metrics.auc(fpr, tpr)

def plot_auroc(truth, predictions):
    fpr, tpr, threshold = metrics.roc_curve(truth, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
       
def apply_model(model, features, validation_function = compute_auroc, print_to_screen = True, show_auroc = False):
    pred_tree_train = model.predict_proba(df_train[features])
    pred_tree_test = model.predict_proba(df_test[features])
    
    pred_churn_tree_train = pd.Series(map(lambda x: x[1], pred_tree_train))
    pred_churn_tree_test = pd.Series(map(lambda x: x[1], pred_tree_test))
    
    validation_result_train = validation_function(df_train['conv_class'], pred_churn_tree_train) 
    validation_result_test = validation_function(df_test['conv_class'], pred_churn_tree_test)
    
    if(print_to_screen):
        print( "Result on trainset:" )
        print( validation_result_train )
        print()
        print( "Result on testset:" )
        print( validation_result_test )
    if(show_auroc):
        plot_auroc(df_test['conv_class'], pred_churn_tree_test)
    
    return (validation_result_train, validation_result_test)
     
def letter_to_float(given_letter):
    if(given_letter == 'x'):
        return 1
    elif(given_letter == 'o'):
        return 0
    else:
        return -1

def pred_to_winner(given_pred):
    if(given_pred == [1]):
        return 'x'
    else:
        return 'o'
        
st.title('Predict who will win a tic-tac-toe match (with an AUROC of 0.82 )') 

# train the model. This is done only once due to the Streamlit caching feature
regr = train_model()

# perform prediction
given_mm = (st.text_input("Enter the middle-middle square (x, o or /): ", '/'))
given_bm = (st.text_input("Enter the bottom-middle square (x, o or /): ", '/'))
given_mr = (st.text_input("Enter the middle-right square (x, o or /): ", '/'))
given_ml = (st.text_input("Enter the middle-left square (x, o or /): ", '/'))
given_tm = (st.text_input("Enter the top-middle square (x, o or /): ", '/'))

print(letter_to_float(given_mm), letter_to_float(given_bm), letter_to_float(given_mr), letter_to_float(given_ml), letter_to_float(given_tm))
pred = regr.predict([[letter_to_float(given_mm), letter_to_float(given_bm), letter_to_float(given_mr), letter_to_float(given_ml), letter_to_float(given_tm)]])

print(pred)

converted_pred = pred_to_winner(pred[0])
print(converted_pred)
st.text('The predicted winner is: ' + str(converted_pred))
st.text('Made by Bart Peereboom, 2139450')