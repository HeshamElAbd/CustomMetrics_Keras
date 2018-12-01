# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:34:26 2018

@author: Hesham El Abd

@description: A module to compute classification metrics for binary and 
multi-class classification models that are build and trained using Keras.

@aim: To provide a detailed  information regard the 
performance of a classification model during training by measuring its 
performance on each batch.
 
*** The functions in this module operate in a batch-wise manner, 
meaning, that the values they generate on the validation set 
will be the average of all the batches used.
***  
"""
# imporing modules 
import keras.backend as k 
import numpy as np 
from sklearn.metrics import confusion_matrix
import tensorflow as tf
###############################################################################
def binary_recall(y_true,y_pred):
    """
    A function to compute the recall or Sensitivity or the true positive rate 
    (TPR) for a binary test using the following formula: 
    TPR=TP/(TP+FN), where TP is the True Positives and FN is the False Negative.
    
    # inputs:
    y_true= a tesnor of size(batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction, a 
    float which ranges from 0 to 1. 
    
    # output: 
    TPR, a scaler 
    """
    true_positive=k.sum(y_true*k.round(y_pred))
    true_positive_and_flase_negative=k.sum(y_true) # equall TP+FN 
    return true_positive/(true_positive_and_flase_negative+k.epsilon())  

def binary_specificity (y_true,y_pred):
    """
    A function to compute the Specificity or the true negative rate (TPR) for 
    a binary test using the following formula: 
    TNR=TN/(TN+FP), where TN is the True Negatives 
    and FP is the False Positives.
    
    # inputs:
    y_true= a tesnor of size (batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction,a 
    float which ranges from 0 to 1. 
    
    # output: 
    specificity, a scaler 
    """
    true_negative=k.sum(
            k.cast(k.not_equal(y_true,1),dtype='int32')*
            k.cast(k.not_equal(k.round(y_pred),1),dtype='int32'))
    
    true_negative_and_false_positive=k.sum(k.cast(k.not_equal(y_true,1),
                                                  dtype='int32'))
    return true_negative/(true_negative_and_false_positive+k.epsilon())

def binary_precision(y_true,y_pred):
    """
    A function to compute the percision of a binary test using the following 
    formula: 
        percision=TP/TP+FP, where TP is the True Positives 
    and FP is the False Positive.
    
    # inputs:
    y_true= a tesnor of size(batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction, a 
    float which ranges from 0 to 1. 
    
    # output: 
    percision, a scaler 
    """
    true_positive=k.sum(y_true*k.round(y_pred))
    true_positives_and_false_positives=k.sum(k.round(y_pred))# equals FP+TP
    return true_positive/(true_positives_and_false_positives+k.epsilon())

def binary_flase_positive_rate(y_true,y_pred):
    """
    A function to compute the false positive rate in a batch wise
    manner for a binary test using 
    the following formula: 
    False Positive Rates=FP/FP+TN
    where FP is False positive and TN is the true negative. 
    # inputs:
    y_true= a tesnor of size(batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction, a 
    float which ranges from 0 to 1. 
    
    # output: 
    False Positive Rate (FPR), a scaler 
    """
    false_positives=k.sum(k.round(y_pred)*
                         k.cast(k.not_equal(y_true,1),dtype='int32'))
    
    false_positives_and_true_negatives=k.sum(k.cast(k.not_equal(y_true,1),dtype='int32'))
    return false_positives/(false_positives_and_true_negatives+k.epsilon())

def binary_false_negative_rate(y_true,y_pred): 
    """
    A function to compute the fasle negative Rate for a binary test using 
    the following formula: 
        FNR=FN/TP+FN where FN is false negative and TP is the true positives
    # inputs:
    y_true= a tesnor of size(batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction
    
    # output: 
    False Negative Rate (FPR), a scaler 
    """
    false_negatives=k.sum(y_true*
                         k.cast(k.not_equal(k.round(y_pred),1),dtype='int32'))
    true_positive_and_false_negative=k.sum(y_true)
    return false_negatives/(true_positive_and_false_negative+k.epsilon())

def binary_f1_score(y_true,y_pred): 
    """
    A function to compute the F1 scores for a binary test using 
    the following formula: 
        F1=2*Precision*Recall/(Precision + Recall)
        
     # inputs:
    y_true= a tesnor of size(batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction, a 
    float which ranges from 0 to 1. 
    
    # output: 
    False Negative Rate (FPR), a scaler 
    """
    recall=binary_recall(y_true,y_pred)
    precision=binary_precision(y_true,y_pred)
    return 2*precision*recall/(precision + recall + k.epsilon())

def binary_accuracy(y_true,y_pred): 
    """
    A function to compute the accuracy for a binary test using 
    the following formula: 
        Accuracy=TP+TN/(P+N) where TP is the true positive and TN is the true 
    negative. 
        
     # inputs:
    y_true= a tesnor of size(batch_size,1) containing the correct binary labels
    y_pred= a tensor of size (batch_size,1) containing the model prediction, a 
    float which ranges from 0 to 1. 
    
    # output: 
    the Accuracy, a scaler 
    """
    return k.mean(k.equal(y_true,k.round(y_pred)))

def macro_multiClass_recall_py(y_true,y_pred):
    """
    A Python function to compute the average macro multiClass recall which is 
    the average of the recall vector, which contian the recall for each class.
    the dimension of the recall vector is the number of classes where the ith 
    element represent the recall for the ith class. The function work by first
    constructing the confusion matrix using the "confusion_matrix" function
    from sklearn.metrics modules. Next, the recall vector is constructed by 
    dividing the diagonal of the confusion matrix which represent the 
    number of true positives by the col sum of the confusion matrix. Finally,
    the macro recall is computed by taking the mean of the recall vector. 

    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes
    
    # outputs: 
        The average macro multiClass recall, a scaler. 
    """
    conf_matrix=confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    all_positive=np.sum(conf_matrix,axis=0)
    true_positive=np.diag(conf_matrix)
    recall=true_positive/(all_positive+1e-7) # to avoid dividing by zero
    return np.mean(recall)

def macro_multiClass_recall(y_true,y_pred):
    """
    a wrapper function for the function macro_multiClass_recall_py it feeds the
    function to the function tf.py_func, making it a tensorflow op. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The macro multiClass recall, a scaler
    """
    return tf.py_func(macro_multiClass_recall_py,(y_true,y_pred),tf.double,
                      name='Macro_multiClass_Recall')
    
def macro_multiClass_precision_py(y_true,y_pred):
    """
    A Python function to compute the average macro multiClass precision which 
    is the average of the precision vector, which contian the precision for 
    each class. The dimension of the precision vector is the number of classes 
    where the ith element represent the precision for the ith class. 
    The function work by first constructing the confusion matrix using the 
    "confusion_matrix" function from sklearn.metrics modules. Next, the
    precision vector is constructed by dividing the diagonal of the confusion 
    matrix which represent the number of true positives by the row sum of the
    confusion matrix. Finally, the micro precision is computed by taking the
    mean of the precision vector.

    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes
    
    # outputs: 
        The macro multiClass precision, a scaler
    """
    conf_matrix=confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    all_predicted_perclass=np.sum(conf_matrix,axis=1)
    true_positive=np.diag(conf_matrix)
    precision=true_positive/(all_predicted_perclass+1e-7) # to avoid dividing by zero
    return np.mean(precision)

def macro_multiClass_precision(y_true,y_pred):
    """
    a wrapper function for the function macro_multiClass_precision_py, it feeds 
    the function to the function tf.py_func, making it a tensorflow op. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The macro multiClass precision, a scaler
    """
    return tf.py_func(macro_multiClass_precision_py,(y_true,y_pred),tf.double,
                      name='Macro_multiClass_precision')
    
def macro_f1_multiClass_py(y_true,y_pred):
    """
    A function to compute the macro F1 scores for a multi class classifier. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The  macro multiClass F1scores, a scaler
    """
    conf_matrix=confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    all_goldTrue_positives=np.sum(conf_matrix,axis=0)
    all_predicted_positives=np.sum(conf_matrix,axis=1)
    true_positive=np.diag(conf_matrix)
    recall_vector=true_positive/(all_goldTrue_positives+1e-7) 
    precession_vector=true_positive/(all_predicted_positives+1e-7)
    return np.mean(2*recall_vector*precession_vector/(recall_vector+
                                                      precession_vector+
                                                      k.epsilon()))

def macro_f1_multiClass(y_true,y_pred):
    """
    a wrapper function for the function micro_f1_multiClass_py, it feeds 
    the function to the function tf.py_func, making it a tensorflow op. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels.
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The micro multiClass F1 scores, a scaler
    """
    return tf.py_func(micro_multiClass_recall_py,(y_true,y_pred),tf.double,
                      name='Macro_f1_scores')

def micro_multiClass_recall_py(y_true,y_pred):
    """
    A Python function to compute the macro average multiClass recall according to the
    follwing formula:
    micro_recall=S(TP)/(S(TP)+S(FN), where S is the summation function, TP is 
    the true positives vector or the diagonal of the confusion matrix. 
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes
    
    # outputs: 
        The micro multiClass recall, a scaler. 
    """
    conf_matrix=confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    all_positive=np.sum(conf_matrix,axis=0)
    true_positive=np.diag(conf_matrix)
    recall=sum(true_positive)/(sum(all_positive)+1e-7) # to avoid dividing by zero
    return recall
    
def micro_multiClass_recall(y_true,y_pred):
    """
    a wrapper function for the function micro_multiClass_recall_py, it feeds 
    the function to the function tf.py_func, making it a tensorflow op. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The micro multiClass recall, a scaler
    """
    return tf.py_func(micro_multiClass_recall_py,(y_true,y_pred),tf.double,
                      name='Micro_multiClass_Recall')
    
def micro_multiClass_precision_py(y_true,y_pred):
    """
    A Python function to compute the micro average multiClass precision 
    according to the following formula: 
        micro_precision=S(TP)/(S(TP)+S(FP)) where S is the summation function 
    and TP is the true positives vector or the diagonal of the confusion matrix. 

    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes
    
    # outputs: 
        The micro multiClass precision, a scaler
    """
    conf_matrix=confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    all_predicted_perclass=np.sum(conf_matrix,axis=1)
    true_positive=np.diag(conf_matrix)
    precision=sum(true_positive)/(sum(all_predicted_perclass)+1e-7) # to avoid dividing by zero
    return precision

def micro_multiClass_precision(y_true,y_pred):
    """
    a wrapper function for the function micro_multiClass_recall_py, it feeds the
    function to the function tf.py_func, making it a tensorflow op. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The macro multiClass recall, a scaler
    """
    return tf.py_func(micro_multiClass_precision_py,(y_true,y_pred),tf.double,
                      name='Micro_multiClass_Precision')

def micro_f1_multiClass_py(y_true,y_pred):
    """
    A Python function to compute the macro F1 scores for a multi class classifier,
    The Micro F1 scores is the Harmonic mean of the micro precision and micro 
    Recall
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The  macro multiClass F1scores, a scaler
    """
    recall=micro_multiClass_recall(y_true,y_pred)
    precision=micro_multiClass_precision(y_true,y_pred)
    micro_f1=2*precision*recall/(precision + recall + k.epsilon())
    return micro_f1

def micro_f1_multiClass(y_true,y_pred):
    """
    a wrapper function for the function micro_f1_multiClass_py, it feeds 
    the function to the function tf.py_func, making it a tensorflow op. 
    
    # inputs:
    y_true: a 2D tensor of size (batch_size,num_of_classes) which contians the
    one hot encoding of the true labels.
    
    y_pred: a 2D tensor of size (batch_size,num_of_classes) which contians a 
    softmax over different classes.
    
    # outputs: 
        The micro multiClass F1 scores, a scaler
    """
    return tf.py_func(micro_multiClass_precision_py,(y_true,y_pred),tf.double,
                      name='Micro_multiClass_F1_Scores')
    

