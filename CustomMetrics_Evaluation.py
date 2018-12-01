# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:38:50 2018

@author: Hesham El Abd
@aim: A script to compare CustomMetrics results to sklearn.metrics
"""
###########################################################################
from CustomMetrics import (binary_recall, binary_precision, binary_f1_score,
                           macro_multiClass_recall,micro_multiClass_recall,
                           macro_multiClass_precision,
                           micro_multiClass_precision, macro_f1_multiClass,
                           micro_f1_multiClass)
from sklearn.metrics import recall_score,precision_score,f1_score
import keras.backend as k
import numpy as np
############################################################################

### for binary classifications: 

# I- using skLearn 

y_true=np.array([1,0,1,0,1,0])
y_pred=np.array([0,1,1,0,1,0]) 
recall_sklearn=recall_score(y_true,y_pred)
precision_sklearn=precision_score(y_true,y_pred)
f1_score_sklearn=f1_score(y_true,y_pred)

# I- using CustomMetrics:
y_trueTensor=k.variable(y_true)
y_predTensor=k.variable(y_pred)
recall_CustomMetrics=k.eval(binary_recall(y_trueTensor,y_predTensor))
precision_CustomMetrics=k.eval(binary_precision(y_trueTensor,y_predTensor))
f1_score_CustomMetrics=k.eval(binary_f1_score(y_trueTensor,y_predTensor))

# comparing the resultes: 
float('%.5f'%recall_sklearn)==float('%.5f'%recall_CustomMetrics)
float('%.5f'%precision_sklearn)==float('%.5f'%precision_CustomMetrics)
float('%.5f'%f1_score_sklearn)==float('%.5f'%f1_score_CustomMetrics)

### for multiclass classification:

# I- using sklearn: 
y_true=np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,0,0]])
y_pred=np.array([[0,1,0],[0,1,0],[0,0,1],[1,0,0],[0,0,1],[0,0,1]])
recall_macro_sklearn=recall_score(y_true,y_pred,average='macro')
recall_micro_sklearn=recall_score(y_true,y_pred,average='micro')
precision_macro_sklearn=precision_score(y_true,y_pred,average='macro')
precision_micro_sklearn=precision_score(y_true,y_pred,average='micro')
f1_score_macro_sklearn=f1_score(y_true,y_pred,average='macro')
f1_score_micro_sklearn=f1_score(y_true,y_pred,average='micro')

# II- using customMetrics: 
y_trueTensor=k.variable(np.array([[1,0,0],[0,1,0],[0,0,1],
                                  [1,0,0],[0,1,0],[1,0,0]]))
y_predTensor=k.variable(np.array([[0,1,0],[0,1,0],[0,0,1],
                            [1,0,0],[0,0,1],[0,0,1]]))

recall_macro_CustomMetrics=k.eval(macro_multiClass_recall(
        y_trueTensor,y_predTensor))

recall_micro_CustomMetrics=k.eval(micro_multiClass_recall(
        y_trueTensor,y_predTensor))

precision_macro_CustomMetrics=k.eval(macro_multiClass_precision(
        y_trueTensor,y_predTensor))

precision_micro_CustomMetrics=k.eval(micro_multiClass_precision(
        y_trueTensor,y_predTensor))

f1_score_macro_CustomMetrics=k.eval(macro_f1_multiClass(
        y_trueTensor,y_predTensor))

f1_score_micro_CustomMetrics=k.eval(micro_f1_multiClass(
        y_trueTensor,y_predTensor))

# comparing the results:
float('%.5f'%recall_macro_sklearn)==float('%.5f'%recall_macro_CustomMetrics)
float('%.5f'%recall_micro_sklearn)==float('%.5f'%recall_micro_CustomMetrics)
float('%.5f'%precision_macro_sklearn)==float('%.5f'%precision_macro_CustomMetrics)
float('%.5f'%precision_micro_sklearn)==float('%.5f'%precision_micro_CustomMetrics)
float('%.5f'%f1_score_macro_sklearn)==float('%.5f'%f1_score_macro_CustomMetrics)
float('%.5f'%f1_score_micro_sklearn)==float('%.5f'%f1_score_micro_CustomMetrics)
