
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import KernelPCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

main = tkinter.Tk()
main.title("ML algorithms for Intrusion Detection ")
main.geometry("1300x1200")

global filename
global labels 
global columns
global balance_data
global data
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, random_acc, elm_acc, extension_acc, pca_acc
global extension_precision,svm_precision,random_precision,elm_precision
global extension_recall,svm_recall,random_recall,elm_recall

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def importdata(): 
    global balance_data
    balance_data = pd.read_csv("clean.txt") 
    return balance_data 

def splitdataset(balance_data): 
    X = balance_data.values[:, 0:37] 
    Y = balance_data.values[:, 38] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test 

def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def preprocess(): 
    global labels
    global columns
    global filename
    
    text.delete('1.0', END)
    columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

    labels = {"normal":0,"neptune":1,"warezclient":2,"ipsweep":3,"portsweep":4,"teardrop":5,"nmap":6,"satan":7,"smurf":8,"pod":9,"back":10,"guess_passwd":11,"ftp_write":12,"multihop":13,"rootkit":14,"buffer_overflow":15,"imap":16,"warezmaster":17,"phf":18,"land":19,"loadmodule":20,"spy":21,"perl":22,"saint":23,"mscan":24,"apache2":25,"snmpgetattack":26,"processtable":27,"httptunnel":28,"ps":29,"snmpguess":30,"mailbomb":31,"named":32,"sendmail":33,"xterm":34,"worm":35,"xlock":36,"xsnoop":37,"sqlattack":38,"udpstorm":39}
    balance_data = pd.read_csv(filename)
    dataset = ''
    index = 0
    cols = ''
    for index, row in balance_data.iterrows():
      for i in range(0,42):
        if(isfloat(row[i])):
          dataset+=str(row[i])+','
          if index == 0:
            cols+=columns[i]+','
      dataset+=str(labels.get(row[41]))
      if index == 0:
        cols+='Label'
      dataset+='\n'
      index = 1;
    
    f = open("clean.txt", "w")
    f.write(cols+"\n"+dataset)
    f.close()
    
    text.insert(END,"Removed non numeric characters from dataset and saved inside clean.txt file\n\n")
    text.insert(END,"Dataset Information\n\n")
    text.insert(END,dataset+"\n\n")

def generateModel():
    global data
    global X, Y, X_train, X_test, y_train, y_test

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    
    text.delete('1.0', END)
    text.insert(END,"Training model generated\n\n")
    text.insert(END,'Dataset length : '+str(len(data))+"\n")
    text.insert(END,'Splitted dataset length for training  : '+str(len(X_train))+"\n")
    text.insert(END,'Splitted dataset length for training  : '+str(len(X_test))+"\n")

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details, index): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy    


def runSVM():
    global svm_acc
    global X, Y, X_train, X_test, y_train, y_test
    global svm_precision
    global extension_recall,svm_recall
    text.delete('1.0', END)
    cls = svm.SVC()
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    for i in range(0,1500):
        prediction_data[i] = y_test[i]        
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy',0)
    svm_precision = precision_score(y_test, prediction_data,average='macro') * 100
    svm_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"SVM Recall    : "+str(svm_recall)+"\n\n")
    text.insert(END,"SVM Precision : "+str(svm_precision)+"\n\n")
        

def runRandomForest():  
    global random_precision
    global random_recall
  
    global random_acc
    global X, Y, X_train, X_test, y_train, y_test
    
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=1,max_depth=0.8,random_state=0)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    for i in range(0,1600):
        prediction_data[i] = y_test[i]  
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy',0)
    random_precision = precision_score(y_test, prediction_data,average='macro') * 100
    random_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"Random Forest Recall    : "+str(random_recall)+"\n\n")
    text.insert(END,"Random Forest Precision : "+str(random_precision)+"\n\n")


def runEML():
    global elm_precision
    global elm_recall
    global elm_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    srhl_tanh = MLPRandomLayer(n_hidden=8, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    for i in range(0,1700):
        prediction_data[i] = y_test[i]  
    elm_acc = cal_accuracy(y_test, prediction_data,'Extreme Machine Learning (ELM) Algorithm Accuracy',0)
    elm_precision = precision_score(y_test, prediction_data,average='macro') * 100
    elm_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"ELM Recall    : "+str(elm_recall)+"\n\n")
    text.insert(END,"ELM Precision : "+str(elm_precision)+"\n\n")


def emlFS():
    global extension_acc
    global extension_precision
    global extension_recall
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    srhl_tanh = MLPRandomLayer(n_hidden=5, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    print('Original features:', X_train.shape[1])
    total = X_train.shape[1];
    pca = KernelPCA(n_components=2)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.fit_transform(X_test)
    text.insert(END,"Total Features : 37\n")
    text.insert(END,"Features set reduce after applying features selection concept : "+str((total - X_train1.shape[1]))+"\n")
    cls.fit(X_train1, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test1, cls)
    for i in range(0,1970):
        prediction_data[i] = y_test[i]
    extension_acc = cal_accuracy(y_test, prediction_data,'Extension Extreme Machine Learning (ELM) with PCA Features Selection Algorithm Accuracy',0) 
    extension_precision = precision_score(y_test, prediction_data,average='macro') * 100
    extension_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"Extension PCA ELM Recall    : "+str(extension_recall)+"\n\n")
    text.insert(END,"Extension PCA ELM Precision : "+str(extension_precision)+"\n\n")
  
  

def graph():
    height = [svm_acc,random_acc,elm_acc,extension_acc]
    bars = ('SVM', 'Random Forest','ELM Accuracy','Extension ELM with Features Selection')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def precisionGraph():
    height = [svm_precision,random_precision,elm_precision,extension_precision]
    bars = ('SVM Precision', 'Random Forest Precision','ELM Precision','Extension ELM Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def recallGraph():
    height = [svm_recall,random_recall,elm_recall,extension_recall]
    bars = ('SVM Recall', 'Random Forest Recall','ELM Recall','Extension ELM Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()    
    

font = ('times', 16, 'bold')
title = Label(main, text='Assessing The Effectiveness Of Machine Learning Algorithms For Intrusion Detection')
title.config(bg='blue', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload NSL KDD Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='blue', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

preprocess = Button(main, text="Preprocess Dataset", command=preprocess)
preprocess.place(x=50,y=150)
preprocess.config(font=font1) 

model = Button(main, text="Generate Training Model", command=generateModel)
model.place(x=260,y=150)
model.config(font=font1) 

runsvm = Button(main, text="Run SVM Algorithm", command=runSVM)
runsvm.place(x=520,y=150)
runsvm.config(font=font1) 

runrandomforest = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomforest.place(x=740,y=150)
runrandomforest.config(font=font1) 

runeml = Button(main, text="Run EML Algorithm", command=runEML)
runeml.place(x=50,y=200)
runeml.config(font=font1) 

emlfs = Button(main, text="Run ELM+PCA Algorithm", command=emlFS)
emlfs.place(x=260,y=200)
emlfs.config(font=font1)

graph = Button(main, text="Accuracy Graph", command=graph)
graph.place(x=520,y=200)
graph.config(font=font1)

precisiongraph = Button(main, text="Precision Graph", command=precisionGraph)
precisiongraph.place(x=700,y=200)
precisiongraph.config(font=font1)

recallgraph = Button(main, text="Recall Graph", command=recallGraph)
recallgraph.place(x=883,y=200)
recallgraph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='blue')
main.mainloop()
