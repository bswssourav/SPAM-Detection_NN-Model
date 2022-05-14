import sys
import numpy as np
import pandas as pd
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import argparse






def encoding(wordList, wordList_dict,word_vec):
    length=len(wordList)
    vector = np.array(np.zeros(length))
    vector=np.array(vector)
    #print("vector",vector)
    index=[]
    index =[wordList_dict[word] for word in word_vec if word in wordList]
    #for i in word_vec:
        #if i in wordList:
            #index.append(wordList_dict[i])
    #print("index",index)
    vector[index]=1
    #print("vector==",vector)

    return vector




def tokenize(mail, stop_words, stemmer):

    mail_str = str(mail)

    mail_str = mail_str.lower()
    mail_str = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'link', mail_str)

    mail_str = mail_str.replace(":)", " emoticon ").replace(":(", " emoticon ").replace(":D", " emoticon ")
    mail_str = mail_str.replace("£", "money")
    mail_str = mail_str.replace("$", "money")

    mail_str = mail_str.lower()


    #print(mail_str)
    gotToken = re.split("[!. ,?\'\/#\n:\(\)\[\]]", mail_str)
    #gotToken=mail_str.split("[!. ,?\'\/#\n:\(\)\[\]]")
    gotToken = list(filter(lambda x: x not in stop_words and x != '', gotToken))
    gotToken = [stemmer.stem(token.strip()) for token in gotToken]
    return gotToken





def PresProcess(df):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    df["tokens"] = df[1].apply(lambda x: tokenize(x.strip(), stopWords, stemmer))
    #print(df)
    wordList = sum(df["tokens"].values, [])
    wordList = Counter(wordList).most_common(1185)
    #print(wordList)
    wordList_dict={}

    for i in wordList:
        wordList_dict[i[0]]=i[1]
    #print(wordList_dict)
    new_wordList=[]
    for v in wordList:
        new_wordList.append(v[0])
    #print(new_wordList)
    df["encoding_vec"] = df["tokens"].apply(lambda x: encoding(new_wordList, wordList_dict,x))
    #print("hahhah",df["one_hot_vec"])
   # print(df)
    #df["labels"] = df[0].apply(lambda x: 1 if x == 'spam' else 0)
    #print(df)
    df["label"]=0
    #print(df)
    """for i in range(0, len(df)):
        if df.iloc[i][0]=='spam':
            df.loc[i, 'label'] = 1
        else:
            df.loc[i, 'label'] = 0"""
    df["label"] = df[0].apply(lambda x: np.array([1, 0]) if x == 'spam' else np.array([0, 1]))
    df_new = df[["encoding_vec","label"]].values
    df_new_label = df[["label"]].values


    splitIntraintest = int(0.8 * len(df_new))
    train_data_x= df_new[:splitIntraintest]
    test_data_x = df_new[splitIntraintest:]
    train_data_y = df_new_label[:splitIntraintest]
    test_data_y = df_new_label[splitIntraintest:]
    train_data_X=np.array(train_data_x)
    train_data_Y= np.array(train_data_y)
    test_data_X = np.array(test_data_x)
    test_data_Y = np.array(test_data_y)
    #print(train_data_X)
    #print(train_data_Y)
    print("train---",train_data_x)

    return df_new, new_wordList, train_data_x, train_data_y,test_data_x,test_data_y
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def softmax(x):
    sum_exps = np.sum(np.exp(x))
    return np.exp(x) / sum_exps


def forward(input_vec,  w1,w2,w3,b1,b2,b3):
    h1_input=np.dot(input_vec, w1)+b1
    h1_activation = sigmoid(h1_input)
    h2_input=np.dot(h1_activation, w2) + b2
    h2_activation = sigmoid(h2_input)
    o_input=np.dot(h2_activation, w3) + b3
    output = softmax(o_input)

    #print("oaooaoao----",output)
    return h1_activation, h2_activation, output


def forward_test(input_vec,  w1,w2,w3,b1,b2,b3):
    h1_input=np.dot(input_vec, w1)+b1
    h1_activation = sigmoid(h1_input)
    h2_input=np.dot(h1_activation, w2) + b2
    h2_activation = sigmoid(h2_input)
    o_input=np.dot(h2_activation, w3) + b3
    output = softmax(o_input)


    return output


def backpropagation(input_x,h1_activation, h2_activation, output, actual_output, w1, w2, w3, b1,b2, b3):
    slope_3 = np.prod(derivatives_sigmoid(output)) * (output - actual_output)

    d_weight3 = np.dot(h2_activation[:, None], slope_3[None, :])
    b3_delta = slope_3
    a=derivatives_sigmoid(h2_activation)
    b=np.dot(w3, slope_3)
    slope_2 =  a*b
    d_weight2 = np.dot(h1_activation[:, None], slope_2[None, :])
    b2_delta = slope_2
    x1=derivatives_sigmoid(h1_activation)
    x2=np.dot(w2, slope_2)
    slope_1 =  x1* x2
    d_weight1 = np.dot(input_x[:, None], slope_1[None, :])
    b1_delta = slope_1

    w1 =w1 - lr * d_weight1
    b1 = b1 -lr * b1_delta

    w2 = w2-lr * d_weight2
    b2 = b2-lr * b2_delta

    w3 =w3- lr * d_weight3
    b3 =b3- lr * b3_delta

    return w1, w2, w3, b1, b2,b3



def WeightInitialiser(input_layers,hidden_layers1,hidden_layers2,output_layers):
    w1 = np.random.uniform(low=-0.044, high=0.044, size=(input_layers, hidden_layers1))
    b1 = np.random.uniform(low=-0.044, high=0.044, size=(hidden_layers1))
    w2 = np.random.uniform(low=-0.044, high=0.044, size=(hidden_layers1, hidden_layers2))
    b2 = np.random.uniform(low=-0.044, high=0.044, size=(hidden_layers2))
    w3 = np.random.uniform(low=-0.044, high=0.044, size=(hidden_layers2, output_layers))
    b3 = np.random.uniform(low=-0.044, high=0.044, size=(output_layers))

    return w1,w2,w3,b1,b2,b3

def training(w1,w2,w3,b1,b2,b3,X,Y,test_X):
    lr = 0.1
    p_output=[]
    sample1 = len(X)
    epoch = 15
    sample2 = len(test_X)





    test_error=[]
    for epo in range(epoch):
        train_sum_of_error = 0
        test_sum_error=0
        for i in range (sample1):
                train_input=X[i][0]
                train_actual_output=X[i][1]

                #print("train actual output",train_actual_output)
                h1_activation, h2_activation, predicted_output=forward(train_input, w1,w2,w3,b1,b2 ,b3)

                if (np.round(predicted_output.any()) != 0):
                    train_sum_of_error += np.sum(train_actual_output * np.log2(predicted_output))

                w1, w2, w3, b1,b2, b3 = backpropagation(train_input,h1_activation, h2_activation, predicted_output, train_actual_output, w1,w2,w3,b1,b2,b3)
        PO=np.array(p_output)
        for i in range (sample2):
                test_input = test_X[i][0]
                test_actual_output = test_X[i][1]
                test_predicted_output = forward_test(test_input, w1, w2, w3, b1, b2, b3)
                if (test_predicted_output.any()!=0):
                    test_sum_error = test_sum_error + np.sum((np.log2(test_predicted_output) * test_actual_output))

        print("after epoch:", epo+1, "train error---",(-train_sum_of_error/len(X)))
        if(epo==14):
            print("Train accuracy::",(1-(-train_sum_of_error/len(X)))*100," %")
        test_error.append((-test_sum_error/len(test_X))) # test error calculation
    return test_error



def NNMain():
    df=pd.read_csv("Assignment_4_data.txt", delimiter='\t', header=None)
    df_new,new_wordList,  train_data_X, train_data_Y,test_data_X,test_data_Y=PresProcess(df)
    #print("jajjaj",new_wordList)
    #print(train_data)
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("hidden1", help="1 st hidden layer neurons:", type=int)
    parser.add_argument("hidden2", help="2 nd  hidden layer neurons:", type=int)
    args=parser.parse_args()
    input_layers=int(len(new_wordList))
    hidden_layers1=int(args.hidden1)
    hidden_layers2 = int(args.hidden2)
    print("layers",hidden_layers1,hidden_layers2)
    output_layers=2
    w1,w2,w3,b1,b2,b3=WeightInitialiser(input_layers,hidden_layers1,hidden_layers2,output_layers)
    X_train=[]
    for i in range(0,len(train_data_X)):
        X_train.append(train_data_X[i][0])
    X_test = []
    for i in range(0, len(test_data_X)):
        X_test.append(test_data_X[i][0])
    X1 = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])
    X=np.array(X_train)
    x_test=np.array(X_test)
    #print(x_test)
    #print(train_data_Y)
    #print(X1)
    epoch = 500
    lr = 0.1
    test_error=training(w1,w2,w3,b1,b2,b3,train_data_X, train_data_Y,test_data_X)
    for i in range (0,len(test_error)):
        print("After epoch: ",i+1,"test error: ",test_error[i]) # test error printing
    test_acc=(1-test_error[len(test_error)-1])*100
    print("Test accuracy :: ",test_acc, " %")



lr=0.1
NNMain()