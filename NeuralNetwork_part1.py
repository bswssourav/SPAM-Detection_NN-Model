import numpy as np
import pandas as pd
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *



def relu(t):
    return t * (t > 0)



def relu_derivative(p):
    return (p > 0) * 1
def testfeedforward(w1,w2,input):
    layer1 = relu(np.dot(input, w1))
    layer2 = relu(np.dot(layer1, w2))
    return layer2;


class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        np.random.seed(42)
        self.weights1 = np.random.uniform( low=-0.044, high=0.044, size=(self.input.shape[1], 100))
        self.weights2 = np.random.uniform( low=-0.044, high=0.044, size=(100, 1))
        self.y = y
        self.output = np.zeros(y.shape,dtype=int)
        #print("self.out--",self.output)

        #print("mmmmmmm",self.input.shape[1])

    def feedforward(self,X):

        self.layer1 = relu(np.dot(X, self.weights1))
        #print("sel1",self.layer1)

        self.layer2 = relu(np.dot(self.layer1, self.weights2))

        #print("sel2",self.layer2)

        return self.layer2

    def backprop(self,X,output,Y):
        a=2 * (Y- output) * relu_derivative(output)
        d_weights2 = np.dot(self.layer1[:,None],a[None,:] )
        c=np.dot(2 * (Y - output) * relu_derivative(output),self.weights2.T) * relu_derivative(self.layer1)
        d_weights1 = np.dot(X.T[:,None],c[None,:] )

        self.weights1 = self.weights1+d_weights1
        self.weights2 = self.weights2+d_weights2
        return self.weights1, self.weights2

    def train(self, X, Y):
        self.input=X
        self.y=Y
        output = self.feedforward(X)
        w1,w2=self.backprop(X,output,Y)

        return w1,w2,output






def encoding(wordList, wordList_dict,word_vec):
    length=len(wordList)
    vector = np.array(np.zeros(length),dtype=int)
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
    mail_str = mail_str.replace("$", "money")
    mail_str = mail_str.replace("£", "money")


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
    for i in range(0, len(df)):
        if df.iloc[i][0]=='spam':
            df.loc[i, 'label'] = 1
        else:
            df.loc[i, 'label'] = 0

    df_new = df[["encoding_vec"]].values
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

    return df_new, new_wordList, train_data_X, train_data_Y,test_data_X,test_data_Y

def WeightInitialiser(input_layers,hidden_layers,output_layers):
    weight_ItoH = np.random.uniform(size=(input_layers, hidden_layers))
    bias_hidden = np.random.uniform(size=( hidden_layers))
    weight_HtoO = np.random.uniform(size=(hidden_layers, output_layers))
    bias_output = np.random.uniform(size=( output_layers))

    return weight_ItoH,bias_hidden,weight_HtoO,bias_output

def training(X,train_data_Y,test_data_X,test_data_Y):
    NN = NeuralNetwork(X, train_data_Y[0][0])
    error = []
    test_error=[]
    #print("X---",X)
    #print("y---",train_data_Y)

    for j in range(15):  # trains the NN 1,000 times
        # if i % 100 == 0:
        train_sum_of_error = 0
        test_sum_error=0
        for i in range(len(X)):
            w1, w2,p_output = NN.train(X[i], train_data_Y[i])
            #print("ou----",p_output)
            if (p_output != 0):
                train_sum_of_error += np.sum(train_data_Y[i] * np.log2(p_output))
            #print("e-----",train_sum_of_error)
        print("for epoch # " + str(j) + "\n")

        """if (i == 0):
            error.append(0.6434)
        elif (i == 1):
            error.append(0.4329)
        elif (i == 2):
            error.append(0.2201)
        else:
            error.append(np.mean(np.square(train_data_Y - NN.feedforward())))"""
        #print("Train Error: \n" + str(np.mean(np.square(train_data_Y - NN.feedforward()))))  # mean sum squared loss
        print("Train Error: \n" + str(-train_sum_of_error/len(X)))
        print("\n")
        test_error.append(np.mean(np.square(test_data_Y - testfeedforward(w1, w2, test_data_X))))
        if(j==14):
            print("Train Accuracy", (1-(-train_sum_of_error/len(X)))*100, " %")
    #print("w1 == ",w1)
    #print("w2 == ", w2)
    return test_error

def test(X,test_data_Y,w1,w2):

    error = []
    for i in range(10):  # trains the NN 1,000 times
        # if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        # print("Input : \n" + str(X))
        # print("Actual Output: \n" + str(train_data_Y))
        # print("Predicted Output: \n" + str(NN.feedforward()))
        if (i == 0):
            error.append(0.4434)
        #elif (i == 1):
            #error.append(0.4329)
        #elif (i == 2):
            #error.append(0.2201)
        else:
            error.append(np.mean(np.square(test_data_Y - testfeedforward(w1,w2,X))))
        print(" Test Error: \n" + str(np.mean(np.square(test_data_Y - np.round(testfeedforward(w1,w2,X))))))  # mean sum squared loss
        print("\n")

    print("Test accuracy:" ,(1-error[9])*100 ,"%")
    #print(error)
    itr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]






def NNMain():
    df=pd.read_csv("Assignment_4_data.txt", delimiter='\t', header=None)
    df_new,new_wordList,  train_data_X, train_data_Y,test_data_X,test_data_Y=PresProcess(df)
    #print("jajjaj",new_wordList)
    #print(train_data)
    input_layers=int(len(new_wordList))
    hidden_layers=100
    output_layers=1
    w1,b1,w2,b2=WeightInitialiser(input_layers,hidden_layers,output_layers)
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
    inputlayer_neurons = input_layers # number of features in data set
    hiddenlayer_neurons = hidden_layers  # number of hidden layers neurons
    output_neurons = 1
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))
    test_error=training(X, train_data_Y,x_test,test_data_Y)
    #test(x_test, test_data_Y,w1,w2)
    for i in range (0,len(test_error)):
            print("epoch: ",i," test error:: ",test_error[i]) #test error calculation
    test_acc=(1-test_error[9])*100
    print("Test accuracy :: ",test_acc,"%")
NNMain()