Built a multilayer neural network and classify whether a
given Email is SPAM or HAM (non-spam).

For each of the part, I have created separate source file namely 'NeuralNetwork_Part1' and 'NeuralNetwork_Part2'. 
I have wrote all the source file in python programming language.

Each of my source code reading from 'Assignment_4_data.txt'file for further purpose.
   
	


I have import below packages in my source code:
    numpy, pandas, nltk

One must install all these packages before running all my source code.

In the part 2 of the assignment: I have used 'argparse' package for taking command line argument. I have done my code on 'Pycharm' IDE.
In Pycharm, for taking command line argument for both the hidden layers neuron, I have coded following two lines:
      
     hidden_layers1=int(args.hidden1) //args.hidden1 is the argument for 1st hidden layer neurons
     hidden_layers2 = int(args.hidden2) //args.hidden2 is the argument for 2nd hidden layer neurons


