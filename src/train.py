import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
from torch.utils.data import DataLoader
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

import pdb

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api

from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset, Data_Data, Data_Data_Testing

if __name__ == "__main__":

    #print('ere')

    parser = argparse.ArgumentParser()

    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='rnn', type=str)
    parser.add_argument('--rnn_bidirect', default=False, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='glove', type=str)

    args = parser.parse_args()

    if args.init_word_embs == "glove":
        # TODO: Feed the GloVe embeddings to NN modules appropriately
        # for initializing the embeddings
        glove_embs = api.load("glove-wiki-gigaword-50")
        #embeddings = torch.nn.Embedding.from_pretrained(torch.from_numpy(glove_embs.vectors))
        # embeddings(sentence)
        embeddings_1=torch.nn.Embedding.from_pretrained(torch.from_numpy(glove_embs.vectors))
        #print("embedding size" , embeddings_1)

    if args.init_word_embs == "scratch":
        #create the embeddings of size 8192x50
       # embedding_vectors_random = torch.FloatTensor(np.random.rand(8192, 50))
        embeddings_1=torch.nn.Embedding(8192, 50) 
        #embeddings_1.weight = torch.nn.Parameter(embedding_vectors_random, requires_grad=True)
        

    # TODO: Freely modify the inputs to the declaration of each module below
    if args.neural_arch == "dan":
        model = DAN(input_size=50, hidden_size=50, output_size=1, n_layers=1, embeddings_1=embeddings_1, bidirectional=False).to(torch_device)
    elif args.neural_arch == "rnn":
        if args.rnn_bidirect:
            model = RNN(input_size=50,hidden_size=50, output_size=1,n_layers=1, embeddings_1=embeddings_1, bidirectional=True).to(torch_device)
        else:
            model = RNN(input_size=50,hidden_size=50, output_size=1,n_layers=1, embeddings_1=embeddings_1, bidirectional=False).to(torch_device)
    elif args.neural_arch == "lstm":
        if args.rnn_bidirect:
            model = LSTM(input_size=50, hidden_size=50, output_size=1, n_layers=1, embeddings_1=embeddings_1, bidirectional=True).to(torch_device)
        else:
            model = LSTM(input_size=50, hidden_size=50, output_size=1, n_layers=1, embeddings_1=embeddings_1, bidirectional=False).to(torch_device)

    # TODO: Read off the WiC dataset files from the `WiC_dataset' directory
    # (will be located in /homes/cs577/WiC_dataset/(train, dev, test))
    # and initialize PyTorch dataloader appropriately
    # Take a look at this page
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # and implement a PyTorch Dataset class for the WiC dataset in
    # utils.py


    #train
    #get the data set 
    nlp_dataset_train=WiCDataset(input_file_features='/homes/cs577/WiC_dataset/train/train.data.txt', input_file_labels='/homes/cs577/WiC_dataset/train/train.gold.txt')
    temp_length=len(nlp_dataset_train)
    training_context1=[]
    training_context2=[]
    labels_col_train=[]
    #obtain a list of lists 
    for i in range(len(nlp_dataset_train)):
        dict=nlp_dataset_train[i]
        training_context1.append(dict['Context1'])
        training_context2.append(dict['Context2'])
        labels_col_train.append(dict['T_or_F'])
    final_training_list_context1=[]
    final_training_list_context2=[]
    for j in training_context1:
        tempo=(j)
        tempo1=tempo.split()
        final_training_list_context1.append(tempo1)

    for k in training_context2:
        tempo=(k)
        tempo1=tempo.split()
        final_training_list_context2.append(tempo1)

    #creating a vocabulary with only Context1 words
    vocab=set()
    for sentence in final_training_list_context1:
        dicta={}
        c=0
        for word in sentence:
            if word!=".":
                dicta[word]=c
                c+=1
            vocab.add(word)
            #word indices for Context1
            word_to_ixa=dicta
        #print(word_to_ix)
        #print(dicta)

    #adding context2 words to 
    for sentence in final_training_list_context2:
        dictb={}
        c=0
        for word in sentence:
            if word!=".":
                dicta[word]=c
                c+=1
            vocab.add(word)
            #word indices for Context2
            word_to_ixb=dictb

    #creating the y values
    labels_final_train=[]
    for val in labels_col_train:
        if val=='T':
            labels_final_train.append(1)
        else:
            labels_final_train.append(0)
    #print('hi')

    #crea(te the word vectors for scratch
    #print(type(list(vocab)))
    your_dict = {key: i for i, key in enumerate(vocab)}
    
    #for sentence1
    final_sentence_vector1=[]
    for sentence in final_training_list_context1:
        sentence_vector1=[]
        for word in sentence:
            sentence_vector1.append(your_dict[word])
            #print(sentence_vector)
        final_sentence_vector1.append(sentence_vector1)
        #print('*****')
    #print(final_sentence_vector)

    #padding list
    max_tot=0
    for val in final_sentence_vector1:
        if len(val)>max_tot:
            max_tot=len(val)
    #print(max_tot) 31 is the max length
    for val in final_sentence_vector1:
        left=31-len(val)
        for j in range(left):
            val.append(0)
        val=torch.tensor(val)
        #print(val)
        #print('****')
    #print(len(final_sentence_vector))

    #for sentence2
    final_sentence_vector2=[]
    for sentence in final_training_list_context2:
        sentence_vector2=[]
        for word in sentence:
            sentence_vector2.append(your_dict[word])
        final_sentence_vector2.append(sentence_vector2)

    #padding list
    for val in final_sentence_vector2:
        left=31-len(val)
        for j in range(left):
            val.append(0)
        val=torch.tensor(val)
    #print(type(final_sentence_vector2))

    #padded word to index mapping
    final_sentence_vector1=torch.LongTensor(final_sentence_vector1)
    final_sentence_vector2=torch.LongTensor(final_sentence_vector2)

    #print('the train vector dim')
    #print(final_sentence_vector1.size())

    '''
    embedding_vectors = torch.FloatTensor(np.random.rand(vocab_size, 50))
        # embedding = nn.Embedding(33, 50)
        embedding = nn.Embedding(vocab_size, 50)
        # embedding.weight.data.uniform_(-1, 1)
        embedding.weight = nn.Parameter(embedding_vectors, requires_grad=True)'''
 
    

    #print(len(nlp_dataset_train))= 5428

    #printing entire vocabulary 
    #print(vocab)
    #print(len(vocab))
        #print(embeds1)
        #look_up_tensor=torch.tensor([word_to_ix['']], dtype=torch.long)

    #creating word vector for each 
    
    
        
    '''
    word_to_ix={"Hello":0, "worlds":1}
    embeds=torch.nn.Embedding(2, 6)
    print(embeds)
    look_up_tensor=torch.tensor([word_to_ix['Hello']], dtype=torch.long)
    hello_embed=embeds(look_up_tensor)
    print(hello_embed)
    look_up_tensor1=torch.tensor([word_to_ix['worlds']], dtype=torch.long)
    hello_embed1=embeds(look_up_tensor1)
    print(hello_embed1)
    '''

    #print(i, '| target_Word: ',dict['target_word'],'| noun_or_verb: ', dict['noun_or_verb'],'| Index1: ',dict['Index1'],'| Index2: ',dict['Index2'] ,'| Context1: ',dict['Context1'], '| Context2: ',dict['Context2'],'| T_or_F: ', dict['T_or_F'])

    #dev
    nlp_dataset_dev=WiCDataset(input_file_features='/homes/cs577/WiC_dataset/dev/dev.data.txt', input_file_labels='/homes/cs577/WiC_dataset/dev/dev.gold.txt')
    for i in range(len(nlp_dataset_dev)):
        dict=nlp_dataset_dev[i]
    #print(len(nlp_dataset_dev))



    data_instance=Data_Data(final_sentence_vector1, final_sentence_vector2, labels_final_train)
    data_loaded=DataLoader(data_instance, batch_size=120, shuffle=False)

    '''
    for data in data_loaded:
        print(data[0].size(0))
        print(data[1].size(0))
        print(data[2].size(0))
    '''
  

    # TODO: Training and validation loop here
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #criterion = torch.nn.CrossEntropyLoss()

    #print("Training for %d epochs..." % 100)
    #total_params = sum(param.numel() for param in model.parameters())
    #print(total_params)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    #criterion = torch.nn.CrossEntropyLoss()


    #setting optimum parameter values based on cross validation code which is shown below
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    for epoch in range(1, 1+1):
        loss_epoch=[] 
        acc_epoch=[] 
        #print("************")
        #print(epoch)
        l=0.0
        a=0.0
        #optimizer.zero_grad()
        for row in data_loaded:
            optimizer.zero_grad()
            sentence1=row[0]
            #print(sentence1)
            sentence2=row[1]
            #print(sentence2)
            #print("______")
            y=row[2]
            #print(y)
            output=model(sentence1, sentence2)
            #output1=torch.squeeze(output)
            output1 = torch.round(output).squeeze(1)
            #print(output1.size())
            loss = torch.nn.functional.binary_cross_entropy(output1.float(), y.float())
            #print(output1)
            loss.backward()
            optimizer.step()
            l += loss.item()
            a += (output1 == y).sum().item()
            #print((output1 == y))
        #print('Epoch : ', epoch)
        #print('Loss-> ', l/temp_length)
        loss_epoch.append(l/temp_length)
        #print('Accuracy-> ', a/temp_length*100)
        acc_epoch.append(a/temp_length*100)
    #print('**************************')
    #print('Final average loss across all epochs='+str(sum(loss_epoch)/len(loss_epoch)))
    #print('Final average accuracy across all epochs='+str(sum(acc_epoch)/len(acc_epoch)))
    #print('**************************')
    #print("training done")

 
    
    ########################CROSS VALIDATION
    
    #preprocessing the dev data set and padding 
    dev_context1=[]
    dev_context2=[]
    labels_col_dev=[]
    #obtain a list of lists 
    for i in range(len(nlp_dataset_dev)):
        dict_dev=nlp_dataset_dev[i]
        dev_context1.append(dict_dev['Context1'])
        dev_context2.append(dict_dev['Context2'])
        labels_col_dev.append(dict_dev['T_or_F'])
    final_dev_list_context1=[]
    final_dev_list_context2=[]
    for j in dev_context1:
        tempo=(j)
        tempo1=tempo.split()
        final_dev_list_context1.append(tempo1)
    for k in dev_context2:
        tempo=(k)
        tempo1=tempo.split()
        final_dev_list_context2.append(tempo1)
    #creating a vocabulary with only Context1 words
    vocab_dev=set()
    for sentence in final_dev_list_context1:
        dicta_dev={}
        c=0
        for word in sentence:
            if word!=".":
                dicta_dev[word]=c
                c+=1
            vocab_dev.add(word)
            #word indices for Context1
            word_to_ixa_dev=dicta_dev
    #adding context2 words to 
    for sentence in final_dev_list_context2:
        dictb_dev={}
        c=0
        for word in sentence:
            if word!=".":
                dicta_dev[word]=c
                c+=1
            vocab_dev.add(word)
            #word indices for Context2
            word_to_ixb_dev=dictb_dev
    #creating the y values
    labels_final_dev=[]
    for val in labels_col_dev:
        if val=='T':
            labels_final_dev.append(1)
        else:
            labels_final_dev.append(0)
    your_dict_dev = {key: i for i, key in enumerate(vocab_dev)}
    #for sentence1
    final_sentence_vector1_dev=[]
    for sentence in final_dev_list_context1:
        sentence_vector1=[]
        for word in sentence:
            sentence_vector1.append(your_dict_dev[word])
            #print(sentence_vector)
        final_sentence_vector1_dev.append(sentence_vector1)
    #padding list
    max_tot=0
    for val in final_sentence_vector1_dev:
        if len(val)>max_tot:
            max_tot=len(val)
    #print('size of max sentence')
    #print(max_tot)
    #print(max_tot) 31 is the max length
    for val in final_sentence_vector1_dev:
        left=max_tot-len(val)
        for j in range(left):
            val.append(0)
        val=torch.tensor(val)
    #for sentence2
    final_sentence_vector2_dev=[]
    for sentence in final_dev_list_context2:
        sentence_vector2=[]
        for word in sentence:
            sentence_vector2.append(your_dict_dev[word])
        final_sentence_vector2_dev.append(sentence_vector2)
    #padding list
    for val in final_sentence_vector2_dev:
        left=max_tot-len(val)
        for j in range(left):
            val.append(0)
        val=torch.tensor(val)
    #padded word to index mapping
    final_sentence_vector1_dev=torch.LongTensor(final_sentence_vector1_dev)
    final_sentence_vector2_dev=torch.LongTensor(final_sentence_vector2_dev)


    '''
    #I have run the same cross validation for different models by 
    #simply calling a different model everytime i run the cross validation loops
    print("**************cross validation for models*****************")

    #setting the parameters for hyperparameter tuning
    epochs=50
    #batch_size_list=[1,20, 40, 60, 80, 100, 200,400, 800, 1600, 2400]
    batch_size_list=[20]
    #learning_rate_list=[0.0001, 0.01, 0.9]
    learning_rate_list=[0.7]
    #weight_decay_rate=[1, 1e-5]
    weight_decay_rate=[0.00001]
    best_loss_value=10000
    best_accuracy_value=0
    best_loss_value_dev=10000
    best_accuracy_value_dev=0
    final_parameters={}
    final_parameters_dev={}
    #print("**********")
    #print(len(data_instance_dev))
    
    #iterate across the hyperparameters
    for ba in batch_size_list:
        #data loading 
        data_instance1=Data_Data(final_sentence_vector1, final_sentence_vector2, labels_final_train)
        data_loaded1=DataLoader(data_instance1, batch_size=ba, shuffle=False)
        #calling the Dataloader for dev
        data_instance_dev=Data_Data(final_sentence_vector1_dev, final_sentence_vector2_dev, labels_final_dev)
        data_loaded_dev=DataLoader(data_instance_dev, batch_size=ba, shuffle=False)
        for l in learning_rate_list:
            for w in weight_decay_rate:
                accuracy_current=[]
                loss_current=[]
                accuracy_current_val=[]
                loss_current_val=[]
                #call the model
                model = RNN(input_size=50,hidden_size=50, output_size=1,n_layers=1, embeddings_1=embeddings_1,bidirectional=False).to(torch_device)
                #model = DAN(input_size=50,hidden_size=50, output_size=1,n_layers=1, embeddings_1=embeddings_1,bidirectional=False ).to(torch_device)
                optimizer=torch.optim.Adam(params=model.parameters(),lr=l, weight_decay=w)
                for epoch in range(epochs):
                    total_length=0
                    total_length_dev=0
                    train_loss=0.0
                    train_acc=0
                    val_loss=0.0
                    val_acc=0
                    for sentence1,sentence2,y in data_loaded1:
                        optimizer.zero_grad()
                        prediction= model(sentence1, sentence2)
                        output= torch.round(prediction).squeeze(1)
                        loss=torch.nn.functional.binary_cross_entropy(output.float(), y.float())
                        loss.backward()
                        optimizer.step()
                        train_loss =train_loss+ loss.item()
                        train_acc =train_acc+ (output == y).sum().item()
                        total_length =total_length+ y.size(0)
                    accuracy_current.append(train_acc/total_length*100)
                    loss_current.append(train_loss/total_length)
                    #print('epoch number: '+str(epoch))
                    for context1, context2, y_dev in data_loaded_dev:
                        prediction_dev=model(context1, context2)
                        output_dev=torch.round(prediction_dev).squeeze(1)
                        loss_dev=torch.nn.functional.binary_cross_entropy(output_dev.float(), y_dev.float())
                        val_loss =val_loss+ loss_dev.item()
                        val_acc =val_acc+ (output_dev == y_dev).sum().item()
                        total_length_dev =total_length_dev+ y_dev.size(0)
                    accuracy_current_val.append(val_acc/total_length_dev*100)
                    loss_current_val.append(val_loss/total_length_dev)
                print('number of epochs run: '+str(epoch+1))
                val_avg_acc=sum(accuracy_current)/len(accuracy_current)
                val_avg_loss=sum(loss_current)/len(loss_current)
                val_avg_acc1=sum(accuracy_current_val)/len(accuracy_current_val)
                val_avg_loss1=sum(loss_current_val)/len(loss_current_val)

                #print("-------------BEST RESULTS--------------")
                #print('PARAMETERS: bs: '+str(ba)+' l: '+str(l)+' and wd: '+str(w))
                #print('ACCURACY: '+str(val_avg_acc)+', LOSS: '+str(val_avg_loss))
                if best_loss_value>val_avg_loss:
                    best_loss_value=val_avg_loss
                    #print('Best Loss found for these parameters=> bs:'+str(ba)+'; lr:+'+str(l)+'; wd:'+ str(w))
                    #print('bs:'+str(ba)+' lr:'+str(l)+' wd:'+ str(w)+' -> Accuracy: '+str(val_avg_acc)+' and Loss: '+str(val_avg_loss))
                if best_accuracy_value<val_avg_acc:
                    best_accuracy_value=val_avg_acc
                    #print('Best Accuracy found for these parameters=> bs:'+str(ba)+'; lr:'+str(l)+'; wd:'+ str(w))
                    #print('bs:'+str(ba)+' lr:'+str(l)+' wd:'+ str(w)+' -> Accuracy: '+str(val_avg_acc)+' and Loss: '+str(val_avg_loss))
                #append all teh values
                if best_loss_value_dev>val_avg_loss1:
                    best_loss_value_dev=val_avg_loss1
                if best_accuracy_value_dev<val_avg_acc1:
                    best_accuracy_value_dev=val_avg_acc1

                temp_list1=[]
                temp_list2=[]
                temp_list3=[]
                temp_list4=[]
                temp_list1.append(ba)
                temp_list1.append(l)
                temp_list1.append(w)
                temp_list4.append(ba)
                temp_list4.append(l)
                temp_list4.append(w)
                temp_list2.append(val_avg_acc)
                temp_list2.append(val_avg_loss)
                temp_list3.append(val_avg_acc1)
                temp_list3.append(val_avg_loss1)
                final_parameters[tuple(temp_list1)]=temp_list2
                final_parameters_dev[tuple(temp_list4)]=temp_list3

    #DISPLAY THE HYPERPARAMETER RESULTS NEATLY
        
    #print the results of cross validation
    print("\nTRAINING SET")
    print("\n\nPARAMETERS \t\t\t\t\t ACCURACY/LOSS")
    print('(Batch size, Learning rate, weight decay) \t (Accuracy\Loss)')
    for key in final_parameters:
        if key==(80, 0.01, 1.0) or key==(80, 0.9, 1.0) or key==(120, 0.9, 1.0) or key==(256, 0.9, 1.0) or key==(80, 0.0001, 1) or key==(120, 0.01, 1) or key==(256, 0.01, 1):
            print(str(key)+'\t\t\t\t\t'+str(final_parameters[key]))
        else:
            print(str(key)+'\t\t\t\t'+str(final_parameters[key]))
    print('\n')
    print("\nDEV SET")
    #print the results of cross validation
    print("\n\nPARAMETERS \t\t\t\t\t ACCURACY/LOSS")
    print('(Batch size, Learning rate, weight decay) \t (Accuracy\Loss)')
    for key in final_parameters_dev:
        if key==(80, 0.01, 1.0) or key==(80, 0.9, 1.0) or key==(120, 0.9, 1.0) or key==(256, 0.9, 1.0) or key==(80, 0.0001, 1) or key==(120, 0.01, 1) or key==(256, 0.01, 1):
            print(str(key)+'\t\t\t\t\t'+str(final_parameters_dev[key]))
        else:
            print(str(key)+'\t\t\t\t'+str(final_parameters_dev[key]))
    print('\n')
    '''

    
    # TODO: Testing loop
    # Write predictions (F or T) for each test example into test.pred.txt
    # One line per each example, in the same order as test.data.txt.

    #test
    nlp_dataset_test=WiCDataset(input_file_features='/homes/cs577/WiC_dataset/test/test.data.txt', input_file_labels='')
    #nlp_dataset_test=WiCDataset(input_file_features='/homes/cs577/WiC_dataset/test/test.data.txt', input_file_labels='')
    for i in range(len(nlp_dataset_test)):
        dict=nlp_dataset_test[i]
    #print(len(nlp_dataset_test))

    #preprocessing the dev data set and padding 
    test_context1=[]
    test_context2=[]
    #obtain a list of lists 
    for i in range(len(nlp_dataset_test)):
        dict_test=nlp_dataset_test[i]
        test_context1.append(dict_test['Context1'])
        test_context2.append(dict_test['Context2'])
 
    final_t_1_list_context1=[]
    final_t_2_list_context2=[]
    for j in test_context1:
        tempo=(j)
        tempo1=tempo.split()
        final_t_1_list_context1.append(tempo1)
    for k in test_context2:
        tempo=(k)
        tempo1=tempo.split()
        final_t_2_list_context2.append(tempo1)
    #print(final_t_1_list_context1)
    #print(final_t_2_list_context2)

    
    your_dict_test = {key: i for i, key in enumerate(vocab)}
    #print(your_dict_test)
    #for sentence1
    final_sentence_vector1_test=[]
    for sentence in final_t_1_list_context1:
        sentence_vector1=[]
        for word in sentence:
            if word in vocab:
                sentence_vector1.append(your_dict_test[word])
            else:
                sentence_vector1.append(8999)
            #print(sentence_vector)
        final_sentence_vector1_test.append(sentence_vector1)
    #print(type(final_sentence_vector1_test))

    #padding list
    '''
    max_tot=0
    for val in final_sentence_vector1_test:
        if len(val)>max_tot:
            max_tot=len(val)
    #print('size of max sentence')
    #print(max_tot)
    #print(max_tot) 31 is the max length
    '''
    


    for val in final_sentence_vector1_test:
        left=31-len(val)
        #print(left)
        for j in range(left):
            val.append(9000)
        val=torch.tensor(val)
    
    #print(len(final_sentence_vector1_test))
    
    #for sentence2
    final_sentence_vector2_test=[]
    for sentence in final_t_2_list_context2:
        sentence_vector2=[]
        for word in sentence:
            if word in vocab:
                sentence_vector2.append(your_dict_test[word])
            else:
                sentence_vector2.append(8999)
        final_sentence_vector2_test.append(sentence_vector2)
    #print(len(final_sentence_vector2_test))
    #padding list
    for val in final_sentence_vector2_test:
        left=31-len(val)
        for j in range(left):
            val.append(9000)
        val=torch.tensor(val)

    
    final_sentence_vector1_test_1=torch.LongTensor(final_sentence_vector1_test)
    final_sentence_vector2_test_2=torch.LongTensor(final_sentence_vector2_test)

    #print('the test vector dim')
    #print(final_sentence_vector2_test_2.size())

  

    data_instance_test1=Data_Data_Testing(final_sentence_vector1_test_1, final_sentence_vector2_test_2)
    data_loaded_test1=DataLoader(data_instance_test1, batch_size=1, shuffle=False)
    #print(len(labels_final_dev))
    '''
    #data_instance_dev=Data_Data(final_sentence_vector1_dev, final_sentence_vector2_dev, labels_final_dev)
    #data_loaded_dev=DataLoader(data_instance_dev, batch_size=100, shuffle=False)
    print('second time training')
    #setting optimum parameter values based on cross validation code which is shown below
    for epoch in range(1, 1+1):
        print("************")
        for row in data_loaded_test1:
            sentence1=row[0]
            print(sentence1.size())
            sentence2=row[1]
            print(sentence2.size())
            output_test=model(sentence1, sentence2)
            output1 = torch.round(output_test).squeeze(1)
    print('**************************')
    print('**************************')
    print("testing done")
    '''



    
    file_ptr=open('test.pred.txt', 'a')

    #print("************TESTING*************")
    for row in data_loaded_test1:
        sentence1=row[0]
        #print(sentence1.size())
        sentence2=row[1]
        #print(sentence2.size())
        output_test=model(sentence1, sentence2)
        #print(output_test)
        output1_test=torch.round(output_test).squeeze(1)
        for predicted_val in output1_test.tolist():
            if predicted_val==1.0:
                file_ptr.write('T')
            else:
                file_ptr.write('F')
            file_ptr.write('\n')

    file_ptr.close()
    #print("testing done")

    




    
    
    
