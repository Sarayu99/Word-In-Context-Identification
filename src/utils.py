from torch.utils.data import Dataset
import pandas as pd
import torch
import gensim
import numpy as nm
import nltk 
import scipy

'''
#torch tester code
import torch
x = torch.rand(5, 3)
print(x)
'''

'''
print(nm.__version__)
import torch
import gensim
print(torch.__version__)
print(pd.__version__)
print(nltk.__version__)
print(gensim.__version__)
print(scipy.__version__)
'''


#print('ere')

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class WiCDataset(Dataset):
    test=0

    def __init__(self, input_file_features, input_file_labels):
        if ('dev' in input_file_features or 'train' in input_file_features):
            self.df1=pd.read_csv(input_file_features,sep='\t', header=None)
            self.df2=pd.read_csv(input_file_labels,sep='\t', header=None)
            self.combined_data=pd.concat([self.df1, self.df2], axis=1)
            #self.combined_data=self.combined_data[2].iloc[:,2].str.replace('-', '\t')
            #print(self.combined_data)
            #print(len(self.combined_data.columns))
            self.combined_data.columns=['target_word', 'noun_or_verb','Indices','Context1', 'Context2', 'T_or_F' ]
            self.combined_data[['Index1','Index2']] = self.combined_data['Indices'].str.split('-',expand=True)
            #print(self.combined_data)
            self.final_combined=self.combined_data[['target_word', 'noun_or_verb','Index1', 'Index2','Context1', 'Context2', 'T_or_F']]
            #print(self.final_combined)
        else:
            self.df1=pd.read_csv(input_file_features,sep='\t', header=None)
            self.combined_data=self.df1
            self.combined_data.columns=['target_word', 'noun_or_verb','Indices','Context1', 'Context2']
            self.combined_data[['Index1','Index2']] = self.combined_data['Indices'].str.split('-',expand=True)
            self.final_combined=self.combined_data[['target_word', 'noun_or_verb','Index1', 'Index2','Context1', 'Context2']]
            self.test=1


   
        

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        #for the 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.test==0:
            self.final_combined=self.combined_data[['target_word', 'noun_or_verb','Index1', 'Index2','Context1', 'Context2', 'T_or_F']]
        else:
            self.final_combined=self.combined_data[['target_word', 'noun_or_verb','Index1', 'Index2','Context1', 'Context2']]


        target_word = self.final_combined.iloc[idx, 0]
        noun_or_verb = self.final_combined.iloc[idx, 1]
        Index1 = self.final_combined.iloc[idx, 2]
        Index2 = self.final_combined.iloc[idx, 3]
        Context1 = self.final_combined.iloc[idx, 4]
        Context2 = self.final_combined.iloc[idx, 5]
        if self.test==0:
            T_or_F = self.final_combined.iloc[idx,6 ]
            dict = {'target_word': target_word, 'noun_or_verb': noun_or_verb,'Index1':Index1, 'Index2':Index2, 'Context1':Context1, 'Context2':Context2,'T_or_F':T_or_F}
        else:
            dict = {'target_word': target_word, 'noun_or_verb': noun_or_verb,'Index1':Index1, 'Index2':Index2, 'Context1':Context1, 'Context2':Context2}


        return dict
    
class Data_Data(Dataset):
    def __init__(self, Context1, Context2, label):
        self.Context1=Context1
        self.Context2=Context2
        self.label=label

    def __len__(self):
        return len(self.Context1)

    def __getitem__(self, idx):
        return self.Context1[idx], self.Context2[idx], self.label[idx]
    
class Data_Data_Testing(Dataset):
    def __init__(self, Context1, Context2):
        self.Context1=Context1
        self.Context2=Context2

    def __len__(self):
        return len(self.Context1)

    def __getitem__(self, idx):
        return self.Context1[idx], self.Context2[idx]




