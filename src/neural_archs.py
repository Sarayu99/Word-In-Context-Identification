import torch
import torch.nn.functional
import pdb


# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.
class DAN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, embeddings_1,bidirectional=False ):
        # TODO: Declare DAN architecture
        super(DAN, self).__init__()
        #print("DAN")
        self.input_size=input_size
        self.embeddings=embeddings_1
        self.linear_layer1=torch.nn.Linear(input_size, hidden_size)
        self.linear_layer12=torch.nn.Linear(hidden_size, 32)
        self.linear_layer2 =torch.nn.Linear(input_size, hidden_size)
        self.linear_layer21=torch.nn.Linear(hidden_size, 32)
        self.linear=torch.nn.Linear(64,output_size)
        self.sigmoid=torch.nn.Sigmoid()
        self.flat=torch.nn.Flatten()

    def forward(self, sentence1, sentence2):
        # TODO: Implement DAN forward pass
        embedded_setence1=self.embeddings(sentence1)
        p_1=torch.mean(embedded_setence1, dim=1)
        temp1=self.linear_layer1(p_1.float())
        h1=torch.nn.functional.relu(temp1)
        o1=self.linear_layer12(h1)
        embedded_setence2=self.embeddings(sentence2)
        p_2=torch.mean(embedded_setence2, dim=1)
        temp2=self.linear_layer2(p_2.float())
        h2=torch.nn.functional.relu(temp2)
        o2=self.linear_layer21(h2)
        combined=torch.cat((o1, o2),dim=1)
        final_answer=self.flat(combined)
        lin_final=self.linear(final_answer)
        o=self.sigmoid(lin_final)     
        return o


class RNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, output_size, n_layers, embeddings_1, bidirectional):
       # TODO: Declare RNN model architecture
        super(RNN, self).__init__() 
        #print("RNN")
        self.hidden_size=hidden_size
        if(int(bidirectional)==1):
            self.num_of_directions=2
        elif (int(bidirectional)==0):
            self.num_of_directions=1
        #print('number of directions'+str(self.num_of_directions))
        self.rnn1 = torch.nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.rnn2 = torch.nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.linear = torch.nn.Linear(hidden_size * 2 * self.num_of_directions, output_size)
        self.embeddings_1=embeddings_1
        #pdb.set_trace()
        self.flatten_layer=torch.nn.Flatten()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, sentence1, sentence2):
        # TODO: Implement RNN forward pass
        #pdb.set_trace()
        sentence1_embedding=self.embeddings_1(sentence1)
        sentence2_embedding=self.embeddings_1(sentence2)
        h1, _ = self.rnn1(sentence1_embedding)
        h2, _ = self.rnn2(sentence2_embedding)
        #h = torch.cat((h1[:, -1, :], h2[:, -1, :]), dim=1)
        h1_mean = torch.mean(h1, dim=1)
        h2_mean = torch.mean(h2, dim=1)
        #concatenated_output = torch.cat((h1[:, -1, :], h2[:, -1, :]), dim=1)
        concatenated_output = torch.cat((h1_mean, h2_mean), dim=1)
        h_out_1 = self.linear(concatenated_output)
        y = self.sigmoid(h_out_1)
        #pdb.set_trace()
        return y

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, embeddings_1, bidirectional):
        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()
        #print("LSTM")
        self.hidden_size=hidden_size
        if(int(bidirectional)==1):
            self.num_of_directions=2
        elif (int(bidirectional)==0):
            self.num_of_directions=1
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.linear = torch.nn.Linear(hidden_size * 2 * self.num_of_directions, output_size)
        self.embeddings_1=embeddings_1
        self.flatten_layer=torch.nn.Flatten()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, sentence1, sentence2):
        # TODO: Implement LSTM forward pass
        pass
        sentence1_embedding=self.embeddings_1(sentence1)
        sentence2_embedding=self.embeddings_1(sentence2)
        h1, _ = self.lstm1(sentence1_embedding)
        h2, _ = self.lstm2(sentence2_embedding)
        #h = torch.cat((h1[:, -1, :], h2[:, -1, :]), dim=1)
        h1_mean = torch.mean(h1, dim=1)
        h2_mean = torch.mean(h2, dim=1)
        concatenated_output = torch.cat((h1_mean, h2_mean), dim=1)
        h_out_1 = self.linear(concatenated_output)
        y = self.sigmoid(h_out_1)
        return y
