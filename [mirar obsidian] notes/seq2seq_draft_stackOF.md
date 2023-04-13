Hello, I'm trying to implement a simple seq2seq with atention model, and the problem I am testing the model on, it is a very simple vector inversion (5,2,8,4) --> (4,8,2,5), just with numbers and without embeddings.

I am following some code from:

- https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53 
- https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Python/blob/master/Chapter08/nmt_rnn_attention/rnn_attention.ipynb

The model "works", so I can do feedforward and backprop, but it does not train, and I really think that it must be a problem with my code, but I dont know where.

Model:


```
# seq2seq with attention

class EncoderRNN(nn.Module):

    def __init__(self,d_input,d_hidden):
        super().__init__()
        self.d_input=d_input
        self.d_hidden=d_hidden
        self.relu=nn.ReLU()
        self.dense1=nn.Linear(d_input,d_hidden)
        self.rnn=nn.GRU(d_hidden,d_hidden,batch_first=True)

    def forward(self,x,hidden):
        x=self.relu(self.dense1(x)) #give B x 1 x input,  get B x 1 x hidden_dim
        output,hidden_out=self.rnn(x,hidden) #give B x 1 x hidden_dim, get 1 x B x hidden_dim 
        return output,hidden_out #give B x len_seq x hidden_dim, get 1 x B x hidden_dim
    
    def init_hidden(self,batch_size=1):
        return torch.zeros(1,batch_size,self.d_hidden)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,hidden_state,encoder_outputs):
        #encoder_outputs:  Batch x len_seq x hidden_dim, hidden state : 1 x B x Hidden_dim, context : B x 1 x hidden_dim and attn_weights B x 1 x len_seq


        hidden_state_transpose=hidden_state.transpose(0,1) # now is B x 1 x hidden
        #encoder_outputs: Batch x len_seq x hidden_dim
        encoder_outputs_transpose=encoder_outputs.transpose(1,2) # Batch x  hidden_dim x len_seq 

        e=torch.bmm(hidden_state_transpose,encoder_outputs_transpose) # B x 1 x len_seq
        att_weights=F.softmax(e,dim=2) # B x 1 x len_seq 
        context=torch.bmm(att_weights,encoder_outputs) #  B x 1 x hidden_dim
        return context, att_weights # return  B x 1 x hidden_dim y B x 1 x len_seq 



class DecoderRNN(nn.Module):

    def __init__(self,input_dim,hidden_size):
        super().__init__()
        self.d_input=input_dim
        self.d_output=input_dim
        self.d_hidden=hidden_size
        self.dense_in=nn.Linear(input_dim,hidden_size)
        self.dense_out1=nn.Linear(hidden_size*2,hidden_size)
        self.dense_out2=nn.Linear(hidden_size,input_dim) # assume dim_input = dim_output

        self.rnn=nn.GRU(hidden_size,hidden_size,batch_first=True)

        self.attn=Attention()

    def forward(self,prev_input,hidden_state,encoder_outputs):

        x=F.relu(self.dense_in(prev_input)) # in B x 1 x input_dim, out B x 1 x Hidden_dim, 

        _,new_hidden_state=self.rnn(x,hidden_state) #in y_(t-1)=B x 1 x Hidden_dim and h_(t-1)= 1 x Batch_size x Hidden_dim and out y_t = B x 1 x Hidden_dim and h_t = 1 x B x Hidden_dim


        #encoder_outputs:  Batch xlen_seq x hidden_dim , hidden state: Q es 1 x B x Hidden_dim,  context: B x 1 x hidden_dim,  attn_weights: B x 1 x len_seq
        context,attn_weigths=self.attn(new_hidden_state,encoder_outputs) # return  B x 1 x hidden_dim y B x 1 x len_seq 
        new_hidden_state_transpose=new_hidden_state.transpose(0,1)
        out=self.dense_out1(torch.cat((new_hidden_state_transpose,context),dim=2))# le metemos B x 1 x 2*hidden_dim y sacamos B x 1 x hidden_dim
        out1=self.dense_out2(out) #in B x 1 x hidden_dim ,out B x 1 x input_dim

        return out1,new_hidden_state, attn_weigths #B x 1 x input_dim , 1 x B x Hidden_dim y B x 1 x len_seq
    
    def init_hidden(self,batch_size=1):
        return torch.zeros(1,batch_size,self.d_hidden)



```


The dataset is very simple:

```
class DataFlip(Dataset):
    def __init__(self,seq_len,input_dim,max_dim):
        super().__init__()
        self.seq_len=seq_len
        self.input_dim=input_dim
        self.max_dim=max_dim

    def __getitem__(self, index):
        x=torch.randint(0,50,(self.seq_len,self.input_dim)).float()
        y=x.flip(1)
        return x,y
    def __len__(self):
        return self.max_dim
```

And finally, and probably where the error is, the training loop:



```
len_seq=7
dataset=DataFlip(len_seq,1,32*10)
dataloader=DataLoader(dataset,batch_size=32)

hidden_dim=15
encoder=EncoderRNN(1,hidden_dim)
decoder=DecoderRNN(1,hidden_dim)

optim_enc=optim.Adam(encoder.parameters(),lr=0.01)
optim_dec=optim.Adam(decoder.parameters(),lr=0.01)
loss_criterion=nn.MSELoss()

batch_size,len_seq,input_dim=x_train.size()


for epoch in range(300):
    total_loss=0
    for j,(x_train,y_train) in enumerate(dataloader):
        optim_dec.zero_grad()
        optim_enc.zero_grad()

        loss=torch.tensor(0.0)
        
        hidden_state_encoder=encoder.init_hidden(batch_size)


        list_encoder_hidden_states,last_encoder_hidden=encoder(x_train,hidden_state_encoder)
    
        #just give some vector that indicates the start of the answer
        prev_input=torch.ones(batch_size,1,input_dim)*-1.0

        for i in range(len_seq):
            out1,new_hidden_state, attn_weigths=decoder(prev_input,last_encoder_hidden,list_encoder_hidden_states)
            last_encoder_hidden=new_hidden_state
            prev_input=y_train[:,[i]]
            loss+=loss_criterion(out1,y_train[:,[i]])

        loss.backward()
        optim_dec.step()
        optim_enc.step()
        total_loss+=loss.item()
    print(epoch,total_loss)

```
