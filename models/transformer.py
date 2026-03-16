import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()

        pe=torch.zeros(max_len,d_model)

        position=torch.arange(0,max_len).unsqueeze(1)

        div_term=torch.exp(
            torch.arange(0,d_model,2)*
            (-math.log(10000.0)/d_model)
        )

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer("pe",pe)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1)]

        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,Q,K,V):
        d_k=Q.size(-1)
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(d_k)
        attention_weights=torch.softmax(scores,dim=-1)
        output=torch.matmul(attention_weights,V)

        return output,attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        assert d_model % num_heads==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)

        self.attention=ScaledDotProductAttention()

        self.W_o=nn.Linear(d_model,d_model)

    def split_heads(self,x):
        batch_size,seq_len,d_model=x.size()
        x=x.view(batch_size,seq_len,self.num_heads,self.d_k)
        x=x.transpose(1,2)
        return x
    
    def forward(self,Q,K,V):
        batch_size=Q.size(0)

        Q=self.W_q(Q)
        K=self.W_k(K)
        V=self.W_v(V)
        print("Q:",Q.shape)
        print("K:",K.shape)
        print("V:",V.shape)


        Q=self.split_heads(Q)
        K=self.split_heads(K)
        V=self.split_heads(V)
        print("Q:",Q.shape)
        print("K:",K.shape)
        print("V:",V.shape)

        output,attention_weights=self.attention(Q,K,V)

        output=output.transpose(1,2)

        output=output.contiguous().view(batch_size,-1,self.d_model)

        output=self.W_o(output)

        return output
    
if __name__=="__main__":
    batch_size=2
    seq_len=5
    d_model=64
    num_heads=8

    x=torch.rand(batch_size,seq_len,d_model)

    mha=MultiHeadAttention(d_model,num_heads)

    out=mha(x,x,x)

    print("Input shape:",x.shape)
    print("Output shape:",out.shape)


   