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

    
if __name__=="__main__":
    batch_size=2
    seq_len=4
    d_model=8

    Q=torch.rand(batch_size,seq_len,d_model)
    K=torch.rand(batch_size,seq_len,d_model)
    V=torch.rand(batch_size,seq_len,d_model)

    attention=ScaledDotProductAttention()

    output,weights=attention(Q,K,V)

    print("Output shape:",output.shape)
    print("Attention weights shape:",weights.shape)