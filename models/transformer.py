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
    
if __name__=="__main__":
    d_model=512
    seq_len=10
    batch_size=2

    x=torch.zeros(batch_size,seq_len,d_model)

    pe=PositionalEncoding(d_model)

    out=pe(x)

    print("Input shape:",x.shape)
    print("Output shape:",out.shape)