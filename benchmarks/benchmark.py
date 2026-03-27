import time
import torch

from models.cnn import CNN
from models.transformer import TransformerClassifier
from utils.dataset import get_dataloaders
from training.train import train

def benchmark_training(model,model_name):
    train_loader,_=get_dataloaders(batch_size=32)

    model.train()

    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    start_time=time.perf_counter()
    train(model,train_loader,optimizer,criterion)
    end_time=time.perf_counter()

    training_time=end_time-start_time
    print(f"{model_name} training time for 1 epoch:{training_time:.2f} seconds")

def benchmark_single_inference():
    runs=100

    cnn=CNN()
    transformer = TransformerClassifier()

    cnn.eval()
    transformer.eval()

    dummy_input=torch.randn(1,1,28,28)

    #CNN
    start_time=time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            cnn(dummy_input)
    end_time=time.perf_counter()
    cnn_latency=(end_time-start_time)*1000 / runs

    #Transformer
    start=time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            transformer(dummy_input)
    end=time.perf_counter()
    transformer_latency=(end-start)*1000 / runs

    print(f"CNN single image inference latency:{cnn_latency:.3f}ms")
    print(f"Transformer single image inference latency:{transformer_latency:.3f}ms")


def benchmark_batch_inference():
    runs=100

    cnn=CNN()
    transformer=TransformerClassifier()

    cnn.eval()
    transformer.eval()

    dummy_batch=torch.randn(32,1,28,28)

    #CNN
    start_time=time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            cnn(dummy_batch)
    end_time=time.perf_counter()
    cnn_latency=(end_time-start_time)*1000/runs

    #Transformer
    start=time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            transformer(dummy_batch)
    end=time.perf_counter()
    transformer_latency=(end-start)*1000/runs

    print(f"CNN Batch inference latency: {cnn_latency:.3f}ms")
    print(f"Transformer batch inference latency: {transformer_latency:.3f}ms")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_parameters():
    cnn=CNN()
    transformer=TransformerClassifier()

    cnn_params=count_parameters(cnn)
    transformer_params=count_parameters(transformer)

    print(f"CNN parameters:{cnn_params}")
    print(f"Transformer parameters:{transformer_params}")

if __name__=="__main__":
    print("Running Benchmarks...")
    print("-"*40)

    # TRAINING
    benchmark_training(CNN(),"CNN")
    benchmark_training(TransformerClassifier(),"Transformer")
    print("-"*40)

    # INFERENCE
    benchmark_single_inference()
    print("-"*40)

    benchmark_batch_inference()
    print("-"*40)

    # PARAMETERS
    benchmark_parameters()