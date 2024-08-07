from torch_geometric.loader import DataLoader
import torch
import numpy as np
import pickle
import random
from networks import KPooling, GNN
from pytorch_metric_learning import losses, miners
from pytorch_tools import EarlyStopping
import sys
import getopt
import os
from argparse import ArgumentParser
from util import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument( "--seed", type=int, default=1, help="Seed" ) 
    parser.add_argument( "--dataset", type=str,required=True, default='Facebook', help="Dataset" ) 
    parser.add_argument( "--pooling", type=bool, default=True, help="Pooling" ) 
    parser.add_argument( "--ratio", type=float, default=0.8, help="Ratio" ) 
    parser.add_argument( "--embedding_size", type=int, default=30,required=True, help="Embedding Size" ) 
    parser.add_argument( "--temperature", type=float, default=0.1, help="Temperature" ) 
    parser.add_argument( "--output_size", type=int, default=10,required=True, help="Output Size" ) 
    parser.add_argument( "--budget", type=int, default=100, help="Budget" ) 
    parser.add_argument( "--batch_size", type=int, default=128, help="Batch Size" ) 
    parser.add_argument( "--learning_rate", type=float, default=1e-3, help="Learning Rate" ) 
    parser.add_argument( "--metric", type=str, default='distance', help="Metric" )

    
    args = parser.parse_args()


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    

    # graph = "wiki_train"
    graph=args.dataset
    pooling = args.pooling
    ratio = args.ratio
    embedding_size = args.embedding_size
    temperature = args.temperature
    output_size = args.output_size
    budget = args.budget
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    metric = args.metric



    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ratio = 0.8
    root_folder='../../data/LeNSE/MVC/train'

    data=load_from_pickle(os.path.join(root_folder,f"{graph}/budget_{budget}/graph_data"))

    
    random.shuffle(data)
    data = data[:2500]
    data = [d.to(device) for d in data]
    n = int(len(data) * train_ratio)


    train_data = data[:n]
    val_data = data[n:]
    del data

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    if pooling:
        encoder = KPooling(ratio, 2, embedding_size, output_size).to(device)
    else:
        encoder = GNN(2, embedding_size, output_size).to(device)
    optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    loss_fn = losses.NTXentLoss(temperature)
    miner = miners.MultiSimilarityMiner()
    es = EarlyStopping(patience=10, percentage=False)

    losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = ""

    for epoch in range(1000):
        epoch_train_loss = []
        epoch_val_loss = []

        for count, batch in enumerate(train_loader):
            optimiser.zero_grad()
            inputs = encoder.forward(batch)
            hard_pairs = miner(inputs, batch.y)
            loss = loss_fn(inputs, batch.y, hard_pairs)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimiser.step()

        for batch in val_loader:
            with torch.no_grad():
                inputs = encoder.forward(batch)
                loss = loss_fn(inputs, batch.y)
                epoch_val_loss.append(loss.item())

        train_loss = np.mean(epoch_train_loss)
        val_loss = np.mean(epoch_val_loss)

        losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}:\n"
            f"Train loss -- {train_loss:.3f}\n"
            f"Val loss -- {val_loss:.3f}\n")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            encoder_folder = os.path.join(root_folder, graph, f"budget_{budget}", "encoder")
            os.makedirs(encoder_folder, exist_ok=True)
            best_model_path = os.path.join(encoder_folder, "encoder.pth")
            torch.save(encoder, best_model_path)
            print(f"Best encoder model saved at: {best_model_path}")

        if es.step(torch.FloatTensor([val_losses[-1]])) and epoch > 20:
            break

    # Print a message indicating the best model has been saved
    print(f"Training complete. Best encoder model saved at: {best_model_path}")

    # losses = []
    # val_losses = []
    # for epoch in range(1000):
    #     epoch_train_loss = []
    #     epoch_val_loss = []
    #     for count, batch in enumerate(train_loader):
    #         optimiser.zero_grad()
    #         inputs = encoder.forward(batch)
    #         hard_pairs = miner(inputs, batch.y)
    #         loss = loss_fn(inputs, batch.y, hard_pairs)
    #         epoch_train_loss.append(loss.item())
    #         loss.backward()
    #         optimiser.step()

    #     for batch in val_loader:
    #         with torch.no_grad():
    #             inputs = encoder.forward(batch)
    #             loss = loss_fn(inputs, batch.y)
    #             epoch_val_loss.append(loss.item())

    #     losses.append(np.mean(epoch_train_loss))
    #     val_losses.append(np.mean(epoch_val_loss))
    #     print(f"Epoch {epoch+1}:\n"
    #         f"Train loss -- {losses[-1]:.3f}\n"
    #         f"Val loss -- {val_losses[-1]:.3f}\n")

    #     if es.step(torch.FloatTensor([val_losses[-1]])) and epoch > 20:
    #         break


    # # Create the full path to the encoder folder
    # encoder_folder = os.path.join(root_folder,graph, f"budget_{budget}", "encoder")

    # # Create all necessary folders if they don't exist
    # os.makedirs(encoder_folder, exist_ok=True)

    # # Save the encoder model
    # encoder_path = os.path.join(encoder_folder, "encoder")


    # # torch.save(encoder, f"{graph}/budget_{budget}/encoder/encoder")
    # torch.save(encoder,encoder_path)

    # # Print a message indicating that the file has been saved
    # print(f"Encoder model saved at: {encoder_path}")