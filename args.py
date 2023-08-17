import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MRIQA")
    parser.add_argument("--mode", type=str, default="fourlier", help="one of ['origin', 'augmentation', 'fourlier']")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size per GPU")
    parser.add_argument("--projection_dim", type=int, default=128, help="output dimention of SimCLR's projector")
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    
    parser.add_argument('--temperature', type=float, default=0.9, help='temperature parameter')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--nprocs", type=int, default=8, help="number of process")  # one process is resposible for one GPU, so it is also equal to the number of GPUs
    parser.add_argument("--workers", type=int, default=0, help="number of workers in DataLoader")
    
    args = parser.parse_args()
    args.ngpus_per_proc = 1  # the number of gpu for each process
    return args

args = parse_args()
