import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN

from tqdm import tqdm
import configargparse
import time
import numpy as np
import pandas as pd
import os

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
# for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq
from trainers import get_trainer

import wandb
wandb.init(project='graph-aug')



def main():
    # fmt: off
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                    description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)

    parser.add_argument('--task', type=str, default='code')
    parser.add_argument('--data_root', type=str, default='/data/zhwu/ogb')
    parser.add_argument('--aug', type=str, default='baseline',
                        help='augment method to use [baseline|flag]')
                        
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test-freq', type=int, default=1)
    # fmt: on

    args, _ = parser.parse_known_args()
    
    # Setup Trainer and add customized args
    trainer = get_trainer(args)
    trainer.add_args(parser)
    train, eval = trainer.train, trainer.eval

    args = parser.parse_args()

    run_name = f'{args.task}+{args.gnn}+{args.aug}'
    wandb.run.name = run_name
    wandb.run.save()

    device = torch.device("cuda") if torch.cuda.is_available() and args.device >= 0 else torch.device("cpu")

    # automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.data_root)

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(
        args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)))

    split_idx = dataset.get_idx_split()

    # building vocabulary for sequence predition. Only use training data.
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    # set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose(
        [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    # Encoding node features into emb_dim vectors.
    # The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes=len(
        nodetypes_mapping['type']), num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)

    def run(run_id):
        best_val, final_test = 0, 0
        if args.gnn == 'gin':
            model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder, num_layer=args.num_layer,
                        gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder, num_layer=args.num_layer,
                        gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        elif args.gnn == 'gcn':
            model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder, num_layer=args.num_layer,
                        gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder, num_layer=args.num_layer,
                        gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        else:
            raise ValueError('Invalid GNN type')

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}=====".format(epoch))
            print('Training...')
            loss = train(model, device, train_loader, optimizer, args)

            if epoch > args.epochs // 2 and epoch % args.test_freq == 0 or epoch in [0, args.epochs]:
                print('Evaluating...')
                train_perf = eval(model, device, train_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
                valid_perf = eval(model, device, valid_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
                test_perf = eval(model, device, test_loader, evaluator,
                                arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

                # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

                train_metric, valid_metric, test_metric = train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric]
                wandb.log({f'train/{dataset.eval_metric}': train_metric,
                        f'valid/{dataset.eval_metric}': valid_metric,
                        f'test/{dataset.eval_metric}': test_metric,
                        'epoch': epoch,
                        'run_id': run_id})

                if best_val > valid_metric:
                    best_val = valid_metric
                    final_test = test_metric
        return best_val, final_test

    vals, tests = [], []
    for run_id in range(args.runs):
        best_val, final_test = run(run_id)
        vals.append(best_val)
        tests.append(final_test)
        print(f'Run {run} - val: {best_val}, test: {final_test}')
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")

if __name__ == "__main__":
    main()
