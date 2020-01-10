import os
import time
import argparse

import torch
import torch.nn as nn
from utils import download_data, read_and_preprocess_text
from dataset import CharacheterDataset
from engine import TextGenerator, train_one_epoch, test_model, generate_text

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

uurl = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'


parser = argparse.ArgumentParser()
parser.add_argument("-n_epochs", "--n_epochs", default=15, help="Number of epochs")
parser.add_argument("-seq_len", "--seq_len", default=10, help="The length of sequences used during training")
parser.add_argument("-emb_dim", "--emb_dim", default=256, help="Dimension of the embedding space")
parser.add_argument("-hid_dim", "--hid_dim", default=512, help="Dimension of LSTM units")
parser.add_argument("-bz", "--bz", default=64, help="Batch size")
parser.add_argument("-lr", "--lr", default=1e-4, help="Learning rate")
parser.add_argument("--download", "-download", default=False, help="Boolean of either download data or no")
parser.add_argument("--train_size", "-test_size", default=0.2,
                    help="The size of data to use for train, before doing train-val split")
parser.add_argument("--path_to_data", default='./data', help="Path to data directory")
parser.add_argument("--path_to_logdir", default='./logdir', help="Path to log directory")
parser.add_argument("--path_to_model", default="./models", help="Path to log model")
parser.add_argument("--to_train", help="Boolean indicating if we will train the model or no")
parser.add_argument("--result_file", default='result.txt', help="Path to file where to write results")
parser.add_argument('input_text', type=str, help='Input sentence')
parser.add_argument('--n_generate', "n-generate", default=200, help="Size of the generated text")


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_text = os.path.join(args.path_to_data, 'shakespeare.txt')

    if not os.path.exists(args.path_to_model):
        os.makedirs(args.path_to_model)

    if not os.path.exists(args.path_to_logdir):
        os.makedirs(args.path_to_logdir)

    if not os.path.exists(args.path_to_data):
        os.makedirs(args.path_to_data)

    if args.download:
        download_data(args.path_to_data, uurl)

    text_as_int, char2idx, idx2char = read_and_preprocess_text(path_to_text)
    vocab_size = len(char2idx)

    dataset = CharacheterDataset(text_as_int, args.seq_len)
    train_size = int(len(dataset) * args.train_size)
    trainset, valset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(trainset, shuffle=True, batch_size=args.bz)
    val_loader = DataLoader(valset, shuffle=False, batch_size=args.bz)

    model = TextGenerator(embedding_dim=args.emb_dim, hidden_dim=args.hid_dim, vocab_size=vocab_size)
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    timestr = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.path_to_logdir, timestr))

    best_acc = 0.
    for epoch in range(0, args.n_epochs):
        print("Epoch %d" % epoch)
        train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        acc = test_model(model, val_loader, criterion, writer, device, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, os.path.join(args.path_to_model, "text_gen_best.pth"))

        writer.close()

    text_generated = generate_text(model, args.input_text, char2idx, idx2char, device, num_generate=args.n_generate)

    res_file = open(args.result_file, "w")
    res_file.write(args.input_text + "".join(text_generated))
    res_file.close()
