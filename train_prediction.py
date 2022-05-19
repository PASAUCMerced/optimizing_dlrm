import argparse
import torch
import json
import os
import editdistance
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from seq2seq_prefetching import Seq2Seq
from torch.utils import data
import pandas as pd
import numpy as np

#processing data: chuckization, chunksize is 1
def data(input_file):
        indices, offsets, lengths = torch.load(input_file)
        print(f"Data file indices = {indices.size()}", f"offsets = {offsets.size()}, lengths = {lengths.size()}) ")
        return indices


def train(model, optimizer, train_loader, state):
    epoch, n_epochs, train_steps = state

    losses = []
    cers = []

    # t = tqdm.tqdm(total=min(len(train_loader), train_steps))
    t = tqdm.tqdm(train_loader)
    model.train()

    for batch in t:
        t.set_description("Epoch {:.0f}/{:.0f} (train={})".format(epoch, n_epochs, model.training))
        loss, _, _, _ = model.loss(batch)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))
        t.update()

    return model, optimizer
    # print(" End of training:  loss={:05.3f} , cer={:03.1f}".format(np.mean(losses), np.mean(cers)*100))


def evaluate(model, eval_loader):

    losses = []
    accs = []

    t = tqdm.tqdm(eval_loader)
    model.eval()

    with torch.no_grad():
        for batch in t:
            t.set_description(" Evaluating... (train={})".format(model.training))
            loss, logits, labels, alignments = model.loss(batch)
            preds = logits.detach().cpu().numpy()
            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            acc = 100 * editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)
            losses.append(loss.item())
            accs.append(acc)
            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()
        align = alignments.detach().cpu().numpy()[:, :, 0]

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig("data/att.png")
    print("  End of evaluation : loss {:05.3f} , acc {:03.1f}".format(np.mean(losses), np.mean(accs)))
    # return {'loss': np.mean(losses), 'cer': np.mean(accs)*100}

def run(dataset, eval_dataset):
    USE_CUDA = torch.cuda.is_available()

    config_path = os.path.join("experiments", FLAGS.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    config["gpu"] = torch.cuda.is_available()

    BATCHSIZE = 30
    train_loader = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
                                  drop_last=True)
    config["batch_size"] = BATCHSIZE

    # Models
    model = Seq2Seq(config)

    if USE_CUDA:
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    print("=" * 60)
    print(model)
    print("=" * 60)
    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(" (" + k + ") : " + str(v))
    print()
    print("=" * 60)

    print("\nInitializing weights...")
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

    for epoch in range(FLAGS.epochs):
        run_state = (epoch, FLAGS.epochs, FLAGS.train_size)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer = train(model, optimizer, train_loader, run_state)
        evaluate(model, eval_loader)

        # TODO implement save models function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('traceFile', type=str,  help='trace file name\n')
    #parser.add_argument('--n', default=10,type=int,  help='input sequence length N\n')
    #parser.add_argument('--m', default=50,type=int,  help='output sequence length\n')
    #parser.add_argument('--w', default=50,type=int,  help='prediction window size length\n')
    args = parser.parse_args() 

    traceFile = args.traceFile
    FLAGS, _ = parser.parse_known_args()
    
    gt_trace = traceFile[0:traceFile.rfind(".pt")] + "_dataset_cache_miss_trace.csv"
    dataset = data(traceFile) 
    csvdata = pd.read_csv(gt_trace)
    gt = csvdata[1].tolist()
    #ensure the training and groudtruth has the same size. When we processing groundtruth, we cutout some data
    dataset = dataset[:len(gt)]

    run(dataset, gt)