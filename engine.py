import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class TextGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2char = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.char_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        char_space = self.hidden2char(lstm_out)
        char_scores = F.log_softmax(char_space, dim=2)
        return char_scores


def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    for batch_id, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.long(), targets.long()

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).permute(0, 2, 1)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_id % 100 == 99:
            writer.add_scalar('training loss',
                              running_loss / 100,
                              epoch * len(loader) + batch_id)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * inputs.shape[0], len(loader.dataset),
                       100. * batch_id / len(loader), running_loss / 100))

            running_loss = 0.0

            writer.flush()

    writer.add_scalar('Train/Loss', loss.item(), epoch)
    writer.flush()


def test_model(model, loader, criterion, writer, device, epoch):
    model.eval()
    i, loss, correct, n = [0, 0, 0, 0]

    print("Testing..")
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.long(), targets.long()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs).permute(0, 2, 1)

            loss += criterion(outputs, targets)

            preds = outputs.data.max(1)[1]
            correct += preds.eq(targets.data).cpu().sum() / targets.shape[1]

    loss /= len(loader)  # loss function already averages over batch size
    accuracy = 100. * correct / len(loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader.dataset),
        accuracy))

    # Record loss and accuracy into the writer
    writer.add_scalar('Test/Loss', loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    writer.flush()
    return accuracy


def generate_text(model, start_string, char2idx, idx2char, device, num_generate=1000):
    input_eval = torch.tensor(np.array([char2idx[c] for c in start_string])).long()
    input_eval = input_eval.unsqueeze(0).to(device)

    text_generated = []

    for i in range(num_generate):
        outputs = model(input_eval).squeeze().cpu()

        if i > 0:
            prediction = torch.multinomial(F.softmax(outputs), num_samples=1)
        if i == 0:
            prediction = torch.multinomial(F.softmax(outputs), num_samples=1)[-1]

        text_generated.append(idx2char[prediction.cpu().numpy()])

        input_eval = prediction.unsqueeze(0).to(device)

    text_generated = list(np.hstack(text_generated))

    return text_generated

