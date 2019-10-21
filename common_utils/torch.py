import csv
from collections import Counter

import numpy as np

import torch


def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]

    if isinstance(obj, dict):
        return {k: to_device(o, device) for k, o in obj.items()}

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = obj.to(device)
    return obj


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def load_weights(model, filename):
    map_location = None

    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        def map_location(storage, loc): return storage

    state_dict = torch.load(str(filename), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)


def save_weights(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), str(filename))


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking).long()

    lengths = masks.sum(dim=dim)

    return lengths


def softmax_masked(inputs, mask, dim=1, epsilon=0.000001):
    inputs_exp = torch.exp(inputs)
    inputs_exp = inputs_exp * mask.float()
    inputs_exp_sum = inputs_exp.sum(dim=dim)
    inputs_attention = inputs_exp / (inputs_exp_sum.unsqueeze(dim) + epsilon)

    return inputs_attention


def get_trainable_parameters(parameters):
    parameters = list(parameters)
    nb_params_before = sum(p.nelement() for p in parameters)

    parameters = [p for p in parameters if p.requires_grad]
    nb_params_after = sum(p.nelement() for p in parameters)

    print(f'Parameters: {nb_params_before} -> {nb_params_after}')
    return parameters


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=1):
    pin_memory = torch.cuda.is_available()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, num_layers=1, bidirectional=False, return_sequence=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence
        self.dropout = dropout

        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True
        )

    def zero_state(self, batch_size):
        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = self.num_layers if not self.bidirectional else self.num_layers * 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        # shape: (num_layers, batch_size, hidden_dim)
        h = to_device(torch.zeros(*state_shape))

        # shape: (num_layers, batch_size, hidden_dim)
        c = torch.zeros_like(h)

        return h, c

    def forward(self, inputs, lengths):
        batch_size = inputs.shape[0]

        # shape: (num_layers, batch_size, hidden_dim)
        h, c = self.zero_state(batch_size)

        lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[inputs_sorted_idx]

        # pack sequences
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths_sorted.detach(), batch_first=True)

        # shape: (batch_size, sequence_len, hidden_dim)
        outputs, (h, c) = self.lstm(packed, (h, c))

        # concatenate if bidirectional
        # shape: (batch_size, hidden_dim)
        h = torch.cat([x for x in h], dim=-1)

        # unpack sequences
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
        outputs = outputs[inputs_unsorted_idx]
        h = h[inputs_unsorted_idx]

        if self.return_sequence:
            return outputs
        else:
            return h


class Vocab(object):
    END_TOKEN = '<end>'
    START_TOKEN = '<start>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self, special_tokens=None):
        super().__init__()

        self.special_tokens = special_tokens

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        if self.special_tokens is not None:
            self.add_document(self.special_tokens)

    def add_document(self, document, rebuild=True):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        if rebuild:
            self._rebuild_id2token()

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc, rebuild=False)

        self._rebuild_id2token()

    def prune_vocab(self, max_size):
        nb_tokens_before = len(self.token2id)

        tokens_all = set(self.token2id.keys())
        tokens_special = set(self.special_tokens)
        tokens_most_common = set(t for t, c in self.token_counts.most_common(max_size)) - tokens_special
        tokens_to_delete = tokens_all - tokens_most_common - tokens_special

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {}
        for i, token in enumerate(self.special_tokens):
            self.token2id[token] = i
        for i, token in enumerate(tokens_most_common):
            self.token2id[token] = i + len(self.special_tokens)

        self._rebuild_id2token()

        nb_tokens_after = len(self.token2id)

        print(f'Vocab pruned: {nb_tokens_before} -> {nb_tokens_after}')

    def _rebuild_id2token(self):
        self.id2token = {i: t for t, i in self.token2id.items()}

    def get(self, item, default=None):
        return self.token2id.get(item, default)

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return len(self.token2id)

    def __str__(self):
        return f'{len(self)} tokens'

    def save(self, filename):
        with open(filename, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['token', 'counts', 'is_special'])
            writer.writeheader()
            for idx in range(len(self.token2id)):
                token = self.id2token[idx]
                is_special = 1 if token in self.special_tokens else 0
                writer.writerow({'token': token, 'counts': self.token_counts[token], 'is_special': is_special})

    @staticmethod
    def load(filename):
        with open(filename, 'r') as csv_file:
            token2id = {}
            tokens_counts = {}
            special_tokens = []
            reader = csv.DictReader(csv_file)
            for i, row in enumerate(reader):
                token2id[row['token']] = i
                tokens_counts[row['token']] = int(row['counts'])
                if bool(int(row['is_special'])):
                    special_tokens.append(row['token'])

        vocab = Vocab()
        vocab.token2id = token2id
        vocab.token_counts = Counter(tokens_counts)
        vocab.special_tokens = special_tokens
        vocab._rebuild_id2token()

        return vocab
