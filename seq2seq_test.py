import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.autograd import Variable

from iou_loss import iou_pytorch

class EncoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    super(EncoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    self.input_size = input_size

    # Output size of the word embedding NN
    self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Regularization parameter
    self.dropout = nn.Dropout(p)
    self.tag = True

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(self.input_size, self.embedding_size)
    
    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p)

  # Shape of x (26, 32) [Sequence_length, batch_size]
  def forward(self, x):

    # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
    embedding = self.dropout(self.embedding(x))
    
    # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]
    outputs, (hidden_state, cell_state) = self.LSTM(embedding)

    return hidden_state, cell_state




class DecoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
    super(DecoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    self.input_size = input_size

    # Output size of the word embedding NN
    self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)
    self.output_size = output_size

    # Regularization parameter
    self.dropout = nn.Dropout(p)
    self.tag = True

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(self.input_size, self.embedding_size)

    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p)

    # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]
    self.fc = nn.Linear(self.hidden_size, self.output_size)

  # Shape of x (32) [batch_size]
  def forward(self, x, hidden_state, cell_state):

    # Shape of x (1, 32) [1, batch_size]
    x = x.unsqueeze(0)

    # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]
    embedding = self.dropout(self.embedding(x))

    # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)
    outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))

    # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]
    predictions = self.fc(outputs)

    # Shape --> predictions (32, 4556) [batch_size , output_size]
    predictions = predictions.squeeze(0)

    return predictions, hidden_state, cell_state

'''
class Seq2Seq(nn.Module):
  def __init__(self, Encoder_LSTM, Decoder_LSTM):
    super(Seq2Seq, self).__init__()
    self.Encoder_LSTM = Encoder_LSTM
    self.Decoder_LSTM = Decoder_LSTM

  def forward(self, source, target, tfr=0.5):
    # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
    batch_size = source.shape[1]

    # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
    target_len = target.shape[0]
    target_vocab_size = len(english.vocab)
    
    # Shape --> outputs (14, 32, 5766) 
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

    # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
    hidden_state_encoder, cell_state_encoder = self.Encoder_LSTM(source)

    # Shape of x (32 elements)
    x = target[0] # Trigger token <SOS>

    for i in range(1, target_len):
      # Shape --> output (32, 5766) 
      output, hidden_state_decoder, cell_state_decoder = self.Decoder_LSTM(x, hidden_state_encoder, cell_state_encoder)
      outputs[i] = output
      best_guess = output.argmax(1) # 0th dimension is batch size, 1st dimension is word embedding
      x = target[i] if random.random() < tfr else best_guess # Either pass the next word correctly from the dataset or use the earlier predicted word

    # Shape --> outputs (14, 32, 5766) 
    return outputs
'''

class Seq2Seq(nn.Module):
    """
        Sequence to sequence module
    """

    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.SOS = config.get("start_index", 1),
        self.vocab_size = config.get("n_classes", 32)
        self.batch_size = config.get("batch_size", 1)
        self.sampling_prob = config.get("sampling_prob", 0.)
        self.gpu = config.get("gpu", False)

        # Encoder
        if config["encoder"] == "PyRNN":
            self._encoder_style = "PyRNN"
            self.encoder = EncoderPyRNN(config)
        else:
            self._encoder_style = "RNN"
            self.encoder = EncoderRNN(config)

        # Decoder
        self.use_attention = config["decoder"] != "RNN"
        if config["decoder"] == "Luong":
            self.decoder = LuongDecoder(config)
        elif config["decoder"] == "Bahdanau":
            self.decoder = BahdanauDecoder(config)
        else:
            self.decoder = RNNDecoder(config)

        if config.get('loss') == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            config['loss'] = 'cross_entropy'
        elif config.get('loss') == 'Intersection_over_Union':
            self.loss_fn = iou_pytorch(ignore_index=0)
            config['loss'] = 'IoU'
        else:
            self.loss_fn = torch.nn.NLLLoss(ignore_index=0)
            config['loss'] = 'NLL'
        self.loss_type = config['loss']
        print(config)

    def encode(self, x, x_len):

        batch_size = self.batch_size
        init_state = self.encoder.init_hidden(batch_size)
        if self._encoder_style == "PyRNN":
            encoder_outputs, encoder_state, input_lengths = self.encoder.forward(x, init_state, x_len)
        else:
            encoder_outputs, encoder_state = self.encoder.forward(x, init_state, x_len)

        assert encoder_outputs.size()[0] == self.batch_size, encoder_outputs.size()
        assert encoder_outputs.size()[-1] == self.decoder.hidden_size

        if self._encoder_style == "PyRNN":
            return encoder_outputs, encoder_state.squeeze(0), input_lengths
        return encoder_outputs, encoder_state.squeeze(0)

    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):
        """
        Args:
            encoder_outputs: (B, T, H)
            encoder_hidden: (B, H)
            targets: (B, L)
            targets_lengths: (B)
            input_lengths: (B)
        Vars:
            decoder_input: (B)
            decoder_context: (B, H)
            hidden_state: (B, H)
            attention_weights: (B, T)
        Outputs:
            alignments: (L, T, B)
            logits: (B*L, V)
            labels: (B*L)
        """

        batch_size = encoder_outputs.size()[0]
        max_length = targets.size()[1]
        # decoder_attns = torch.zeros(batch_size, MAX_LENGTH, MAX_LENGTH)
        decoder_input = Variable(torch.LongTensor([self.SOS] * batch_size)).squeeze(-1)
        decoder_context = encoder_outputs.transpose(1, 0)[-1]
        decoder_hidden = encoder_hidden

        alignments = Variable(torch.zeros(max_length, encoder_outputs.size(1), batch_size))
        logits = Variable(torch.zeros(max_length, batch_size, self.decoder.output_size))

        if self.gpu:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            logits = logits.cuda()

        for t in range(max_length):

            # The decoder accepts, at each time step t :
            # - an input, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - encoder outputs, [B, T, H]

            check_size(decoder_input, self.batch_size)
            check_size(decoder_hidden, self.batch_size, self.decoder.hidden_size)

            # The decoder outputs, at each time step t :
            # - an output, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - weights, [B, T]

            if self.use_attention:
                check_size(decoder_context, self.batch_size, self.decoder.hidden_size)
                outputs, decoder_hidden, attention_weights = self.decoder.forward(
                    input=decoder_input.long(),
                    prev_hidden=decoder_hidden,
                    encoder_outputs=encoder_outputs,
                    seq_len=input_lengths)
                alignments[t] = attention_weights.transpose(1, 0)
            else:
                outputs, hidden = self.decoder.forward(
                    input=decoder_input.long(),
                    hidden=decoder_hidden)

            # print(outputs[0])
            logits[t] = outputs

            use_teacher_forcing = random.random() > self.sampling_prob

            if use_teacher_forcing and self.training:
                decoder_input = targets[:, t]

            # SCHEDULED SAMPLING
            # We use the target sequence at each time step which we feed in the decoder
            else:
                # TODO Instead of taking the direct one-hot prediction from the previous time step as the original paper
                # does, we thought it is better to feed the distribution vector as it encodes more information about
                # prediction from previous step and could reduce bias.
                topv, topi = outputs.data.topk(1)
                decoder_input = topi.squeeze(-1).detach()


        labels = targets.contiguous().view(-1)

        if self.loss_type == 'NLL': # ie softmax already on outputs
            mask_value = -float('inf')
            print(torch.sum(logits, dim=2))
        else:
            mask_value = 0

        logits = mask_3d(logits.transpose(1, 0), targets_lengths, mask_value)
        logits = logits.contiguous().view(-1, self.vocab_size)

        return logits, labels.long(), alignments

    @staticmethod
    def custom_loss(logits, labels):

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = 0
        mask = (labels > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        logits = logits[range(logits.shape[0]), labels] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(logits) / nb_tokens

        return ce_loss

    def step(self, batch):
        x, y, x_len, y_len = batch
        if self.gpu:
            x = x.cuda()
            y = y.cuda()
            x_len = x_len.cuda()
            y_len = y_len.cuda()

        if self._encoder_style == "PyRNN":
            encoder_out, encoder_state, x_len = self.encode(x, x_len)
        else:
            encoder_out, encoder_state = self.encode(x, x_len)
        logits, labels, alignments = self.decode(encoder_out, encoder_state, y, y_len, x_len)
        return logits, labels, alignments

    def loss(self, batch):
        logits, labels, alignments = self.step(batch)
        loss = self.loss_fn(logits, labels)
        # loss2 = self.custom_loss(logits, labels)
        return loss, logits, labels, alignments