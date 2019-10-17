import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
	def __init__(self):
		super(EncoderCNN, self).__init__()

		cnn = models.vgg19(pretrained = False)
		modules = list(cnn.children())[:-2]
		self.cnn = nn.Sequential(*modules)
		self.enc_dim = list(cnn.features.children())[-3].weight.shape[0]

	def forward(self, x):
		x = self.cnn(x) # (batch_size, enc_dim, enc_img_size, enc_img_size)
		x = x.permute(0, 2, 3, 1)
		return x

class AttentionVisual(nn.Module):
	def __init__(self, vis_enc_dim, sent_hidden_dim, att_dim):
		super(AttentionVisual, self).__init__()

		self.enc_att = nn.Linear(vis_enc_dim, att_dim)
		self.dec_att = nn.Linear(sent_hidden_dim, att_dim)
		self.tanh = nn.Tanh()
		self.full_att = nn.Linear(att_dim, 1)
		self.softmax = nn.Softmax(dim = 1)

	def forward(self, vis_enc_output, dec_hidden_state):
		vis_enc_att = self.enc_att(vis_enc_output)	# (batch_size, num_pixels, att_dim)
		dec_output = self.dec_att(dec_hidden_state) # (batch_size, att_dim)

		join_output = self.tanh(vis_enc_att + dec_output.unsqueeze(1)) # (batch_size, num_pixels, att_dim)

		join_output = self.full_att(join_output).squeeze(2) # (batch_size, num_pixels)

		att_scores = self.softmax(join_output) # (batch_size, num_pixels)

		att_output = torch.sum(att_scores.unsqueeze(2) * vis_enc_output, dim = 1)

		return att_output, att_scores

class AttentionSemantic(nn.Module):
	def __init__(self, sem_enc_dim, sent_hidden_dim, att_dim):
		super(AttentionSemantic, self).__init__()

		self.enc_att = nn.Linear(sem_enc_dim, att_dim)
		self.dec_att = nn.Linear(sent_hidden_dim, att_dim)
		self.tanh = nn.Tanh()
		self.full_att = nn.Linear(att_dim, 1)
		self.softmax = nn.Softmax(dim = 1)

	def forward(self, sem_enc_output, dec_hidden_state):
		sem_enc_output = self.enc_att(sem_enc_output)	# (batch_size, no_of_tags, att_dim)
		dec_output = self.dec_att(dec_hidden_state) # (batch_size, att_dim)

		join_output = self.tanh(sem_enc_output + dec_output.unsqueeze(1)) # (batch_size, no_of_tags, att_dim)

		join_output = self.full_att(join_output).squeeze(2) # (batch_size, no_of_tags)

		att_scores = self.softmax(join_output) # (batch_size, no_of_tags)

		att_output = torch.sum(att_scores.unsqueeze(2) * sem_enc_output, dim = 1)

		return att_output, att_scores


class SentenceLSTM(nn.Module):
	def __init__(self, vis_embed_dim, sent_hidden_dim, att_dim, sent_input_dim, word_input_dim, int_stop_dim):
		super(SentenceLSTM, self).__init__()

		self.vis_att = AttentionVisual(vis_embed_dim, sent_hidden_dim, att_dim)
		# self.sem_att = AttentionSemantic(sem_embed_dim, sent_hidden_dim, att_dim)

		# self.contextLayer = nn.Linear(vis_embed_dim + sem_embed_dim, cont_dim)
		self.contextLayer = nn.Linear(vis_embed_dim, sent_input_dim)
		self.lstm = nn.LSTMCell(sent_input_dim, sent_hidden_dim, bias = True)
		
		self.sent_hidden_dim = sent_hidden_dim
		self.word_input_dim = word_input_dim

		self.topic_hid_layer = nn.Linear(sent_hidden_dim, word_input_dim)
		self.topic_context_layer = nn.Linear(sent_input_dim, word_input_dim)
		self.tanh1 = nn.Tanh()

		self.stop_prev_hid = nn.Linear(sent_hidden_dim, int_stop_dim)
		self.stop_cur_hid = nn.Linear(sent_hidden_dim, int_stop_dim)
		self.tanh2 = nn.Tanh()
		self.final_stop_layer = nn.Linear(int_stop_dim, 2)

	def forward(self, vis_enc_output, captions, device):
		"""
        Forward propagation.

        :param vis_enc_output: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: captions, a tensor of dimension (batch_size, max_no_of_sent, max_sent_len)
        :return: topic vector for word LSTM (batch_size, max_no_of_sent, word_input_dim), stop vector for each time step (batch_size, max_no_of_sent, 2)
        """
		batch_size = vis_enc_output.shape[0]
		vis_enc_dim = vis_enc_output.shape[-1]

		vis_enc_output = vis_enc_output.view(batch_size, -1, vis_enc_dim) # (batch_size, num_pixels, vis_enc_dim)

		h = torch.zeros((batch_size, self.sent_hidden_dim)).to(device)
		c = torch.zeros((batch_size, self.sent_hidden_dim)).to(device)

		topics = torch.zeros((batch_size, captions.shape[1], self.word_input_dim)).to(device)
		ps = torch.zeros((batch_size, captions.shape[1], 2)).to(device)

		for t in range(captions.shape[1]):
			vis_att_output, vis_att_scores = self.vis_att(vis_enc_output, h) # (batch_size, vis_enc_dim), (batch_size, num_pixels)

			# can concat with the semantic attention module output
			context_output = self.contextLayer(vis_att_output) # (batch_size, sent_input_dim)

			h_prev = h.clone()

			h, c = self.lstm(context_output, (h, c)) # (batch_size, sent_hidden_dim), (batch_size, sent_hidden_dim)

			topic = self.tanh1(self.topic_hid_layer(h) + self.topic_context_layer(context_output)) # (batch_size, word_input_dim)

			p = self.tanh2(self.stop_prev_hid(h_prev) + self.stop_cur_hid(h)) # (batch_size, int_stop_dim)
			p = self.final_stop_layer(p) # (batch_size, 2)

			topics[:, t, :] = topic
			ps[:, t, :] = p

		return topics, ps

class WordLSTM(nn.Module):
	def __init__(self, word_input_dim, word_hidden_dim, vocab_size, num_layers = 1):
		super(WordLSTM, self).__init__()

		self.embedding = nn.Embedding(vocab_size, word_input_dim)
		self.lstm = nn.LSTM(word_input_dim, word_hidden_dim, num_layers, batch_first=True)

		self.fc = nn.Linear(word_hidden_dim, vocab_size)
		
	def forward(self, topic, caption):
		"""
        Forward propagation.

        :param topic: topic vector, a tensor of dimension (batch_size, word_input_dim)
        :param caption: a single sentence, a tensor of dimension (batch_size, max_sent_len)
        :return: outputs predicting the next word, a tensor of dimension (batch_size, max_sent_len, vocab_size)
        """
		embeddings = self.embedding(caption) # (batch_size, max_sent_len, word_input_dim)

		outputs, _ = self.lstm(torch.cat((topic.unsqueeze(1), embeddings), 1)) # (batch_size, max_sent_len + 1, word_hidden_dim)

		outputs = self.fc(outputs) # (batch_size, max_sent_len + 1, vocab_size)

		outputs = outputs[:, :-1, :] # (batch_size, max_sent_len, vocab_size)

		return outputs