import os
import torch
import argparse
from models import EncoderCNN, SentenceLSTM, WordLSTM
from dataloader import get_loader
from torchvision import transforms
from torch import nn
import numpy as np

def script(args):
	transform = transforms.Compose([ 
            transforms.Resize(args.img_size),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor()])

	train_loader, vocab = 	get_loader(args.root_dir,
							   args.train_tsv_path,
							   args.image_path,
							   transform,
							   args.batch_size,
							   args.shuffle,
							   args.num_workers)

	vocab_size = len(vocab)
	print("vocab_size: ", vocab_size)

	val_loader, _ = get_loader(args.root_dir,
						    args.val_tsv_path,
						    args.image_path,
						    transform,
						    args.batch_size,
						    args.shuffle,
						    args.num_workers)

	encoderCNN = EncoderCNN().to(args.device)

	sentLSTM = SentenceLSTM(encoderCNN.enc_dim, args.sent_hidden_dim, args.att_dim, args.sent_input_dim, args.word_input_dim, args.int_stop_dim).to(args.device)

	wordLSTM = WordLSTM(args.word_input_dim, args.word_hidden_dim, vocab_size, args.num_layers).to(args.device)

	criterion_stop = nn.CrossEntropyLoss().to(args.device)
	criterion_words = nn.CrossEntropyLoss().to(args.device)

	params_cnn = list(encoderCNN.parameters())
	params_lstm = list(sentLSTM.parameters()) + list(wordLSTM.parameters())
        
	optim_cnn = torch.optim.Adam(params = params_cnn, lr = args.learning_rate_cnn)
	optim_lstm = torch.optim.Adam(params = params_lstm, lr = args.learning_rate_lstm)


	total_step = len(train_loader)

	for epoch in range(args.num_epochs):
		for i, (images, captions, prob) in enumerate(train_loader):
			optim_cnn.zero_grad()
			optim_lstm.zero_grad()

			batch_size = images.shape[0]
			images = images.to(args.device)
			captions = captions.to(args.device)
			prob = prob.to(args.device)

			vis_enc_output = encoderCNN(images)

			topics, ps = sentLSTM(vis_enc_output, captions, args.device)

			loss_sent = criterion_stop(ps.view(-1, 2), prob.view(-1))

			loss_word = torch.tensor([0.0]).to(args.device)

			for j in range(captions.shape[1]):
				word_outputs = wordLSTM(topics[:, j, :], captions[:, j, :])

				loss_word += criterion_words(word_outputs.contiguous().view(-1, vocab_size), captions[:, j, :].contiguous().view(-1))

			loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word


			loss.backward()
			optim_cnn.step()
			optim_lstm.step()

			# Print log info
			if i % args.log_step == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				      .format(epoch, args.num_epochs, i, total_step, loss.item())) 
			    
			# Save the model checkpoints
			# if (i+1) % args.save_step == 0:
			    # torch.save(decoder.state_dict(), os.path.join(
			    #     args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
			    # torch.save(encoder.state_dict(), os.path.join(
			    #     args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--root_dir', type=str, default='../SiVL19/iu_xray', help='path for root dir')
	parser.add_argument('--train_tsv_path', type = str, default = 'train_images.tsv', help = 'path of the training tsv file')
	parser.add_argument('--val_tsv_path', type = str, default = 'test_images.tsv', help = 'path of the validation tsv file')
	parser.add_argument('--image_path', type = str, default = 'iu_xray_images', help = 'path of the images file')
	parser.add_argument('--img_size', type = int, default = 224, help = 'size to which image is to be resized')
	parser.add_argument('--crop_size', type = int, default = 224, help = 'size to which the image is to be cropped')
	parser.add_argument('--device_number', type = str, default = "0", help = 'which GPU to run experiment on')


	parser.add_argument('--int_stop_dim', type = int, default = 64, help = 'intermediate state dimension of stop vector network')
	parser.add_argument('--sent_hidden_dim', type = int, default = 512, help = 'hidden state dimension of sentence LSTM')
	parser.add_argument('--sent_input_dim', type = int, default = 1024, help = 'dimension of input to sentence LSTM')
	parser.add_argument('--word_hidden_dim', type = int, default = 512, help = 'hidden state dimension of word LSTM')
	parser.add_argument('--word_input_dim', type = int, default = 512, help = 'dimension of input to word LSTM')
	parser.add_argument('--att_dim', type = int, default = 64, help = 'dimension of intermediate state in co-attention network')
	parser.add_argument('--num_layers', type = int, default = 1, help = 'number of layers in word LSTM')


	parser.add_argument('--lambda_sent', type = int, default = 1, help = 'weight for cross-entropy loss of stop vectors from sentence LSTM')	
	parser.add_argument('--lambda_word', type = int, default = 1, help = 'weight for cross-entropy loss of words predicted from word LSTM with target words')


	parser.add_argument('--batch_size', type = int, default = 32, help = 'size of the batch')
	parser.add_argument('--shuffle', type = bool, default = True, help = 'shuffle the instances in dataset')
	parser.add_argument('--num_workers', type = int, default = 0, help = 'number of workers for the dataloader')
	parser.add_argument('--num_epochs', type = int, default = 50, help = 'number of epochs to train the model')
	parser.add_argument('--learning_rate_cnn', type = int, default = 1e-5, help = 'learning rate for CNN Encoder')
	parser.add_argument('--learning_rate_lstm', type = int, default = 5e-4, help = 'learning rate for LSTM Decoder')


	parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]= args.device_number
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(args.device)

	script(args)