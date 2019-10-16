import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, re
import nltk
from collections import Counter
from build_vocab import Vocabulary, build_vocab

def create_captions(filepath):

    ## the captions have the impression and findings concatenated to form one big caption
    ## i.e. caption = impression + " " + findings
    ## WARNING: in addition to the XXXX in the captions, there are <unk> tokens


    # clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{},0-9]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

    captions = []

    with open(filepath, "r") as file:

        for line in file:
            line = line.replace("\n", "").split("\t")
            
            sentence_tokens = []
            
            for sentence in line[1].split("."):
                tokens = bioclean(sentence)
                if len(tokens) == 0:
                    continue
                caption = " ".join(tokens)
                sentence_tokens.append(caption)
            
            captions.append(sentence_tokens)
    
    return captions

class iuxray(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        
        tsv_file = os.path.join(self.root_dir, self.tsv_path)
        
        self.captions = create_captions(tsv_file)
        self.vocab = build_vocab(self.captions, 1)
        self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_path, self.data_file.iloc[idx, 0])
        image = Image.open(img_name)
        
        if self.transform is not None:
            image = self.transform(image)
        
        caption = self.captions[idx]

        sentences = []

        for i in range(len(caption)):
            tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())
            sentence = []
            sentence.append(self.vocab('<start>'))
            sentence.extend([self.vocab(token) for token in tokens])
            sentence.append(self.vocab('<end>'))
            sentences.append(sentence)
            
        max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])
        
        for i in range(len(sentences)):
            if len(sentences[i]) < max_sent_len:
                sentences[i] = sentences[i] + (max_sent_len - len(sentences[i]))* [self.vocab('<pad>')]
                
        target = torch.Tensor(sentences)

        return image, target, len(sentences), max_sent_len


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, no_of_sent, max_sent_len).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption, no_of_sent, max_sent_len). 
            - image: torch tensor of shape (3, crop_size, crop_size).
            - caption: torch tensor of shape (no_of_sent, max_sent_len); variable length.
            - no_of_sent: number of sentences in the caption
            - max_sent_len: maximum length of a sentence in the caption

    Returns:
        images: torch tensor of shape (batch_size, 3, crop_size, crop_size).
        targets: torch tensor of shape (batch_size, max_no_of_sent, padded_max_sent_len).
        prob: torch tensor of shape (batch_size, max_no_of_sent)
    """
    # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)
    
    images, captions, len_sentences, max_sent_len = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    
    targets = torch.zeros(len(captions), max(len_sentences), max(max_sent_len)).long()
    prob = torch.zeros(len(captions), max(len_sentences)).long()
    
    for i, cap in enumerate(captions):
        for j, sent in enumerate(cap):
            targets[i, j, :len(sent)] = sent[:] 
            prob[i, j] = 1
        # stop after the last sentence
        prob[i, j] = 0
      
    return images, targets, prob

def get_loader(root_dir, tsv_path, image_path, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # dataset
    data = iuxray(root_dir = root_dir, 
             tsv_path = tsv_path, 
             image_path = image_path,
             transform = transform)
    
    # Data loader for dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, resize_length, resize_width).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset = data, 
                                              batch_size = batch_size,
                                              shuffle = shuffle,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    return data_loader, data.vocab