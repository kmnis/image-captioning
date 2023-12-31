import os
import logging
from typing import List, Dict, Tuple, Any, Union, Optional
import shutil

import numpy as np
import pandas as pd 
import spacy  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # NoPep8
import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from PIL import Image 
from sklearn.model_selection import train_test_split

# ----------------------- Download spacy english model ----------------------- #

try:
    spacy_eng = spacy.load('en_core_web_sm')
except OSError:
    os.system('source activate cv_env && python3 -m spacy download en_core_web_sm')
    spacy_eng = spacy.load('en_core_web_sm')

# ------------------------ Class for build vocabulary ------------------------ #

class Vocabulary(object):
    """
    This class implements the logic for building the 
    vocabulary from the captions. Specifically, we only
    add a word to the vocabulary if it occurs at least
    'freq_threshold' times.
    """
    def __init__(self, freq_threshold: int):
        """
        Constructor for the Vocabulary class. 
        
        - <PAD>: padding token
        - <SOS>: start of sequence token
        - <EOS>: end of sequence token
        - <UNK>: unknown token not in the vocabulary

        Parameters
        ----------
        freq_threshold : int
            The threshold for word frequency.
        """
        self.index_to_token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.token_to_index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self) -> int:
        """
        Returns the length of the vocabulary.

        Returns
        -------
        int
            The length of the vocabulary.
        """
        return len(self.index_to_token)
    
    @staticmethod
    def tokenizer_eng(text: str) -> List[str]:
        """
        Tokenize the text using spaCy from a caption string into
        a list of tokens. The tokenizer returns a 'spacy.tokens.doc.Doc', 
        which is a sequence of Token objects. The 'spacy.tokens.doc.Doc',
        supports the python iterator protocol, so we can iterate over it.

        Parameters
        ----------
        text : str
            The caption to be tokenized.

        Returns
        -------
        List[str]
            The list of tokens in a caption.
        """
        # Convert to lowercase
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, captions_list: List[str]):
        """
        Build vocabulary from a list of captions.

        Parameters
        ----------
        captions_list : List[str]
            The list of captions.
        """
        frequencies = {}
        # We start at 4 because 0, 1, 2, 3 are special tokens
        idx = 4

        for caption in captions_list:
            for word in self.tokenizer_eng(caption):
                # If the word is not in the dictionary, add it
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    # If the word is in the dictionary, increment its frequency
                    frequencies[word] += 1

                # Only add the word to the vocabulary if its frequency is greater than the threshold
                if frequencies[word] == self.freq_threshold:
                    self.token_to_index[word] = idx
                    self.index_to_token[idx] = word
                    idx += 1

    def vectorize(self, text: str) -> List[int]:
        """
        Convert words in the captions to their respective indices in the vocabulary.

        Parameters
        ----------
        text : str
            The input caption.

        Returns
        -------
        List[int]
            The indices of words in the captions.
        """
        # A 'doc' object is a sequence of tokens
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.token_to_index[token] if token in self.token_to_index else self.token_to_index['<UNK>']
            for token in tokenized_text
        ]
        
# ------------------------ Class for Flickr8k dataset ------------------------ #

class FlickrDataset(Dataset):
    """
    This class implements the custom Flickr8k dataset, which represents a map from keys to 
    data samples.
    """
    def __init__(self, image_dir: str, captions_file: str, transform: Optional[transforms.Compose] = None, freq_threshold: int = 5, test_mode: bool = False):
        """
        Constructor for the FlickrDataset class.

        Parameters
        ----------
        image_dir : str
            The directory with the images.
        captions_file : str
            The file path for the captions.
        transform : Optional[transforms.Compose], default=None
            Transforms to apply on the images.
        freq_threshold : int, default=5
            The threshold for word frequency.
        test_mode : bool, default=False
            Whether to run in test mode, which samples only 64 examples.
        """
        self.image_dir = image_dir
        # This reads in the txt file mapping image file names to captions, and takes only the first 64 rows if in test mode
        self.data = pd.read_csv(captions_file).iloc[:64] if test_mode else pd.read_csv(captions_file)
        self.transform = transform
        self.images = self.data['image']
        self.captions = self.data['caption']
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self) -> int:
        # The length of the dataset is the number of captions (not images since there are 5 captions per image)
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        """
        Get an item from the dataset. The __getitem__ function loads and returns 
        a sample from the dataset at the given index.

        Parameters
        ----------
        index : int
            The index of the item.

        Returns
        -------
        Tuple[Any, torch.Tensor]
            The image and its caption's embedding.
        """
        caption = self.captions[index]
        image_id = self.images[index]
        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')

        # Apply transformations to the image if specified
        if self.transform is not None:
            image = self.transform(image)

        # Start the 'embedded_caption' list with the start of sequence token
        embedded_caption = [self.vocab.token_to_index['<SOS>']]
        # Concatenate the vectorized tokens list to the 'embedded_caption' list
        embedded_caption += self.vocab.vectorize(caption)
        # End the 'embedded_caption' list with the end of sequence token
        embedded_caption.append(self.vocab.token_to_index['<EOS>'])

        return image, torch.tensor(embedded_caption)
    
class CustomCollate(object):
    """
    A custom `collate_fn` is designed to handle the preprocessing of batches of data before they're 
    passed to a model during training or evaluation. This function defines how multiple data points 
    (each fetched by the dataset's __getitem__ method) are combined into a batch.
    """
    def __init__(self, pad_idx: int):
        """
        Initialize the MyCollate object.

        Parameters
        ----------
        pad_idx : int
            The index for padding.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[Any, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process and collate the batch data.

        Parameters
        ----------
        batch : List[Tuple[Any, torch.Tensor]]
            The batch data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The processed images and captions.
        """
        # Add batch dimension to the image tensor for the batch
        images = [item[0].unsqueeze(0) for item in batch]
        # Concatenate the images in the batch along the batch (0th) dimension
        images = torch.cat(images, dim=0)
        # Get the captions for the batch
        captions = [item[1] for item in batch]
        # Stack a list of Tensors (caption tokens) along a new dimension and pad them to equal length
        captions = pad_sequence(sequences=captions, batch_first=False, padding_value=self.pad_idx)

        return images, captions
    
# ------------------------- Function for data loader ------------------------- #

def data_loader(
    image_dir: str,
    captions_file: str,
    transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    test_mode: bool = False
) -> Tuple[DataLoader, FlickrDataset]:
    """
    Get the data loader.

    Parameters
    ----------
    image_dir : str
        The directory with the images.
    captions_file : str
        The file path for the captions.
    transform : transforms.Compose
        Transforms to apply on the images.
    batch_size : int, default=32
        The size of the batch.
    num_workers : int, default=8
        The number of workers for multi-process data loading (avoiding GIL bottleneck).
    shuffle : bool, default=True
        Whether to shuffle the data.
    pin_memory : bool, default=True
        Whether to pin memory, which speeds up the transfer of data from CPU to GPU.
    test_mode : bool, default=False
        Whether to run in test mode, which samples only two batches worth of data.

    Returns
    -------
    Tuple[DataLoader, FlickrDataset]
        The data loader and the dataset.
    """
    dataset = FlickrDataset(image_dir=image_dir, captions_file=captions_file, transform=transform, test_mode=test_mode)
    pad_idx = dataset.vocab.token_to_index['<PAD>']
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CustomCollate(pad_idx=pad_idx)
    )

    return data_loader, dataset

# ------------------- Functions for train, val, test split ------------------- #

def split_data(data: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train, validation, and test sets based on unique images.

    Parameters
    ----------
    data : pd.DataFrame
        The original data.
    val_ratio : float
        The ratio of the validation set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The train, validation, and test sets.
    """
    unique_images = data['image'].unique()
    
    # Split unique images into 90% train_temp and 10% test
    train_temp_images, test_images = train_test_split(unique_images, test_size=0.1, random_state=42)
    
    # Split train_temp_images into 80% train and 20% val
    train_images, val_images = train_test_split(train_temp_images, test_size=val_ratio, random_state=42)
    
    # Get the corresponding rows from the data DataFrame
    train_data = data[data['image'].isin(train_images)]
    val_data = data[data['image'].isin(val_images)]
    test_data = data[data['image'].isin(test_images)]

    return train_data, val_data, test_data

def move_images(source_dir: str, target_dir: str, image_list: List[str]):
    """
    Move the images from the source directory to the target directory.

    Parameters
    ----------
    source_dir : str
        The source directory.
    target_dir : str
        The target directory.
    image_list : List[str]
        The list of image filenames to move.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for img in image_list:
        shutil.move(os.path.join(source_dir, img), target_dir)

def save_captions(data: pd.DataFrame, filename: str):
    """
    Save the captions to a file.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the captions.
    filename : str
        The filename to save the captions to.
    """
    data.to_csv(filename, index=False)

if __name__ == '__main__':
    
    # ---------------------------------- Set up ---------------------------------- #
    
    from custom_utils import get_logger, parser, add_additional_args
    
    logger = get_logger(name='torch_data_loader')
    
    additional_args = {
        'image_dir': str,
        'captions_file': str,
        'batch_size': int,
        'num_workers': int
    }
    args = add_additional_args(parser_func=parser, additional_args=additional_args)() 
    
    # -------------------------- Transformation pipeline ------------------------- #
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    loader, dataset = data_loader(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        test_mode=args.test_mode
    )
    
    # --------------------- Log information about the dataset -------------------- #
    
    logger.info(f'Number of captions: {len(dataset)}')
    logger.info(f'Number of tokens in vocabulary: {len(dataset.vocab)}')
    
    num_batches = len(loader)
    logger.info(f'Number of batches: {num_batches}')
    
    # -------------------------- Train, val, test split -------------------------- #
    
    # Split the data
    data = pd.read_csv(args.captions_file)
    train_data, val_data, test_data = split_data(data, val_ratio=0.2)
    
    # Create directories for train, validation, and test images
    train_dir = os.path.join(args.image_dir, 'train')
    val_dir = os.path.join(args.image_dir, 'val')
    test_dir = os.path.join(args.image_dir, 'test')

    # Move the images to their respective directories
    move_images(args.image_dir, train_dir, train_data['image'].unique())
    move_images(args.image_dir, val_dir, val_data['image'].unique())
    move_images(args.image_dir, test_dir, test_data['image'].unique())

    # Save the split captions
    save_captions(train_data, os.path.join(os.path.dirname(args.captions_file), 'train_captions.txt'))
    save_captions(val_data, os.path.join(os.path.dirname(args.captions_file), 'val_captions.txt'))
    save_captions(test_data, os.path.join(os.path.dirname(args.captions_file), 'test_captions.txt'))