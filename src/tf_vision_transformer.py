import concurrent.futures
import collections
import dataclasses
import hashlib
import copy
import itertools
import json
import pickle
import math
import random
import os
import pathlib
import platform
import random
import re
import string
import time
import urllib.request
from typing import List, Tuple, Dict, Union, Any, Callable, Optional

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm
import optuna

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # NoPep8
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

# ------------------------------- Download data ------------------------------ #

def download_flickr8k(path: str = 'flickr8k', sample: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Downloads and processes the Flickr8k dataset.

    Parameters
    ----------
    path : str, optional
        The local directory path where the dataset will be downloaded, by default 'flickr8k'.
    sample : bool, optional
        Whether to sample the dataset, by default True.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset]
        A tuple of two tf.data.Dataset objects. The first is the training dataset and the second is the test dataset.
        Each dataset is composed of tuples (image_path, list_of_captions).
    """
    path = pathlib.Path(path)

    # If the number of files in all subdirectories of 'path' is less than 16197
    if len(list(path.rglob('*'))) < 16197:

      # Download zip files
      tf.keras.utils.get_file(
          origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
          cache_dir='.',
          cache_subdir=path,
          extract=True
        )
      tf.keras.utils.get_file(
          origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
          cache_dir='.',
          cache_subdir=path,
          extract=True
        )

    # Read in and process captions
    captions = (path/'Flickr8k.token.txt').read_text().splitlines()
    # List of lists, where the list elements are the tokens seperated by '\t'
    captions = (line.split('\t') for line in captions)
    # list of tuples, where the elements within the tuple are the processed filename and the associated caption
    captions = ((fname.split('#')[0], caption) for (fname, caption) in captions)

    # Place captions with the same file name in a list belonging to the same 'key'
    cap_dict = collections.defaultdict(list)
    for fname, cap in captions:
      cap_dict[fname].append(cap)

    # Create a list of tuples (image_path, list_of_captions)
    train_files = (path/'Flickr_8k.trainImages.txt').read_text().splitlines()
    train_captions = [(str(path/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in train_files]

    test_files = (path/'Flickr_8k.testImages.txt').read_text().splitlines()
    test_captions = [(str(path/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in test_files]
    
    # Take a sample of the training and testing data
    train_captions = random.sample(train_captions, 200)
    test_captions = random.sample(test_captions, 200)

    # Create datasets from list of (tf.tensor(image_path), tf.tensor(list_of_captions)) tuples
    train_ds = tf.data.experimental.from_list(train_captions)
    test_ds = tf.data.experimental.from_list(test_captions)

    return train_ds, test_ds

def load_image(image_path: str, image_size: Tuple[int, int, int]) -> tf.Tensor:
    """
    This function loads in a single image and resizes it in order to feat the image into the feature extractor
    pre-trianed model.

    Parameters
    ----------
    image_path: str
        The image path.
    image_size: Tuple[int, int, int]
        The input size of the image that the pretrained model expects, to which we have to resize the original image.

    Returns
    -------
    tf.Tensor
        The returned object is an eagar tensor.
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size[:-1])
    return img

def standardize(caption: Union[str, tf.Tensor]) -> tf.Tensor:
    """
    Standardizes the given caption (string or tensor containing a string) by transforming to lower case, removing punctuation, and 
    adding '[START]' and '[END]' tokens at the beginning and end of the string, respectively. The output is a tensor 
    containing the standardized string.

    Parameters
    ----------
    caption : Union[str, tf.Tensor]
        The input string or tensor containing a string to be standardized.

    Returns
    -------
    tf.Tensor
        A tensor containing the standardized string.
    """
    caption = tf.strings.lower(caption)
    caption = tf.strings.regex_replace(caption, f'[{re.escape(string.punctuation)}]', '')
    caption = tf.strings.join(['[START]', caption, '[END]'], separator=' ')
    return caption

def match_shapes(images: Union[tf.Tensor, tf.RaggedTensor], captions: Union[tf.Tensor, tf.RaggedTensor]) -> Tuple[Union[tf.Tensor, tf.RaggedTensor], Union[tf.Tensor, tf.RaggedTensor]]:
    """
    Function to match the shapes of images and captions tensors.

    It does so by rearranging the captions tensor from shape (b, c) to (b*c) and repeating the images tensor 'c' times 
    along a new dimension. The new shape of images will be (b*c, ...).

    Parameters
    ----------
    image_paths : Union[tf.Tensor, tf.RaggedTensor]
        Tensor of images with any shape.
    captions : Union[tf.Tensor, tf.RaggedTensor]
        Tensor of captions with shape (b, c).

    Returns
    -------
    Tuple[Union[tf.Tensor, tf.RaggedTensor], Union[tf.Tensor, tf.RaggedTensor]]
        A tuple of tensors (images, captions), both reshaped to have the same first dimension.
    """
    # Get a dictionary of {'b': batch_size, 'c': num_of_captions}
    caption_shape = einops.parse_shape(captions, 'b c')
    # This reshapes captions fomr a tensor with shape (b, c) to (b*c)
    captions = einops.rearrange(captions, 'b c -> (b c)')
    # Repeat the images tensor for each caption in the batch
    images = einops.repeat(
        images, 'b ... -> (b c) ...',
        c = caption_shape['c']
    )
    return images, captions

def prepare_txt(images: Union[tf.Tensor, tf.RaggedTensor], 
                texts: Union[tf.Tensor, tf.RaggedTensor, list[str]]) -> Tuple[Tuple[tf.Tensor, tf.RaggedTensor], tf.RaggedTensor]:
    """
    This function tokenizes the text data and prepares it for use with Keras. The training dataset should contain 
    (inputs, labels) pairs. For caption generation, the tokens are both inputs and labels, shifted by one step. In 
    other words, for each step, given the previously generated tokens, we wish to generate the next token. This 
    function will convert an (tf.tensor(images), tf.tensor(texts)) pair to an ((images, input_tokens), label_tokens) 
    pair.
    
    The function first tokenizes the input texts, then creates two new tensors:
    
    - input_tokens, which includes all but the last token in each sequence
    - label_tokens, which includes all but the first token in each sequence
    
    This is a common preparation step for sequence-to-sequence models, where the model is trained to predict each token 
    in the sequence given all the previous tokens.

    Parameters
    ----------
    images : Union[tf.Tensor, tf.RaggedTensor]
        Tensor of images with any shape.
    texts : Union[tf.Tensor, tf.RaggedTensor, list[str]]
        Tensor or list of texts to be tokenized.

    Returns
    -------
    Tuple[Tuple[tf.Tensor, tf.RaggedTensor], tf.RaggedTensor]
        A tuple where the first element is another tuple containing the images and the input tokens tensor, 
        and the second element is the label tokens tensor.
    """
    tokens = tokenizer(texts)
    # Subset for all but the last token
    input_tokens = tokens[..., :-1]
    # Subset for all but the first token 
    label_tokens = tokens[..., 1:]
    return (images, input_tokens), label_tokens

def prepare_dataset(ds: tf.data.Dataset, 
                    tokenizer: Callable, 
                    image_size: Tuple[int, int],
                    batch_size: Optional[int] = 32, 
                    shuffle_buffer: Optional[int] = 1000) -> tf.data.Dataset:
    """
    This function prepares a dataset for use in model training. The function applies several transformations to 
    the dataset one at a time:

    - Shuffles the dataset.
    - Loads the images, then resizes and batches them.
    - Matches the shapes of the images and captions so each pair of examples is 1:1.
    - Unbatches the dataset, shuffles the dataset again, and then batches the dataset again.
    - Prepares the text by tokenizing and splitting the texts into input and label tokens.
    - Converts the input and label tokens to tensors.

    Parameters
    ----------
    ds : tf.data.Dataset
        The input dataset to prepare.
    tokenizer : Callable
        The tokenizer function to use to tokenize the text data (must be adapted to the training data).
    image_size: Tuple[int, int]
        Image size for resizing.
    batch_size : Optional[int], default=32
        The size of the batches to create.
    shuffle_buffer : Optional[int], default=1000
        The size of the buffer to use when shuffling the dataset.

    Returns
    -------
    tf.data.Dataset
        The prepared dataset with transformations applied.
    """
    # Load the images, resize, make batches
    ds = (ds.shuffle(10000)
             .map(lambda image_path, caption: (load_image(image_path, image_size), caption))
             .ignore_errors()
             .batch(batch_size))

    # Convert tf.RaggedTensor to tf.Tensor
    def to_tensor(inputs, labels):
        (images, in_tok), out_tok = inputs, labels
        return (images, in_tok.to_tensor()), out_tok.to_tensor()

    return (ds.map(match_shapes, tf.data.AUTOTUNE)
              .unbatch()
              .shuffle(shuffle_buffer)
              .batch(batch_size)
              .map(prepare_txt, tf.data.AUTOTUNE)
              .map(to_tensor, tf.data.AUTOTUNE))
    
 # ------------------------------- BaseAttention ------------------------------ #

@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='BaseAttention')
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        # Residual connection
        self.add = tf.keras.layers.Add()
        # Normalization to maintain scale for the outputs
        self.layernorm = tf.keras.layers.LayerNormalization()
        
    def get_config(self):
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
# ------------------------------- SeqEmbedding ------------------------------- #
        
@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='SeqEmbedding')
class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.depth = depth
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=self.max_length, output_dim=self.depth)
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.depth,
            mask_zero=True
        )
        self.add = tf.keras.layers.Add()

    def call(self, seq):
        # Output size (batch, seq, depth)
        seq = self.token_embedding(seq)
        # Sequence of integers ranging from 0 to (seq - 1) with length (seq_length) representing positions
        x = tf.range(tf.shape(seq)[1])  
        # Add dimension to match what `pos_embedding` expects, which is a (batch_size, sequence_length) 2D input
        x = x[tf.newaxis, :] 
        # Output size (1, seq, depth) where depth is the output dimension specified above
        x = self.pos_embedding(x)  

        # Token embeddings plus positional encoding
        return self.add([seq, x])
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'depth': self.depth,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# ---------------------------- CausalSelfAttention --------------------------- #
    
@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='CausalSelfAttention')
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        self.x = x
        attn = self.mha(
            query=self.x, 
            # Since 'key' is not given, 'key' will default to 'value'
            value=self.x,
            # Prevent tokens from attending to future tokens
            use_causal_mask=True
        )
        x = self.add([x, attn])
        return self.layernorm(x)
        
# ------------------------------ CrossAttention ------------------------------ #
    
@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='CrossAttention')
class CrossAttention(BaseAttention):
    def call(self, x, y, **kwargs):
        self.x = x
        self.y = y
        attn, attention_scores = self.mha(
            # X is the text features
            query=self.x, 
            # Y is the image features, and 'key' will default to 'value'
            value=self.y,
            return_attention_scores=True
        )
        # Cache the attention scores for plotting
        self.last_attention_scores = attention_scores
        x = self.add([x, attn])
        return self.layernorm(x)
        
# -------------------------------- FeedForward ------------------------------- #
    
@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='FeedForward')
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.units = units
        self.dropout_rate = dropout_rate
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=2 * self.units, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(units=self.units, activation='linear', kernel_initializer='he_normal'),
            tf.keras.layers.Dropout(rate=self.dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        return self.layernorm(x)
    
    def get_config(self):
        return {
            'units': self.units,
            'dropout_rate': self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
# ------------------------------- Decoder Layer ------------------------------ #

@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='DecoderLayer')
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.self_attention = CausalSelfAttention(
            num_heads=self.num_heads, 
            key_dim=self.units,
            dropout=self.dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=self.num_heads,
            key_dim=self.units,
            dropout=self.dropout_rate
        )
        self.feed_forward = FeedForward(units=self.units, dropout_rate=self.dropout_rate)


    def call(self, inputs, training=False):
        # The in_seq are the image features and out_seq are the text features
        in_seq, out_seq = inputs

        out_seq = self.self_attention(out_seq)

        out_seq = self.cross_attention(out_seq, in_seq)

        self.last_attention_scores = self.cross_attention.last_attention_scores

        out_seq = self.feed_forward(out_seq)

        return out_seq
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='TokenOutput')
class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.dense = tf.keras.layers.Dense(units=self.tokenizer.vocabulary_size(), **kwargs)
        self.bias = None

    def adapt(self, ds):
        # This is a subclass of dictionary that allows for mapping keys to counts
        counts = collections.Counter()
        vocab_dict = {name: id for id, name in enumerate(self.tokenizer.get_vocabulary())}

        # Count the occurence of each token in the dataset
        for tokens in tqdm.tqdm(ds):
            # Each token should be string after 'numpy().flatten()'
            counts.update(tokens.numpy().flatten())

        # The element counts_arr[i] is the count of the token with index 'i'
        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

        # This creates a 'shallow' copy to avoid warnings
        counts_arr = counts_arr[:] 
        for token in self.banned_tokens:
            # Set counts of banned tokens to zero
            counts_arr[vocab_dict[token]] = 0

        # Compute the relative frequency of each token
        total = counts_arr.sum()
        p = counts_arr / total
        # Banned token (or other tokens) with 0 counts get probabilities of 1
        p[counts_arr == 0] = 1.0
        # Log(1) == 0 for banned tokens
        log_p = np.log(p) 

        entropy = -(log_p*p).sum()
        print()
        print(f'Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}')
        print(f'Marginal entropy: {entropy:0.2f}')

        # Smart initialization of bias terms using log of probability
        self.bias = log_p
        # Initialize bias terms for banned tokens with large negative numbers
        self.bias[counts_arr == 0] = -1e9

    def call(self, x):
        x = self.dense(x)
        # Use '+' since Add layers do not work with 'x' and 'self.bias' having different shapes
        return x + self.bias
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'tokenizer': self.tokenizer,
            'banned_tokens': self.banned_tokens
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
# ----------------------------------- Model ---------------------------------- #

@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='CaptionModel')
class CaptionModel(tf.keras.Model):
  """
  A custom Keras Model for image captioning.
  """
  def __init__(self, 
               tokenizer: tf.keras.layers.TextVectorization, 
               feature_extractor: tf.keras.Model, 
               output_layer: tf.keras.layers.Layer, 
               units: int,
               num_layers: int = 1,
               max_length: int = 50, 
               num_heads: int = 1, 
               dropout_rate: float = 0.1):
    """
    Constructor for the model.

    Parameters
    ----------
    tokenizer : tf.keras.layers.TextVectorization
        A tokenizer object used to convert between text and numeric tokens.
    feature_extractor : tf.keras.Model
        A feature extractor model (e.g., a ConvNet) to extract features from images.
    output_layer : tf.keras.layers.Layer
        An output layer that predicts the next token.
    units : int
        The dimensionality of the output space.
    num_layers : int, optional
        Number of decoder layers in the model, default is 1.
    max_length : int, optional
        The maximum length of the sequence, default is 50.
    num_heads : int, optional
        The number of attention heads for the multi-head attention mechanisms, default is 1.
    dropout_rate : float, optional
        The dropout rate, default is 0.1.
    """
    super().__init__()
    # Tokenizer for text data
    self.tokenizer = tokenizer
    # Feature extractor for image data
    self.feature_extractor = feature_extractor
    # The output layer predicts a point-wise prediction of the next token
    self.output_layer = output_layer
    self.units = units
    self.num_layers = num_layers
    self.max_length = max_length
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    
    self.word_to_index = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary())
    self.index_to_word = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary(), invert=True) 
    # Token embedding and positional encoding
    self.seq_embedding = SeqEmbedding(
        vocab_size=tokenizer.vocabulary_size(),
        depth=self.units,
        max_length=self.max_length
    )
    # Stack of decoder layers
    self.decoder_layers = [
        DecoderLayer(units=self.units, num_heads=self.num_heads, dropout_rate=self.dropout_rate) for n in range(self.num_layers)
    ]

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """
    Forward pass for the model.

    Parameters
    ----------
    inputs : Tuple[tf.Tensor, tf.Tensor]
        A tuple containing an image tensor and a text tensor.

    Returns
    -------
    tf.Tensor
        The output tensor after passing through the model.
    """
    # Each inputs is (image, txt) pair
    image, txt = inputs

    # If RBG, apply the feature-extractor
    if image.shape[-1] == 3:
      image = self.feature_extractor(image)

    # Flatten the feature maps' spatial dimensions
    image = einops.rearrange(image, 'b h w c -> b (h w) c')

    # Apply the tokenizerfor string inputs
    if txt.dtype == tf.string:
      txt = tokenizer(txt)

    # Token embedding and positional encoding
    txt = self.seq_embedding(txt)

    # Apply self-attention to the tokens and cross-attention between the tokens and the images
    for dec_layer in self.decoder_layers:
      txt = dec_layer(inputs=(image, txt))

    # Predict next token
    txt = self.output_layer(txt)

    return txt
    
  def generate_caption(self, image: tf.Tensor, temperature: float = 1.0) -> str:
      """
      Generate a caption for an image.

      Parameters
      ----------
      image : tf.Tensor
        A tensor representing the image to be captioned, which should be resized to match what the feature extractor expects.
      temperature : float, optional
          A parameter controlling the randomness of the token predictions. Higher values produce more random outputs, default is 1.0.

      Returns
      -------
      str
          The generated caption as a string.
      """
      # Add batch dimension and extract features
      img_features = self.feature_extractor(image[tf.newaxis, ...])

      # Initial tokens with shape (batch, sequence)
      initial = self.word_to_index([['[START]']]) 
      tokens = initial 
      for n in range(self.max_length):

        # Shape (batch, sequence, vocab)
        preds = self((img_features, tokens)).numpy()  
        # Shape (batch, vocab)
        preds = preds[:, -1, :] 

        # The prediction 'next' has shape (batch, 1)
        if temperature == 0.0:
          next = tf.argmax(preds, axis=-1)[:, tf.newaxis] 
        else:
          next = tf.random.categorical(preds / temperature, num_samples=1) 
        # Shape (batch, sequence) 
        tokens = tf.concat([tokens, next], axis=1) 

        # Break out of the generation loop once the 'END' token is predicted
        if next[0] == self.word_to_index('[END]'):
          break

      # Convert to string
      words = self.index_to_word(tokens[0, 1:-1])  # Fixed line
      result = tf.strings.reduce_join(words, axis=-1, separator=' ')

      return result.numpy().decode()

  def plot_attention_maps(self, image: tf.Tensor, str_tokens: List[str], attention_map: tf.Tensor, figsize: Tuple[int, int] = (16, 9)) -> None:
      """
      Plot the attention maps over the image.

      Parameters
      ----------
      image : tf.Tensor
          A tensor representing the image.
      str_tokens : List[str]
          The list of string tokens.
      attention_map : tf.Tensor
          The attention map tensor.
      figsize : Tuple[int, int], optional
          The size of the figure, by default (16, 9).
      """
      fig = plt.figure(figsize=figsize)
      len_result = len(str_tokens)

      for i in range(len_result):
        map = attention_map[i]
        grid_size = max(int(np.ceil(len_result / 2)), 2)
        ax = fig.add_subplot(3, grid_size, i + 1)
        ax.set_title(str_tokens[i])
        img = ax.imshow(image)
        ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(), clim=[0.0, np.max(map)])

      plt.tight_layout()

  def generate_caption_and_plot(self, image: tf.Tensor, temperature: float = 0.0) -> None:
      """
      Generate a caption for an image and plot the attention maps.

      Parameters
      ----------
      image : tf.Tensor
          A tensor representing the image to be captioned.
      temperature : float, optional
          A parameter controlling the randomness of the token predictions. Higher values produce more random outputs, default is 0.0.
      """
      result_txt = self.generate_caption(image, temperature)
      str_tokens = result_txt.split()
      str_tokens.append('[END]')

      attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
      attention_maps = tf.concat(attention_maps, axis=0)
      attention_maps = einops.reduce(
          attention_maps,
          'batch heads sequence (height width) -> sequence height width',
          height=7, width=7,          
          reduction='mean'
      )

      self.plot_attention_maps(image / 255, str_tokens, attention_maps)
      t = plt.suptitle(result_txt)
      t.set_y(1.05)
      
  def get_config(self):
      return {
          'tokenizer': self.tokenizer,
          'feature_extractor': self.feature_extractor,
          'output_layer': self.output_layer,
          'units': self.units,
          'num_layers': self.num_layers,
          'max_length': self.max_length,
          'num_heads': self.num_heads,
          'dropout_rate': self.dropout_rate
        }

  @classmethod
  def from_config(cls, config):
      return cls(**config)
 
# ------------------------------ Loss and Metric ----------------------------- #

@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='masked_loss')
def masked_loss(labels: tf.Tensor, preds: tf.Tensor) -> tf.Tensor:
    """
    Compute the masked loss.

    Parameters
    ----------
    labels : tf.Tensor
        The true labels.
    preds : tf.Tensor
        The predicted labels.

    Returns
    -------
    tf.Tensor
        The computed masked loss.
    """
    # Measures the probability error in discrete classification tasks in which the classes are mutually exclusive
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

    # Discard the large losses for 'banned_tokens'
    mask = (labels != 0) & (loss < 1e8) 
    mask = tf.cast(mask, loss.dtype)

    loss = loss * mask
    # Average loss for elements where the mask is 1
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

@tf.keras.saving.register_keras_serializable(package='VisionTransformer', name='masked_acc')
def masked_acc(labels: tf.Tensor, preds: tf.Tensor) -> tf.Tensor:
    """
    Compute the masked accuracy.

    Parameters
    ----------
    labels : tf.Tensor
        The true labels.
    preds : tf.Tensor
        The predicted labels.

    Returns
    -------
    tf.Tensor
        The computed masked accuracy.
    """
    mask = tf.cast(labels != 0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    matched = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(matched * mask) / tf.reduce_sum(mask)
    return acc

# ------------------------------- Trainer class ------------------------------ #

class Trainer(object):
  """
  This class encapsulates the training of the vision transformer model.
  """
  def __init__(self, 
               train_ds: tf.data.Dataset, 
               test_ds: tf.data.Dataset,
               feature_extractor: tf.keras.Model, 
               tokenizer: tf.keras.layers.TextVectorization,
               output_layer: TokenOutput,
               hyperparameters: Dict[str, Any],
               steps_per_epoch: int,
               validation_steps: int,
               patience: int = 5,
               min_delta: float = 1e-3,
               restore_best_model: bool = True):
    """
    Constructor for the Trainer class.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        The training dataset.
    test_ds : tf.data.Dataset
        The testing dataset.
    feature_extractor : tf.keras.Model
        A feature extractor model (e.g., a ConvNet) to extract features from images.
    tokenizer : tf.keras.layers.TextVectorization
        A tokenizer object used to convert between text and numeric tokens (must be adapted to the training set).
    output_layer: TokenOutput
        A token output layer that is adapted to the training set.
    hyperparameters : Dict[str, Any]
        A dictionary containing the hyperparameters for model training.
    steps_per_epoch: int
        Total number of steps (batches of samples) before declaring one epoch finished, typically set to ceil(num_of_samples / batch_size).
    validation_steps: int
        Total number of steps (batches of samples) to draw before stopping when performing validation, again, usually set to ceil(num_of_samples / batch_size).
    patience : int, optional
        The number of epochs to wait before stopping training if the validation loss does not improve, 
        by default 5.
    min_delta : float, optional
        The minimum change in validation loss to qualify as an improvement, by default 1e-3.
    restore_best_model : bool, optional
        Whether to restore the model with the lowest validation loss found during training after 
        training has ended, by default True.
    """
    self.train_ds = train_ds
    self.test_ds = test_ds
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.output_layer = output_layer
    self.hyperparameters = hyperparameters
    self.steps_per_epoch = steps_per_epoch
    self.validation_steps = validation_steps
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_model = restore_best_model

  def fit(self) -> Tuple[CaptionModel, tf.keras.callbacks.EarlyStopping]:
    """
    Model training.

    Returns
    -------
    Tuple[CaptionModel, tf.keras.callbacks.EarlyStopping]
        The trained CaptionModel and the EarlyStopping callback used during training. The callback can be used to obtain information about the training process, such 
        as the epoch in which training was stopped.
    """
    # Instantiate the model
    model = CaptionModel(
        tokenizer=self.tokenizer,
        feature_extractor=self.feature_extractor,
        output_layer=self.output_layer,
        num_layers=self.hyperparameters['num_layers'],
        units=self.hyperparameters['units'],
        max_length=50,
        num_heads=self.hyperparameters['num_heads'],
        dropout_rate=0.1
    )
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.hyperparameters['learning_rate'],
        beta_1=self.hyperparameters['beta_1'],
        beta_2=self.hyperparameters['beta_2'],
        epsilon=self.hyperparameters['epsilon'],
        clipnorm=self.hyperparameters['clipnorm']
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_acc]
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_masked_acc',
        patience=self.patience,
        min_delta=self.min_delta,
        restore_best_weights=self.restore_best_model
    )
    
    model.fit(
        x=self.train_ds.repeat(),
        steps_per_epoch=self.steps_per_epoch,
        validation_data=self.test_ds.repeat(),
        validation_steps=self.validation_steps,
        epochs=self.hyperparameters['epochs'],
        callbacks=[early_stopping]
    )
    
    return model, early_stopping
    
if __name__ == '__main__':
    
    train_raw, test_raw = download_flickr8k()
    
    # -------------------------- Image feature extractor ------------------------- #
    
    image_size = (224, 224, 3)

    mobilenetv3_large = tf.keras.applications.MobileNetV3Large(
        input_shape=image_size,
        include_top=False,
        weights='imagenet',
        pooling=None,
        include_preprocessing=True
    )

    mobilenetv3_large.trainable = False
    
    # --------------------------------- Tokenizer -------------------------------- #
    
    # Use the top 5000 words for the vocabulary
    vocabulary_size = 5000

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        # Return RaggedTensor, which allows for some dimensions to have slices with different lengths
        ragged=True
    )
    
    # Batch with size 1024
    tokenizer.adapt(train_raw.map(lambda image_path, captions: captions).unbatch().batch(1024))
    
    # ------------------------------ Prepare dataset ----------------------------- #
    
    batch_size = 32

    train_ds = prepare_dataset(train_raw, tokenizer, image_size, batch_size)
    test_ds = prepare_dataset(test_raw, tokenizer, image_size, batch_size)

    # -------------------------- Adapt output bias terms ------------------------- #
    
    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
    output_layer.adapt(train_ds.map(lambda inputs, labels: labels))
    
    # ---------------------------------- Trainer --------------------------------- #
    
    trainer = Trainer(
        train_ds=train_ds,
        test_ds=test_ds,
        feature_extractor=mobilenetv3_large,
        tokenizer=tokenizer,
        output_layer=output_layer,
        hyperparameters={
            'num_layers': 1,
            'units': 128,
            'num_heads': 1,
            'learning_rate': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'clipnorm': 1.0,
            'epochs': 1
        },
        steps_per_epoch=np.ceil(len(train_raw) / batch_size),
        validation_steps=np.ceil(len(test_raw) / batch_size),
        patience=5,
        min_delta=1e-3,
        restore_best_model=True
    )
    
    trainer.fit()