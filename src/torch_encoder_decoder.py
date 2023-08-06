import os
from typing import List, Tuple, Any, Dict, Union, Optional, Callable
from copy import deepcopy
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # NoPep8
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from torch_data_loader import data_loader, FlickrDataset

# ------------------------- Encoder using InceptionV3 ------------------------ #

class EncoderCNN(nn.Module):
    """
    CNN Encoder that uses InceptionV3.
    """
    def __init__(self, embed_size: int, dropout: float = 0.3, fine_tune: bool = True):
        """
        Constructor for the EncoderCNN class.

        Parameters
        ----------
        embed_size : int
            The size of the embedding.
        dropout : float, default=0.3
            Dropout rate.
        fine_tune : bool, default=True
            Whether to fine-tune the InceptionV3 model's top layer.
        """
        super(EncoderCNN, self).__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Linear(in_features=self.inception.fc.in_features, out_features=embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fine_tune = fine_tune
        
        # Initialize the dense layer's weights with 'he_normal' initialization
        nn.init.kaiming_normal_(self.inception.fc.weight, nonlinearity='relu')
        
        # Only fine-tune the top classifier layer's parameters
        if self.fine_tune:
            for name, param in self.inception.named_parameters():
                if 'fc.weight' in name or 'fc.bias' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Freeze all the layers
            for param in self.inception.parameters():
                param.requires_grad = False


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Parameters
        ----------
        images : torch.Tensor
            Input images.

        Returns
        -------
        torch.Tensor
            Encoded images.
        """
        if self.inception.training:  # if the model is in training mode
            features, _ = self.inception(images)  # unpack the tuple
        else:
            features = self.inception(images)  # directly get the tensor
        return self.dropout(self.relu(features))

# ---------------------------- Decoder using LSTM ---------------------------- #

class DecoderRNN(nn.Module):
    """
    RNN Decoder using LSTM to generate captions.
    """
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int, dropout: float = 0.3):
        """
        Constructor for the DecoderRNN class.

        Parameters
        ----------
        embed_size : int
            The size of the embedding.
        hidden_size : int
            The size of the hidden state.
        vocab_size : int
            The size of the vocabulary.
        num_layers : int
            Number of layers.
        dropout : float, default=0.3
            Dropout rate.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder. Note that Sequence dimension is first, so concatenating 
        the encoded images with the embedding along the sequence dimension treats them as the 
        initial context or 'start token'. By providing the image features as the first step, 
        the LSTM's hidden state is initialized with information about the image. This state can 
        then influence the generation of each word in the caption.

        Parameters
        ----------
        features : torch.Tensor
            Encoded images.
        captions : torch.Tensor
            Input captions.

        Returns
        -------
        torch.Tensor
            Decoded captions.
        """
        # Get the embeddings of the caption (tokens)
        embeddings = self.dropout(self.embed(captions))
        # Use encoded image features as the context for modeling the sequence of tokens
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # Discard cell state and keep the final hidden state
        hidden_states, _ = self.lstm(embeddings)
        # Produce the output for each token in the vocabulary, indicating the likelihood of each token being the next word in the caption
        outputs = self.linear(hidden_states)
        return outputs

# --------------------------- Encoder-Decoder Model -------------------------- #

class CaptionModel(nn.Module):
    """
    Use the Encoder-Decoder architecture for generating captions.
    """
    def __init__(self, 
                 embed_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int,
                 dropout: float = 0.3, 
                 fine_tune: bool = True):
        """
        Constructor for the CaptionModel class.

        Parameters
        ----------
        embed_size : int
            The size of the embedding.
        hidden_size : int
            The size of the hidden state.
        vocab_size : int
            The size of the vocabulary.
        num_layers : int
            Number of layers.
        dropout : float, default=0.3
            Dropout rate.
        fine_tune : bool, default=True
            Whether to fine-tune the InceptionV3 model's top layer.
        """
        super(CaptionModel, self).__init__()
        self.encoder_cnn = EncoderCNN(embed_size, dropout, fine_tune)
        self.decoder_rnn = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.

        Parameters
        ----------
        images : torch.Tensor
            Input images.
        captions : torch.Tensor
            Input captions.

        Returns
        -------
        torch.Tensor
            Output captions.
        """
        features = self.encoder_cnn(images)
        outputs = self.decoder_rnn(features, captions)
        return outputs

    def caption_image(self, image: torch.Tensor, vocabulary: Any, max_length: int = 50) -> List[str]:
        """
        Generates captions for an image using the trained model.

        Parameters
        ----------
        image : torch.Tensor
            Input image.
        vocabulary : Any
            Vocabulary object with mapping between tokens and indices.
        max_length : int, default=50
            Maximum length for generated caption.

        Returns
        -------
        List[str]
            List of tokens representing the generated caption.
        """
        result_caption = []

        with torch.no_grad():
            # Encode the image and add batch dimension
            x = self.encoder_cnn(image).unsqueeze(0)
            # Initialize the states
            states = None

            for _ in range(max_length):
                hidden_states, states = self.decoder_rnn.lstm(x, states)
                output = self.decoder_rnn.linear(hidden_states.squeeze(0))
                # Return the index of the word with the highest score
                predicted = output.argmax(1)
                # Use 'item' to return a standard python number
                predicted_token = vocabulary.index_to_token[predicted.item()]
                
                # Check if the predicted token is the <UNK> token
                if predicted_token == '<UNK>':
                    # If so, continue to the next iteration
                    continue
                
                result_caption.append(predicted_token)
                
                if predicted_token == '<EOS>':
                    break
                
                # Embed the last predicted word to be the new input of the LSTM
                x = self.decoder_rnn.embed(predicted).unsqueeze(0)

        # Return generated caption (without the <SOS> and <EOS> tokens)
        return result_caption[1:-1]
    
# ------------------------------ Early stopping ------------------------------ #

class EarlyStopping(object):
    """
    This class implements early stopping for training models.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_model: bool = True):
        """
        Constructor for the EarlyStopping class.

        Parameters
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped, by default 5.
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement, by default 0.0.
        restore_best_model : bool, optional
            Whether to restore the weights of the model from the epoch with the best value, by default True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0
        self.restore_best_model = restore_best_model
        self.best_state_dict = None

    def __call__(self, model, val_loss: float, epoch: int) -> Tuple[bool, float]:
        """
        Check if the training should stop.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model from which we can get the state dict.
        val_loss : float
            The validation loss.
        epoch : int
            The current epoch number.

        Returns
        -------
        Tuple[bool, float]
            A tuple of a boolean indicating whether the training should stop and the best validation loss.
        """
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_model:
                self.best_state_dict = deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True, self.best_loss
        
        return False, self.best_loss
    
# ------------------------------- Trainer class ------------------------------ #

class Trainer(object):
    """
    Class for training the CaptionModel for image captioning.
    """
    def __init__(self, 
                 hyperparameters: Dict[str, Any], 
                 train: Dict[str, Union[DataLoader, FlickrDataset]],
                 val: Dict[str, Union[DataLoader, FlickrDataset]],
                 logger: logging.Logger,
                 patience: int = 5,
                 min_delta: float = 1e-3,
                 restore_best_model: bool = True):
        """
        Constructor for the Trainer class.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            Hyperparameters for training the model. 
        train: Dict[DataLoader, FlickrDataset]
            Training data loader and custom dataset: {'loader': train_loader, 'dataset': train_dataset}
        val: Dict[DataLoader, FlickrDataset]
            Validation data loader and custom dataset: {'loader': val_loader, 'dataset': val_dataset}
        logger : logging.Logger
            Logger object for logging.
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped, by default 5.
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement, by default 1e-3.
        restore_best_model : bool, optional
            Whether to restore the weights of the model from the epoch with the best value, by default True.
        """
        # Encoder (CNN) hyperparameters
        self.fine_tune = hyperparameters['fine_tune']
        # Decoder (RNN) hyperparameters
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['num_layers']
        # Shared hyperparameters between the encoder and decoder
        self.embed_size = hyperparameters['embed_size']
        self.dropout = hyperparameters['dropout']
        self.l2_reg = hyperparameters['l2_reg']
        # Optimization hyperparameters
        self.learning_rate = hyperparameters['learning_rate']
        self.beta_1 = hyperparameters['beta_1']
        self.beta_2 = hyperparameters['beta_2']
        # Training hyperparameters
        self.epochs = hyperparameters['epochs']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data
        self.train = train
        self.val = val
        self.vocab_size = len(self.train['dataset'].vocab)
        
        # Model
        self.model = CaptionModel(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers, self.dropout, self.fine_tune).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.train['dataset'].vocab.token_to_index['<PAD>'])
        self.optimizer = optim.Adam(
            params=self.model.parameters(), 
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.l2_reg
        )
        
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, restore_best_model=restore_best_model)
        self.logger = logger
        
    def replace_oov_tokens(self, captions: torch.Tensor) -> torch.Tensor:
        """
        Replace out-of-vocabulary tokens in captions with the '<UNK>' token.
        """
        # If any token index is greater than the vocabulary size, replace it with the '<UNK>' token, or else keep the original token index
        return torch.where(captions < len(self.train['dataset'].vocab), captions, self.train['dataset'].vocab.token_to_index['<UNK>'])
        
    def evaluate_model(self) -> float:
        """
        Evaluates the model on the validation set.

        Returns
        -------
        float
            Validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, captions in self.val['loader']:
                images = images.to(self.device)
                captions = captions.to(self.device)
                # Replace OOV tokens
                captions = self.replace_oov_tokens(captions)
                outputs = self.model(images, captions[:-1])
                # Reshape outputs from (seq, batch, vocab_size) to (seq * batch, vocab_size) and captions from (seq, batch) to (seq * batch)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                total_loss += loss.item()
        # Average validation loss over all batches
        return total_loss / len(self.val['loader'])
        

    def train_one_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Loss value after the epoch.
        """
        self.model.train()
        for idx, (images, captions) in tqdm(enumerate(self.train['loader']), total=len(self.train['loader']), leave=False):
            images = images.to(self.device)
            captions = captions.to(self.device)
            outputs = self.model(images, captions[:-1])
            loss = self.criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            self.optimizer.zero_grad()
            # Back-propagation
            loss.backward()
            # Single update step
            self.optimizer.step()
        return loss.item()

    def train_model(self) -> Tuple[CaptionModel, float]:
        """
        Train the model for multiple epochs with early stopping.

        Returns
        -------
        Tuple[CaptionModel, float]
            Trained model and the best validation loss.
        """
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate_model()
            
            self.logger.info(f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            # Check early stopping criteria
            stop_training, best_val_loss = self.early_stopping(self.model, val_loss, epoch)
            
            # Save the model if the current validation loss is the best so far
            if best_val_loss == val_loss:
                self.logger.info(f'Current Best Validation Loss at epoch {epoch + 1}: {best_val_loss:.4f}')
            if stop_training:
                self.logger.info(f'Early stopping triggered at epoch {epoch + 1}, restoring best model weights')
                if self.early_stopping.restore_best_model:
                    matched_keys = self.model.load_state_dict(self.early_stopping.best_state_dict)
                break
            
        return self.model, self.early_stopping.best_loss
    
if __name__ == '__main__':
    
    # ---------------------------------- Set up ---------------------------------- #
    
    from custom_utils import get_logger, parser, add_additional_args
    from PIL import Image
    
    logger = get_logger('torch_encoder_decoder')
    
    additional_args = {
        'train_image_dir': str,
        'val_image_dir': str,
        'train_captions_file': str,
        'val_captions_file': str,
        'batch_size': int,
        'num_workers': int
    }
    args = add_additional_args(parser_func=parser, additional_args=additional_args)() 
    
    # --------------------------------- Load data -------------------------------- #
    
    transform_pipeline = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_loader, train_dataset = data_loader(
        image_dir=args.train_image_dir,
        captions_file=args.train_captions_file,
        transform=transform_pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        test_mode=args.test_mode
    )
    
    val_loader, val_dataset = data_loader(
        image_dir=args.val_image_dir,
        captions_file=args.val_captions_file,
        transform=transform_pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        test_mode=args.test_mode
    )  
    
    # ---------------------------------- Trainer --------------------------------- #
    
    hyperparamters = {
        'fine_tune': True,
        'hidden_size': 8,
        'num_layers': 1,
        'embed_size': 8,
        'dropout': 0.5,
        'l2_reg': 0.0,
        'learning_rate': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epochs': 1
    }
    
    trainer = Trainer(
        hyperparamters,
        train={'loader': train_loader, 'dataset': train_dataset},
        val={'loader': val_loader, 'dataset': val_dataset},
        logger=logger,
        patience=5,
        min_delta=1e-3,
        restore_best_model=True
    )
    
    model, best_val_loss = trainer.train_model()
    
    logger.info(f'Best validation loss: {best_val_loss:.4f}')