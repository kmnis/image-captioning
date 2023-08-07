import os
import logging
from typing import Dict, List, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_data_loader import data_loader, FlickrDataset

import optuna
from optuna.trial import TrialState

from torch_encoder_decoder import Trainer

# ------------------------- Optuna objective function ------------------------ #

def objective(trial: optuna.Trial,
              train: Dict[str, Union[DataLoader, FlickrDataset]],
              val: Dict[str, Union[DataLoader, FlickrDataset]],
              logger: logging.Logger,
              patience: int = 3,
              min_delta: float = 1e-3,
              restore_best_model: bool = True):
    """
    Surrogate objective function for optuna.

    Parameters
    ----------
    trial : optuna.Trial
        An optuna trial object.
    train : Dict[str, Union[DataLoader, FlickrDataset]]
        A dictionary containing the training data and the corresponding data loader.
    val : Dict[str, Union[DataLoader, FlickrDataset]]
        A dictionary containing the validation data and the corresponding data loader.
    logger : logging.Logger
        A logger object.
    patience : int, optional
        The number of epochs to wait before stopping training if the validation loss does not improve,
        by default 5.
    min_delta : float, optional
        The minimum change in validation loss to qualify as an improvement, by default 1e-3.
    restore_best_model : bool, optional
        Whether to restore the model with the lowest validation loss found during training after
        training has ended, by default True.

    Returns
    -------
    float
        The validation loss.
    """
    search_space = {
        'fine_tune': trial.suggest_categorical('fine_tune', [True]),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64]),
        'num_layers': trial.suggest_categorical('num_layers', [2, 3]),
        'embed_size': trial.suggest_categorical('embed_size', [32, 64]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'l2_reg': trial.suggest_float('l2_reg', 1e-4, 1e-1, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'beta_1': trial.suggest_float('beta_1', 0.6, 0.9),
        'beta_2': trial.suggest_float('beta_2', 0.8, 0.999),
        'epochs': trial.suggest_int('epochs', 10, 15)
    }

    # Instantiate the trainer
    trainer = Trainer(
        hyperparameters=search_space,
        train=train,
        val=val,
        logger=logger,
        patience=patience,
        min_delta=min_delta,
        restore_best_model=restore_best_model
    )

    model, best_val_loss = trainer.train_model()

    return best_val_loss

# ------------------------ Function for creating study ----------------------- #

def create_study(study_name: str, storage: str, direction: str = 'minimize') -> optuna.study.Study:
    """
    Create Optuna study instance.

    Parameters
    ----------
    study_name : str
        Name of the study.
    storage : str
        Database url.
    direction: str
        Direction of the metric--- maximize or minimize.

    Returns
    -------
    optuna.study.Study
        Optuna study instance.
    """
    study = optuna.create_study(
        storage=storage,
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name,
        direction=direction,
        load_if_exists=True
    )

    return study

if __name__ == '__main__':
    
    # ---------------------------------- Set up ---------------------------------- #
    
    from custom_utils import get_logger, parser, add_additional_args
    
    logger = get_logger('torch_hpo')
    
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
    
    # ---------------------------------- Optuna ---------------------------------- #
    
    study = create_study(
        study_name='test-torch-hpo',
        storage=None,
        direction='minimize'
    )
     
    def objective_wrapper(trial: optuna.Trial) -> Callable:
        return objective(
            trial=trial,
            train={'loader': train_loader, 'dataset': train_dataset},
            val={'loader': val_loader, 'dataset': val_dataset},
            logger=logger,
            patience=5,
            min_delta=1e-3,
            restore_best_model=True
        )
        
    best_val = objective_wrapper(
        trial=optuna.trial.FixedTrial({
            'fine_tune': True,
            'hidden_size': 64,
            'num_layers': 2,
            'embed_size': 128,
            'dropout': 0.5,
            'l2_reg': 0.0001,
            'learning_rate': 0.0001,
            'beta_1': 0.6,
            'beta_2': 0.8,
            'epochs': 10
        })
    )
    
    logger.info(f'Best validation loss: {best_val}')