"""training functions for pytorch model"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    learning_rate=0.001,
    device='cpu',
    save_dir='models',
    patience=15,
    prediction_days=5
):
    """
    train a pytorch model
    
    args:
        model: pytorch model
        train_loader: dataloader for training data
        test_loader: dataloader for test data
        num_epochs: maximum number of training epochs
        learning_rate: learning rate for optimizer
        device: 'cuda' or 'cpu'
        save_dir: directory to save model checkpoints
        patience: early stopping patience (epochs)
        prediction_days: days ahead being predicted (saved with model)
    
    returns:
        trained model and dict with training history
    """
    # create save directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # move model to device
    model = model.to(device)
    
    # training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'learning_rates': []
    }
    
    # early stopping variables
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    
    print(f"training on device: {device}")
    print(f"total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # calculate average training loss
        avg_train_loss = np.mean(train_losses)
        
        # evaluation phase
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_losses.append(loss.item())
        
        avg_test_loss = np.mean(test_losses)
        
        # update learning rate scheduler
        scheduler.step(avg_test_loss)
        
        # record history
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{num_epochs} | "
                  f"train loss: {avg_train_loss:.6f} | "
                  f"test loss: {avg_test_loss:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")
        
        # check for improvement
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # save best model with prediction_days
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'prediction_days': prediction_days
            }, f"{save_dir}/best_model.pth")
        else:
            epochs_without_improvement += 1
        
        # early stopping
        if epochs_without_improvement >= patience:
            print(f"\nearly stopping triggered after {epoch+1} epochs")
            print(f"best test loss: {best_test_loss:.6f} at epoch {best_epoch}")
            break
    
    training_time = time.time() - start_time
    
    print("-" * 60)
    print(f"training completed in {training_time:.2f} seconds")
    print(f"best test loss: {best_test_loss:.6f} at epoch {best_epoch}")
    
    # load best model
    checkpoint = torch.load(f"{save_dir}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    history['best_epoch'] = best_epoch
    history['best_test_loss'] = best_test_loss
    history['training_time'] = training_time
    
    return model, history