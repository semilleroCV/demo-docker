"""
CIFAR-10 Training Script with PyTorch Best Practices
Based on llm.txt recommendations for optimal DNN training
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model import load_model, Net, get_device

# Paths
CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')
LOG_DIR = './runs'

# Training hyperparameters
BATCH_SIZE = 128  # Larger batch size for better GPU utilization
NUM_EPOCHS = 20
INITIAL_LR = 3e-4  # Good starting point for AdamW
WEIGHT_DECAY = 0.01  # L2 regularization
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold
LABEL_SMOOTHING = 0.1  # Label smoothing for better generalization
DROPOUT_RATE = 0.3

# Early stopping
PATIENCE = 15
VAL_FREQUENCY = 1  # Validate every epoch

# Reproducibility
SEED = 42


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: Setting deterministic mode trades speed for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, is_best=False):
    """Save comprehensive checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_accuracy': best_acc,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    
    # Always save last checkpoint
    torch.save(checkpoint, LAST_MODEL_PATH)
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f'✓ Best model saved with accuracy: {best_acc:.2f}%')


def train_epoch(model, trainloader, criterion, optimizer, scheduler, scaler, device, epoch, writer):
    """Train for one epoch with best practices"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Step the scheduler (OneCycleLR requires per-batch stepping)
        scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log batch metrics
        if batch_idx % 100 == 99:
            avg_loss = running_loss / 100
            accuracy = 100. * correct / total
            global_step = epoch * len(trainloader) + batch_idx
            
            writer.add_scalar('Train/BatchLoss', avg_loss, global_step)
            writer.add_scalar('Train/BatchAccuracy', accuracy, global_step)
            writer.add_scalar('Train/GradientNorm', grad_norm.item(), global_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
            
            print(f'Epoch: {epoch+1} [{batch_idx}/{len(trainloader)}] '
                  f'Loss: {avg_loss:.3f} | Acc: {accuracy:.2f}% | '
                  f'GradNorm: {grad_norm:.3f}')
            
            running_loss = 0.0
    
    train_acc = 100. * correct / total
    return train_acc


def validate(model, testloader, criterion, device, epoch, writer):
    """Validate model with proper evaluation mode"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    
    # Log validation metrics
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', accuracy, epoch)
    
    print(f'Validation - Loss: {avg_loss:.3f} | Accuracy: {accuracy:.2f}%')
    
    return accuracy, avg_loss


def log_model_parameters(model, writer, epoch):
    """Log model weights and gradients for monitoring"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        writer.add_histogram(f'Weights/{name}', param.data, epoch)


def main():
    """Main training function with all best practices applied"""
    print("="*60)
    print("CIFAR-10 Training with PyTorch Best Practices")
    print("="*60)
    
    # Set reproducibility
    set_seed(SEED)
    
    # Device configuration
    device = get_device()
    print(f"Using device: {device}")
    
    # Enable TF32 for faster training on Ampere GPUs
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Data loading
    print(f"\nLoading data with batch_size={BATCH_SIZE}...")
    trainloader, testloader, classes = load_model(batch_size=BATCH_SIZE)
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    # Model initialization
    print("\nInitializing model with proper weight initialization...")
    model = Net(dropout_rate=DROPOUT_RATE).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler: OneCycleLR for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=INITIAL_LR * 10,  # Peak LR is 10x base LR
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,  # Warmup for first 30% of training
        anneal_strategy='cos',
        div_factor=25.0,  # initial_lr = max_lr / 25
        final_div_factor=1e4  # min_lr = initial_lr / 1e4
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Training phase
        train_acc = train_epoch(
            model, trainloader, criterion, optimizer, scheduler,
            scaler, device, epoch, writer
        )
        
        # Validation phase
        if (epoch + 1) % VAL_FREQUENCY == 0:
            val_acc, val_loss = validate(
                model, testloader, criterion, device, epoch, writer
            )
            
            # Log model parameters periodically
            if (epoch + 1) % 10 == 0:
                log_model_parameters(model, writer, epoch)
            
            # Save checkpoint if best model
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            save_checkpoint(
                epoch, model, optimizer, scheduler, scaler, 
                best_acc, is_best
            )
            
            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_acc:.2f}%")
                print(f"{'='*60}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"{'='*60}\n")
    
    # Close TensorBoard writer
    writer.close()
    
    # Save final model for compatibility
    torch.save(model.state_dict(), f'./{CHECKPOINT_DIR}/cifar_net.pth')
    print("Final model also saved to: ./cifar_net.pth")

    
if __name__ == "__main__":
    main()
