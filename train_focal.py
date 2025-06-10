import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader import VirtualKITTI2GroupedDataset, transform
from model_focal import EnhancedMultiConditionFusionNet
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np

def train_across_conditions(model, dataloader, num_epochs, learning_rate, device, best_model_path="best_model_ACDC.pth"):
    """
    Train the model by comparing images across conditions for the same scene.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    #iou_loss_fn = IoULoss()
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch in dataloader:
            # Prepare the batch
            images = [img.to(device) for img in batch["images"]]  # List of images across conditions
            label = batch["label"].squeeze(1).long().to(device) # Corresponding label for the scene

            images = torch.stack(images)

            # Forward pass across conditions
            with autocast():
                outputs_list = []
                consistency_losses = []
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        # Cross-condition comparison
                        outputs, consistency_loss = model(images[i], x_condition=images[j])
                        outputs_list.append(outputs)
                        if consistency_loss is not None:
                            consistency_losses.append(consistency_loss)

                # Compute segmentation loss
                avg_output = torch.stack(outputs_list).mean(dim=0)
                segmentation_loss = criterion(avg_output, label) #cross entropy loss
                #iou_loss = iou_loss_fn(avg_output, label)
                #segmentation_loss = criterion(avg_output, label, class_weights) #focal loss

                # Combine segmentation and consistency losses
                total_loss = segmentation_loss
                #print(f"Segmentation loss: {segmentation_loss.item()}")
                #print(f"IoU loss: {iou_loss.item()}")
                if consistency_losses:
                    #print(f"Consistency loss: {torch.stack(consistency_loss).mean().item()}")
                    total_loss += 0.1 * torch.stack(consistency_losses).mean()

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 150
    batch_size = 4
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = VirtualKITTI2GroupedDataset(
        root_dir='./acdc',
        conditions=['Fog', 'Night', 'Rain', 'Snow'],
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = EnhancedMultiConditionFusionNet(input_channels=3, num_conditions=4, num_classes=20).cuda()

    # Train the model
    train_across_conditions(model, dataloader, num_epochs, learning_rate, device)
