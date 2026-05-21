from models.ensemble import *
from utils.misc import *
from configs.config import *
import json
from datetime import datetime

def train_model_old(decoder_model, train_loader, val_loader, criterion, optimizer, input_dict):

    # Extract parameters from original input dict
    num_epochs = input_dict["num_epochs"]
    lambda_div = input_dict["lambda_div"]
    enforce_diversity = input_dict["enforce_diversity"]
    resume = input_dict["resume"]

    decoder_model.train()
    optimizer.zero_grad()
    save_interval = 5

    decoder_model.to(DEVICE)
    start_epoch = 0

    # Resume from a saved checkpoint if one is available
    if resume and LAST_CHECKPOINT_PATH.exists():
        log_msg(f"--> User requested resume. Loading {LAST_CHECKPOINT_PATH}...")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)

        decoder_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        log_msg(f"--> Successfully resumed from epoch {start_epoch + 1}")

    elif not resume and LAST_CHECKPOINT_PATH.exists():
        # Safety warning so you don't accidentally overwrite previous work
        log_msg("!! WARNING: Checkpoint exists but --resume was not used. !!")
        log_msg("!! This run will OVERWRITE your existing checkpoint. !!")

    log_msg(f"Starting Training")
    for epoch in range(start_epoch, num_epochs):
        log_msg(f"Starting epoch {epoch+1}...")

        epoch_task_loss = 0
        epoch_div_loss = 0

        for features, targets in train_loader:
            optimizer.zero_grad()

            # Move data to GPU - Note: features are already [1024, 28, 28]
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()

            # Forward pass
            all_preds = decoder_model(features)  # [M, Batch, 24, 224, 224]

            # Task Loss: Process one head at a time (Memory Efficient)
            total_task_loss = 0
            for i in range(decoder_model.M):
                # Interpolate head 'i' to 224x224
                head_high_res = F.interpolate(all_preds[i], size=(224, 224), mode='bilinear', align_corners=False)
                total_task_loss += criterion(head_high_res, targets)

            # 2. Ensemble Mean Loss
            # Average at 28x28 first (cheap), then upsample once (expensive)
            mean_logits_low = all_preds.mean(dim=0)
            mean_logits_high = F.interpolate(mean_logits_low, size=(224, 224), mode='bilinear')
            total_task_loss += criterion(mean_logits_high, targets)

            # Divide by (M + 1) because we have M heads + 1 mean head
            task_loss = total_task_loss / (decoder_model.M + 1)

            # 3. Diversity Loss (JSD) - Always calculate at 28x28
            div_loss = 0
            if enforce_diversity:
                # Convert to probs for JSD
                # Adding in temperature to soften prob profiles ?
                div_loss = js_divergence_loss(all_preds)

            # 4. Total Loss Calculation
            # Have swapped the convention to the standard so we use a plus sign
            total_loss = task_loss + (lambda_div * div_loss)
            total_loss.backward()
            optimizer.step()

            # Record losses
            if torch.is_tensor(div_loss):
                epoch_div_loss += div_loss.item()
            else:
                epoch_div_loss += div_loss

            if torch.is_tensor(task_loss):
                epoch_task_loss += task_loss.item()
            else:
                epoch_task_loss += task_loss

        if (epoch + 1) % save_interval == 0:
            # Save the progress at the end of every 20th epoch
            log_msg(f"Periodic save at epoch {epoch+1}")
            current_state = {
                'epoch': epoch + 1,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            out_dir = os.getenv("OUT_DIR", "results")
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            save_checkpoint(current_state, checkpoint_dir)

        # Print epoch summary
        avg_task = epoch_task_loss / len(train_loader)
        avg_div = epoch_div_loss / len(train_loader)

        log_msg(f"Epoch [{epoch + 1}/{num_epochs}] - Task Loss: {avg_task:.8f}, Div Loss: {avg_div:.8f}")

        # --- Validation Block ---
        if val_loader is not None:
            decoder_model.eval()  # Turn off dropout/batchnorm
            val_task_loss = 0

            with torch.no_grad():
                for v_features, v_targets in val_loader:
                    v_features = v_features.to(DEVICE)
                    v_targets = v_targets.to(DEVICE).long()

                    # We evaluate the ensemble mean for validation metrics
                    all_preds = decoder_model(v_features)
                    mean_logits_low = all_preds.mean(dim=0)
                    mean_logits_high = F.interpolate(mean_logits_low, size=(224, 224), mode='bilinear')

                    v_loss = criterion(mean_logits_high, v_targets)
                    val_task_loss += v_loss.item()

            avg_val_loss = val_task_loss / len(val_loader)
            log_msg(f"Validation Loss: {avg_val_loss:.8f}")

            # Switch back to train mode for the next epoch
            decoder_model.train()

    log_msg(f"Training completed")

    return decoder_model

def train_model(decoder_model, train_loader, val_loader, criterion, optimizer, input_dict):
    num_epochs = input_dict["num_epochs"]
    lambda_div = input_dict["lambda_div"]
    enforce_diversity = input_dict["enforce_diversity"]
    resume = input_dict["resume"]

    run_name = input_dict.get("run_name", "run")
    runs_dir = os.path.join(os.getenv("OUT_DIR", "results"), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_last_checkpoint.pth"
    loss_history = {"train": [], "val": []}

    decoder_model.train()
    optimizer.zero_grad()
    save_interval = 5
    decoder_model.to(DEVICE)
    start_epoch = 0

    loss_history = {"train": [], "val": []}

    # Build checkpoint path
    if input_dict.get("checkpoint_path"):
        checkpoint_path = Path(input_dict["checkpoint_path"])
    else:
        checkpoint_path = CHECKPOINT_DIR / f"{run_name}_last_checkpoint.pth"

    if resume and checkpoint_path.exists():
        log_msg(f"--> Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        decoder_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_history = json.load(f)
        log_msg(f"--> Successfully resumed from epoch {start_epoch + 1}")

    elif not resume and checkpoint_path.exists():
        log_msg("!! WARNING: Checkpoint exists but --resume was not used. !!")
        log_msg("!! This run will OVERWRITE your existing checkpoint. !!")

    log_msg(f"Starting Training")

    for epoch in range(start_epoch, num_epochs):
        log_msg(f"Starting epoch {epoch+1}...")
        epoch_task_loss = 0
        epoch_div_loss = 0

        decoder_model.train()

        for features, targets in train_loader:
            optimizer.zero_grad()
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()

            all_preds = decoder_model(features)

            total_task_loss = 0
            for head_idx in range(decoder_model.M):
                head_high_res = F.interpolate(all_preds[head_idx], size=(224, 224),
                                              mode='bilinear', align_corners=False)
                total_task_loss += criterion(head_high_res, targets)

            mean_logits_low = all_preds.mean(dim=0)
            mean_logits_high = F.interpolate(mean_logits_low, size=(224, 224), mode='bilinear')
            total_task_loss += criterion(mean_logits_high, targets)
            task_loss = total_task_loss / (decoder_model.M + 1)

            div_loss = 0
            if enforce_diversity:
                all_probs = torch.softmax(all_preds, dim=2)
                div_loss = js_divergence_loss(all_probs)

            total_loss = task_loss - (lambda_div * div_loss)
            total_loss.backward()
            optimizer.step()

            if torch.is_tensor(div_loss):
                epoch_div_loss += div_loss.item()
            else:
                epoch_div_loss += div_loss

            if torch.is_tensor(task_loss):
                epoch_task_loss += task_loss.item()
            else:
                epoch_task_loss += task_loss

        avg_task = epoch_task_loss / len(train_loader)
        avg_div = epoch_div_loss / len(train_loader)
        loss_history["train"].append(avg_task)

        log_msg(f"Epoch [{epoch+1}/{num_epochs}] - Task Loss: {avg_task:.8f}, Div Loss: {avg_div:.8f}")

        if val_loader is not None:
            decoder_model.eval()
            val_task_loss = 0
            with torch.no_grad():
                for v_features, v_targets in val_loader:
                    v_features = v_features.to(DEVICE)
                    v_targets = v_targets.to(DEVICE).long()
                    all_preds = decoder_model(v_features)
                    mean_logits_low = all_preds.mean(dim=0)
                    mean_logits_high = F.interpolate(mean_logits_low, size=(224, 224), mode='bilinear')
                    v_loss = criterion(mean_logits_high, v_targets)
                    val_task_loss += v_loss.item()

            avg_val_loss = val_task_loss / len(val_loader)
            loss_history["val"].append(avg_val_loss)
            log_msg(f"Validation Loss: {avg_val_loss:.8f}")

        if (epoch + 1) % save_interval == 0:
            log_msg(f"Periodic save at epoch {epoch + 1}")
            current_state = {
                'epoch': epoch + 1,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_checkpoint(current_state, str(CHECKPOINT_DIR),
                            filename=f"{run_name}_last_checkpoint.pth")
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_state = {
        'epoch': num_epochs,
        'model_state_dict': decoder_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dict': input_dict
    }
    save_checkpoint(final_state, str(CHECKPOINT_DIR),
                    filename=f"{run_name}_final_model_{timestamp}.pth")
    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log_msg(f"Final model saved.")

    log_msg(f"Training completed")
    return decoder_model


# Fn to run instantiation, training, and evaluation (visual and metrics) of decoder ensemble model
def full_decoder_training_run(input_dict, train_loader, val_loader=None):

    log_msg(f"Running full decoder training and evaluation with inputs:\n {input_dict}")

    # Extract inputs
    in_channels = input_dict["in_channels"]
    embed_dim = input_dict["embed_dim"]
    ensemble_size = input_dict["ensemble_size"]
    num_classes = input_dict["num_classes"]
    learning_rate = input_dict["learning_rate"]

    # Instantiate Decoder Ensemble
    log_msg("Instantiating model")
    this_ensemble = DecoderEnsemble(ensemble_size, in_channels, embed_dim, num_classes)
    # Leaving this in for now as I'm not sure if it is a good idea or not
    #this_ensemble.apply(init_weights)
    this_ensemble.to(DEVICE)

    # Training of the Decoder ensemble
    optimizer = torch.optim.Adam(this_ensemble.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore index 0 because these are pixels not labelled by humans

    # Train
    trained_decoder_model = train_model(
        this_ensemble, train_loader, val_loader, criterion, optimizer, input_dict
    )

    return trained_decoder_model

