from models.ensemble import *
from utils.misc import *
from configs.config import *
import json
from datetime import datetime


def train_model(decoder_model, train_loader, val_loader, criterion, optimizer, input_dict):
    num_epochs = input_dict["num_epochs"]
    #lambda_div = input_dict["lambda_div"]
    #enforce_diversity = input_dict["enforce_diversity"]
    resume = input_dict["resume"]
    diversity_methods = input_dict.get("diversity_methods", [])
    lam_jsd = input_dict.get("lam_jsd", 0.0)
    lam_pearson = input_dict.get("lam_pearson", 0.0)
    lam_orth = input_dict.get("lam_orth", 0.0)

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
    best_val_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

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
        #epoch_task_loss = 0
        #epoch_div_loss = 0
        epoch_task_loss = 0
        epoch_jsd_loss = 0
        epoch_pearson_loss = 0
        epoch_orth_loss = 0

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

            #div_loss = 0
            #if enforce_diversity:
                # Various diversity options
            #    div_loss = js_divergence_loss(all_preds)
                #div_loss = pearson_diversity_loss(all_preds)

            # Make sure that for all diversity options, more loss is a bad thing, as our sign convention below needs to be respected
            # total_loss = task_loss + (lambda_div * div_loss)

            div_loss_jsd = 0.0
            div_loss_pearson = 0.0
            div_loss_orth = 0.0

            if "jsd" in diversity_methods:
                div_loss_jsd = js_divergence_loss(all_preds)

            if "pearson" in diversity_methods:
                div_loss_pearson = pearson_diversity_loss(all_preds)

            if "orth" in diversity_methods:
                div_loss_orth = orthogonality_loss(decoder_model)

            # Make sure that for all diversity options, more loss is a bad thing, as our sign convention below needs to be respected
            total_loss = task_loss \
                + lam_jsd * div_loss_jsd \
                + lam_pearson * div_loss_pearson \
                + lam_orth * div_loss_orth

            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item() if torch.is_tensor(task_loss) else task_loss
            epoch_jsd_loss += div_loss_jsd.item() if torch.is_tensor(div_loss_jsd) else div_loss_jsd
            epoch_pearson_loss += div_loss_pearson.item() if torch.is_tensor(div_loss_pearson) else div_loss_pearson
            epoch_orth_loss += div_loss_orth.item() if torch.is_tensor(div_loss_orth) else div_loss_orth

        #avg_task = epoch_task_loss / len(train_loader)
        #avg_div = epoch_div_loss / len(train_loader)
        #loss_history["train"].append(avg_task)
        avg_task = epoch_task_loss / len(train_loader)
        avg_jsd = epoch_jsd_loss / len(train_loader)
        avg_pearson = epoch_pearson_loss / len(train_loader)
        avg_orth = epoch_orth_loss / len(train_loader)

        loss_history["train"].append(avg_task)

        #log_msg(f"Epoch [{epoch+1}/{num_epochs}] - Task Loss: {avg_task:.8f}, Div Loss: {avg_div:.8f}")
        log_msg(f"Epoch [{epoch + 1}/{num_epochs}] - Task: {avg_task:.4f} | JSD: {avg_jsd:.4f} | Pearson: {avg_pearson:.4f} | Orth: {avg_orth:.4f}")

        """
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
            
            """
        if val_loader is not None:
            decoder_model.eval()
            val_task_loss = 0
            with torch.no_grad():
                for v_features, v_targets in val_loader:
                    v_features = v_features.to(DEVICE)
                    v_targets = v_targets.to(DEVICE).long()
                    all_preds = decoder_model(v_features)

                    total_val_loss = 0
                    for head_idx in range(decoder_model.M):
                        head_high = F.interpolate(all_preds[head_idx], size=(224, 224),
                                                  mode='bilinear', align_corners=False)
                        total_val_loss += criterion(head_high, v_targets)
                    mean_logits_low = all_preds.mean(dim=0)
                    mean_logits_high = F.interpolate(mean_logits_low, size=(224, 224), mode='bilinear')
                    total_val_loss += criterion(mean_logits_high, v_targets)
                    v_loss = total_val_loss / (decoder_model.M + 1)

                    val_task_loss += v_loss.item()
            avg_val_loss = val_task_loss / len(val_loader)

            # Save best model (on validation data) up until this point
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': decoder_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, str(CHECKPOINT_DIR), filename=f"{run_name}_best_model.pth")
                log_msg(f"New best val loss {best_val_loss:.6f} — best model saved")

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

        scheduler.step()

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
    optimizer = torch.optim.AdamW(this_ensemble.parameters(), lr=learning_rate, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore index 0 because these are pixels not labelled by humans

    # Train
    trained_decoder_model = train_model(
        this_ensemble, train_loader, val_loader, criterion, optimizer, input_dict
    )

    return trained_decoder_model

