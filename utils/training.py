from models.ensemble import *
from utils.misc import *
from configs.config import *
from profile_flops import ensure_flops_profile, start_compute_tracking, record_compute_cost
import json
from datetime import datetime


def _ensemble_forward(decoder_model, features, bottleneck=None):
    """Runs the M-head ensemble forward pass. With a bottleneck, each head samples from
    its own per-head distribution. The student always trains on the raw encoder features
    (never a bottleneck-derived embedding), so its input distribution is identical whether
    or not the bottleneck is active. Also returns the mean learned std across heads/pixels
    (0 when no bottleneck), purely for diagnostic logging of whether the bottleneck is
    collapsing its injected noise toward zero over training."""
    if bottleneck is not None:
        mus, logvars, kl = bottleneck(features)
        z_samples = bottleneck.sample(mus, logvars)
        all_preds = torch.stack([decoder_model.heads[m](z_samples[m]) for m in range(decoder_model.M)])
        mean_std = torch.stack([lv.exp().sqrt().mean() for lv in logvars]).mean()
        return all_preds, kl, mean_std
    else:
        zero = torch.tensor(0.0, device=features.device)
        return decoder_model(features), zero, zero


def student_step(student_model, student_optimizer, features, teacher_logits_detached, temperature=1.0):
    """Single batch update for the AS4 student. Teacher logits must already be detached."""
    student_optimizer.zero_grad()
    alphas = student_model(features)
    loss = endd_loss(alphas, teacher_logits_detached, temperature=temperature)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
    student_optimizer.step()
    return loss.item()


def student_val_epoch(student_model, teacher_model, val_loader, temperature=1.0, bottleneck=None):
    """Compute student EnDD loss on the validation set."""
    student_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, _ in val_loader:
            features = features.to(DEVICE)
            teacher_logits, _, _ = _ensemble_forward(teacher_model, features, bottleneck)
            alphas = student_model(features)
            total_loss += endd_loss(alphas, teacher_logits, temperature=temperature).item()
    return total_loss / len(val_loader)


def train_model(decoder_model, train_loader, val_loader, criterion, optimizer, input_dict,
                student_model=None, student_optimizer=None, student_scheduler=None, flops_profile=None,
                bottleneck=None):
    num_epochs = input_dict["num_epochs"]
    #lambda_div = input_dict["lambda_div"]
    #enforce_diversity = input_dict["enforce_diversity"]
    resume = input_dict["resume"]
    diversity_methods = input_dict.get("diversity_methods", [])
    lam_jsd = input_dict.get("lam_jsd", 0.0)
    lam_pearson = input_dict.get("lam_pearson", 0.0)
    lam_orth = input_dict.get("lam_orth", 0.0)
    lam_kl = input_dict.get("lam_kl", 0.0)

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
    best_student_val_loss = float('inf')
    student_warmup_epochs = input_dict.get("student_warmup_epochs", 10)
    student_T_start = input_dict.get("student_T_start", 4.0)
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
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_student_val_loss = checkpoint.get('best_student_val_loss', float('inf'))
        if bottleneck is not None and checkpoint.get('bottleneck_state_dict') is not None:
            bottleneck.load_state_dict(checkpoint['bottleneck_state_dict'])
            log_msg("--> Resumed VariationalBottleneck weights")
        loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_history = json.load(f)
        log_msg(f"--> Successfully resumed from epoch {start_epoch + 1}")

    elif not resume and checkpoint_path.exists():
        log_msg("!! WARNING: Checkpoint exists but --resume was not used. !!")
        log_msg("!! This run will OVERWRITE your existing checkpoint. !!")

    log_msg(f"Starting Training")

    compute_start_epoch = start_epoch  # captured before the loop mutates epoch, for the compute-cost summary
    training_start_time = start_compute_tracking()

    for epoch in range(start_epoch, num_epochs):
        log_msg(f"Starting epoch {epoch+1}...")
        #epoch_task_loss = 0
        #epoch_div_loss = 0
        epoch_task_loss = 0
        epoch_jsd_loss = 0
        epoch_pearson_loss = 0
        epoch_orth_loss = 0
        epoch_kl_loss = 0
        epoch_std_loss = 0
        epoch_student_loss = 0.0
        student_T = get_temperature(epoch - student_warmup_epochs, num_epochs - student_warmup_epochs, student_T_start) \
            if (student_model is not None and epoch >= student_warmup_epochs) else None

        decoder_model.train()

        for features, targets in train_loader:
            optimizer.zero_grad()
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()

            all_preds, kl_loss, mean_std = _ensemble_forward(decoder_model, features, bottleneck)

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

            if "orthogonality" in diversity_methods:
                div_loss_orth = orthogonality_loss(decoder_model)

            # Make sure that for all diversity options, more loss is a bad thing, as our sign convention below needs to be respected
            total_loss = task_loss \
                + lam_jsd * div_loss_jsd \
                + lam_pearson * div_loss_pearson \
                + lam_orth * div_loss_orth \
                + lam_kl * kl_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), max_norm=1.0)
            if bottleneck is not None:
                torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), max_norm=1.0)
            optimizer.step()

            if student_model is not None and epoch >= student_warmup_epochs:
                epoch_student_loss += student_step(student_model, student_optimizer, features.detach(), all_preds.detach(),
                                                   temperature=student_T)

            epoch_task_loss += task_loss.item() if torch.is_tensor(task_loss) else task_loss
            epoch_jsd_loss += div_loss_jsd.item() if torch.is_tensor(div_loss_jsd) else div_loss_jsd
            epoch_pearson_loss += div_loss_pearson.item() if torch.is_tensor(div_loss_pearson) else div_loss_pearson
            epoch_orth_loss += div_loss_orth.item() if torch.is_tensor(div_loss_orth) else div_loss_orth
            epoch_kl_loss += kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
            epoch_std_loss += mean_std.item() if torch.is_tensor(mean_std) else mean_std

        #avg_task = epoch_task_loss / len(train_loader)
        #avg_div = epoch_div_loss / len(train_loader)
        #loss_history["train"].append(avg_task)
        avg_task = epoch_task_loss / len(train_loader)
        avg_jsd = epoch_jsd_loss / len(train_loader)
        avg_pearson = epoch_pearson_loss / len(train_loader)
        avg_orth = epoch_orth_loss / len(train_loader)
        avg_kl = epoch_kl_loss / len(train_loader)
        avg_std = epoch_std_loss / len(train_loader)

        loss_history["train"].append(avg_task)
        if "jsd" in diversity_methods:
            loss_history.setdefault("train_jsd", []).append(avg_jsd)
        if "pearson" in diversity_methods:
            loss_history.setdefault("train_pearson", []).append(avg_pearson)
        if "orthogonality" in diversity_methods:
            loss_history.setdefault("train_orth", []).append(avg_orth)
        if bottleneck is not None:
            loss_history.setdefault("train_kl", []).append(avg_kl)
            loss_history.setdefault("train_bottleneck_std", []).append(avg_std)
        if student_model is not None and epoch >= student_warmup_epochs:
            loss_history.setdefault("student_train", []).append(epoch_student_loss / len(train_loader))

        log_msg(f"Epoch [{epoch + 1}/{num_epochs}] - Task: {avg_task:.4f} | JSD: {avg_jsd:.4f} | Pearson: {avg_pearson:.4f} | Orth: {avg_orth:.4f} | KL: {avg_kl:.4f} | BottleneckStd: {avg_std:.4f}")
        if student_model is not None and epoch >= student_warmup_epochs:
            log_msg(f"  Student EnDD train loss: {epoch_student_loss / len(train_loader):.4f} | T={student_T:.2f}")

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
                    all_preds, _, _ = _ensemble_forward(decoder_model, v_features, bottleneck)

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
                    'bottleneck_state_dict': bottleneck.state_dict() if bottleneck is not None else None,
                }, str(CHECKPOINT_DIR), filename=f"{run_name}_best_model.pth")
                log_msg(f"New best val loss {best_val_loss:.6f} — best model saved")

            loss_history["val"].append(avg_val_loss)
            log_msg(f"Validation Loss: {avg_val_loss:.8f}")

            if student_model is not None and epoch >= student_warmup_epochs:
                avg_student_val = student_val_epoch(student_model, decoder_model, val_loader, temperature=student_T,
                                                     bottleneck=bottleneck)
                loss_history.setdefault("student_val", []).append(avg_student_val)
                log_msg(f"  Student Val Loss: {avg_student_val:.6f}")
                if avg_student_val < best_student_val_loss:
                    best_student_val_loss = avg_student_val
                    save_checkpoint(
                        {'epoch': epoch + 1, 'model_state_dict': student_model.state_dict()},
                        str(CHECKPOINT_DIR), filename=f"{run_name}_best_student.pth"
                    )
                    log_msg(f"  New best student val loss {best_student_val_loss:.6f} — saved")

        if (epoch + 1) % save_interval == 0:
            log_msg(f"Periodic save at epoch {epoch + 1}")
            current_state = {
                'epoch': epoch + 1,
                'model_state_dict': decoder_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_student_val_loss': best_student_val_loss,
                'bottleneck_state_dict': bottleneck.state_dict() if bottleneck is not None else None,
            }
            save_checkpoint(current_state, str(CHECKPOINT_DIR),
                            filename=f"{run_name}_last_checkpoint.pth")
            loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f, indent=2)

        scheduler.step()
        if student_scheduler is not None:
            student_scheduler.step()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_state = {
        'epoch': num_epochs,
        'model_state_dict': decoder_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dict': input_dict,
        'bottleneck_state_dict': bottleneck.state_dict() if bottleneck is not None else None,
    }
    save_checkpoint(final_state, str(CHECKPOINT_DIR),
                    filename=f"{run_name}_final_model_{timestamp}.pth")
    if student_model is not None:
        save_checkpoint(
            {'epoch': num_epochs, 'model_state_dict': student_model.state_dict()},
            str(CHECKPOINT_DIR), filename=f"{run_name}_final_student_{timestamp}.pth"
        )
    loss_path = os.path.join(runs_dir, f"{run_name}_loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log_msg(f"Final model saved.")

    if flops_profile is not None:
        record_compute_cost(flops_profile, epochs_completed=num_epochs - compute_start_epoch,
                            batches_per_epoch=len(train_loader), start_time=training_start_time,
                            run_name=run_name)

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

    bottleneck = None
    if input_dict.get("use_variational_bottleneck", False):
        sigma_prior = input_dict.get("sigma_prior", 1.0)
        bottleneck = VariationalBottleneck(channels=in_channels, num_heads=ensemble_size, sigma_prior=sigma_prior)
        bottleneck.to(DEVICE)
        log_msg(f"VariationalBottleneck enabled (sigma_prior={sigma_prior}, lam_kl={input_dict.get('lam_kl', 0.0)})")

    # Training of the Decoder ensemble
    trainable_params = list(this_ensemble.parameters())
    if bottleneck is not None:
        trainable_params += list(bottleneck.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.05)

    use_focal = input_dict.get("use_focal_loss", False)
    use_dice  = input_dict.get("use_dice_loss", False)
    if use_focal:
        gamma = input_dict.get("focal_gamma", 2.0)
        criterion = FocalLoss(gamma=gamma, ignore_index=0)
        log_msg(f"Using Focal Loss (gamma={gamma})")
    elif use_dice:
        criterion = DiceLoss(ignore_index=0)
        log_msg("Using Dice Loss")
    else:
        log_msg("Computing class weights (log-damped inverse frequency, capped at 10)...")
        class_weights = compute_class_weights(train_loader, num_classes, ignore_index=0).to(DEVICE)
        log_msg(f"Class weights — min: {class_weights[class_weights>0].min():.3f}, median: {class_weights[class_weights>0].median():.3f}, max: {class_weights.max():.3f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

    student_model = None
    student_optimizer = None
    student_scheduler = None
    if input_dict.get("train_student", False):
        student_model = StudentHead(in_channels, embed_dim, num_classes)
        student_model.to(DEVICE)
        student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=0.05)
        student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            student_optimizer, T_max=input_dict["num_epochs"], eta_min=1e-6
        )
        log_msg("StudentHead instantiated for parallel AS4 training")

    flops_profile = ensure_flops_profile(
        config={
            "dataset": "fbp",
            "encoder_type": "frozen",
            "n_unfrozen_blocks": None,
            "ensemble_size": input_dict["ensemble_size"],
            "decoder_embed_dim": input_dict["decoder_embed_dim"],
            "num_classes": input_dict["num_classes"],
            "in_channels": None,
            "batch_size": input_dict["batch_size"],
            "include_student": input_dict.get("train_student", False),
            "diversity_methods": input_dict.get("diversity_methods", []),
        },
        decoder=this_ensemble,
        student=student_model,
    )

    # Train
    trained_decoder_model = train_model(
        this_ensemble, train_loader, val_loader, criterion, optimizer, input_dict,
        student_model=student_model, student_optimizer=student_optimizer, student_scheduler=student_scheduler,
        flops_profile=flops_profile, bottleneck=bottleneck
    )

    return trained_decoder_model, student_model, bottleneck

