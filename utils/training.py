from models.ensemble import *
from utils.misc import *


def train_model(decoder_model, encoder_model, train_loader, num_epochs, criterion, optimizer, lambda_div, enforce_diversity):

    decoder_model.train()
    optimizer.zero_grad()

    print(f"Starting Training")

    for epoch in range(num_epochs):
        epoch_task_loss = 0
        epoch_div_loss  = 0

        # Iterate over DataLoader
        for batch_idx, (images, targets) in enumerate(train_loader):

            optimizer.zero_grad()

            # Move to device
            images = images.to(DEVICE)
            targets = targets.to(DEVICE).long()

            # Pass through frozen Encoder
            with torch.no_grad():
                grid_features = get_encoder_representation(images, encoder_model)
                # This should be returning [Batch, 1024, 28, 28]

            # Forward pass through Decoder Ensemble
            all_preds = decoder_model(grid_features)  # [5, Batch, 24, 224, 224]

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
                # IMPORTANT: Convert to probabilities for JSD!
                all_probs = torch.softmax(all_preds, dim=2)
                div_loss = js_divergence_loss(all_probs)

            # 4. Total Loss Calculation
            # Higher div_loss = More diverse = Lower total_loss
            # Keep in mind here the names are maybe not ideal, a higher div_loss is good as it means the heads are more diverse
            # this is then subtracted to make the total loss smaller
            total_loss = task_loss - (lambda_div * div_loss)

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

        # Print epoch summary
        avg_task = epoch_task_loss / len(train_loader)
        avg_div = epoch_div_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Task Loss: {avg_task:.4f}, Div Loss: {avg_div:.4f}")

    print(f"Training completed")

    return decoder_model

# Fn to run instantiation, training, and evaluation (visual and metrics) of decoder ensemble model
def full_decoder_training_run(input_dict, train_loader):

    print(f"Running full decoder training and evaluation with inputs:\n {input_dict}")

    # Extract inputs
    in_channels = input_dict["in_channels"]
    embed_dim = input_dict["embed_dim"]
    ensemble_size = input_dict["ensemble_size"]
    num_classes = input_dict["num_classes"]
    num_epochs = input_dict["num_epochs"]
    lambda_div = input_dict["lambda_div"]
    enforce_diversity = input_dict["enforce_diversity"]
    learning_rate = input_dict["learning_rate"]
    hide_unlabelled_pixels = input_dict["hide_unlabelled_pixels"]

    # Load Encoder
    encoder_model = initialize_clay_encoder()

    # Instantiate Decoder Ensemble
    print("Instantiating model")
    this_ensemble = DecoderEnsemble(ensemble_size, in_channels, embed_dim, num_classes)
    this_ensemble.apply(init_weights)
    this_ensemble.to(DEVICE)

    # Training of the Decoder ensemble
    optimizer = torch.optim.Adam(this_ensemble.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore index 0 because these are pixels not labelled by humans

    # Train
    trained_decoder_model = train_model(
        this_ensemble, encoder_model, train_loader, num_epochs, criterion, optimizer, lambda_div, enforce_diversity
    )

    # Since we don't have a single 'ground_truth' anymore,
    # we grab one batch from the loader to use as our "Visual Test"
    test_images, test_ground_truth = get_random_batch(train_loader, DEVICE)
    test_images = test_images.to(DEVICE)

    # We need to get the encoder features for this test batch to visualize them
    with torch.no_grad():
        test_grid_features = get_encoder_representation(test_images, encoder_model)

    # Evaluate
    print("Evaluating model on test batch")
    # We use index [0] to just look at the first image in that test batch
    mean_probs, class_map, variance_map, total_entropy, mutual_info = get_decoder_output_maps(
        trained_decoder_model, test_grid_features[0].unsqueeze(0)
    )

    # Now call visualization using the mask from that same test image
    visualise_all_metrics(class_map, variance_map, total_entropy, mutual_info, test_ground_truth[0], hide_unlabelled_pixels)
    display_truth_proportions(test_ground_truth[0])

    # 2. Get Confidence without destroying your existing class_map
    # Use dim=1 because mean_probs is [1, 25, 224, 224]
    conf_tensor, _ = torch.max(mean_probs, dim=1)

    # 3. Squeeze to get [224, 224]
    confidence_map_2d = conf_tensor.squeeze().cpu().numpy()
    ground_truth_2d = test_ground_truth[0].squeeze().cpu().numpy()

    # 4. Call metrics using the class_map you already have from Step 1
    # Note: class_map is already 2D numpy from your function return!
    miou, overall_accuracy, avg_unc, ece = evaluate_metrics(
        class_map,
        ground_truth_2d,
        confidence_map_2d
    )

    return miou, overall_accuracy, avg_unc, ece
