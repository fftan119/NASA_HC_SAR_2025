print("\nVisualizing sample predictions on test set...")

def plot_ground_truth_and_predictions(pil_img, predictions, ground_truths, id2label, confidence_threshold=0.9):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # Create 1 row, 2 columns for subplots

    # Plot Ground Truth
    ax_gt = axes[0]
    ax_gt.imshow(pil_img)
    ax_gt.set_title('Ground Truth')
    ax_gt.axis('off')

    img_width, img_height = pil_img.size
    for gt_box, gt_label_id in zip(ground_truths['boxes'], ground_truths['labels']):
        cx_norm, cy_norm, w_norm, h_norm = gt_box
        xmin_gt = (cx_norm - w_norm / 2) * img_width
        ymin_gt = (cy_norm - h_norm / 2) * img_height
        xmax_gt = (cx_norm + w_norm / 2) * img_width
        ymax_gt = (cy_norm + h_norm / 2) * img_height
        
        label_gt = id2label[gt_label_id]
        ax_gt.add_patch(patches.Rectangle((xmin_gt, ymin_gt), xmax_gt - xmin_gt, ymax_gt - ymin_gt,
                                       fill=False, edgecolor='blue', linestyle='--', linewidth=2))
        ax_gt.text(xmin_gt, ymin_gt - 10, f'GT: {label_gt}',
                bbox=dict(facecolor='blue', alpha=0.7), fontsize=8, color='white')

    # Plot Predictions
    ax_pred = axes[1]
    ax_pred.imshow(pil_img)
    ax_pred.set_title(f'Predicted (Confidence > {confidence_threshold:.2f})')
    ax_pred.axis('off')

    for score, label_id, (xmin, ymin, xmax, ymax) in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        if score > confidence_threshold: # Apply confidence threshold for display
            label = id2label[label_id]
            ax_pred.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                               fill=False, edgecolor='red', linewidth=2))
            ax_pred.text(xmin, ymin - 10, f'{label}: {score:.2f}',
                         bbox=dict(facecolor='red', alpha=0.7), fontsize=8, color='white')
    
    plt.tight_layout()
    plt.show()


# Iterate through all samples collected and plot them
for i, sample in enumerate(sample_predictions):
    print(f"\n--- Image {i+1}/{len(sample_predictions)} (Image ID: {sample['image_id']}) ---")
    plot_ground_truth_and_predictions(
        sample['original_image'],
        sample['predictions'],
        sample['ground_truths'],
        ID_TO_LABEL,
        confidence_threshold=0.8 # You can adjust this threshold for visualization
    )

print("\nTest inference and visualization complete!")