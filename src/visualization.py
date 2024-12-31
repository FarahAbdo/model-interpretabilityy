import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def plot_activations(activations):
    # Detach the tensor from the computation graph before converting to NumPy
    activations = activations.detach().cpu()  # Ensure it's on the CPU if it's on a GPU
    if len(activations.shape) > 2:
        activations = activations.mean(dim=0)
    
    # Now you can safely call .numpy()
    sns.heatmap(activations.numpy(), cmap='viridis')
    plt.title('Layer Activations', fontsize=14)
    plt.xlabel('Neuron Index', fontsize=12)
    plt.ylabel('Activation Value', fontsize=12)
    plt.show()


def plot_attributions(attribution):
    attribution = attribution.detach().cpu().numpy()  # Ensure it's on the CPU and detached from the graph
    
    # If the attribution is 4D (e.g., batch_size, channels, height, width)
    if attribution.ndim == 4:
        # Average across the batch dimension (axis=0) and channels (axis=1), then take the spatial dimensions
        attribution = attribution.mean(axis=0)  # Average over batch size
        attribution = attribution.mean(axis=0)  # Average over channels (if necessary, depending on your model)

    # If the attribution is 3D, you can take the mean across channels or choose a strategy that fits your model
    elif attribution.ndim == 3:
        attribution = attribution.mean(axis=0)  # Average across channels

    if attribution.ndim != 2:
        raise ValueError(f"Expected 2D attribution map, but got {attribution.ndim}D tensor.")
    
    # Plot the heatmap
    sns.heatmap(attribution, cmap='seismic', center=0)


def plot_gradcam(original_image, gradcam_map):
    """
    Plot original image alongside Grad-CAM visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    img = original_image[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    # Plot Grad-CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(gradcam_map[0, 0].cpu().numpy(), cmap='jet')
    plt.title('Grad-CAM Attention Map', fontsize=14)
    plt.colorbar(label='Attention Score')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(1, 3, 3)
    heatmap = gradcam_map[0, 0].cpu().numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]
    overlay = 0.6 * heatmap + 0.4 * img
    plt.imshow(overlay)
    plt.title('Overlay Visualization', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_summary_plot(results):
    """
    Create a summary plot combining all analysis results
    """
    plt.figure(figsize=(20, 10))
    
    # Layer activations summary
    plt.subplot(2, 2, 1)
    sns.heatmap(results['activations'].mean(dim=0).numpy(), cmap='viridis')
    plt.title('Layer Activation Summary')
    
    # Feature attribution summary
    plt.subplot(2, 2, 2)
    sns.heatmap(results['attributions'].mean(dim=0).detach().numpy(), cmap='seismic', center=0)
    plt.title('Feature Attribution Summary')
    
    # Grad-CAM summary
    plt.subplot(2, 2, 3)
    plt.imshow(results['gradcam'][0, 0].cpu().numpy(), cmap='jet')
    plt.title('Grad-CAM Analysis')
    
    # Metrics summary
    plt.subplot(2, 2, 4)
    metrics = {
        'Mean Activation': float(results['activations'].mean()),
        'Max Attribution': float(results['attributions'].max()),
        'Attention Coverage': float((results['gradcam'] > 0.5).float().mean())
    }
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Key Metrics Summary')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()