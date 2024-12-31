import torch
import torch.nn as nn
import numpy as np
from .visualization import plot_activations, plot_attributions, plot_gradcam

class ModelInterpreter:
    def __init__(self, model):
        """Initialize the interpreter with a PyTorch model"""
        self.model = model
        self.layer_outputs = {}
        self.layer_gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on all layers"""
        def forward_hook(module, input, output):
            self.layer_outputs[module] = output
            
        def backward_hook(module, grad_input, grad_output):
            self.layer_gradients[module] = grad_output[0]
            
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
    
    def get_layer_activations(self, input_data, layer_idx=-1):
        """Get activations for a specific layer"""
        self.model.eval()
        _ = self.model(input_data)
        layer = list(self.layer_outputs.keys())[layer_idx]
        return self.layer_outputs[layer]
    
    def compute_feature_attribution(self, input_data, target_class):
        """Compute feature attribution using integrated gradients"""
        self.model.eval()
        input_data.requires_grad = True
        baseline = torch.zeros_like(input_data)
        steps = 50
        alphas = torch.linspace(0, 1, steps)
        
        integrated_gradients = torch.zeros_like(input_data)
        
        for alpha in alphas:
            # Ensure the interpolated input is a leaf tensor with requires_grad set to True
            interpolated_input = baseline + alpha * (input_data - baseline)
            interpolated_input = interpolated_input.clone().detach().requires_grad_(True)
            
            output = self.model(interpolated_input)
            score = output[:, target_class].sum()
            
            self.model.zero_grad()  # Clear previous gradients
            score.backward()  # Compute gradients
            
            integrated_gradients += interpolated_input.grad  # Accumulate gradients
        
        # Calculate final attribution
        attribution = (input_data - baseline) * integrated_gradients / steps
        return attribution


    def compute_gradcam(self, input_data, target_class, layer_idx=-2):
        """
        Compute Grad-CAM activation maps showing which parts of the input
        were most important for the model's decision.
        """
        self.model.eval()
        
        # Get the target layer (usually the last convolutional layer)
        target_layer = list(self.layer_outputs.keys())[layer_idx]
        
        # Forward pass
        output = self.model(input_data)
        
        if len(output.shape) > 1:
            score = output[:, target_class].sum()
        else:
            score = output.sum()
            
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.layer_gradients[target_layer]
        activations = self.layer_outputs[target_layer]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
        gradcam = torch.relu(gradcam)  # ReLU to only show positive contributions
        
        # Normalize
        gradcam = gradcam - gradcam.min()
        gradcam = gradcam / (gradcam.max() + 1e-8)
        
        # Upsample to input size
        gradcam = torch.nn.functional.interpolate(
            gradcam,
            size=input_data.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return gradcam
    
    def analyze_model(self, input_data, target_class=None):
        """Perform complete model analysis with all MI techniques"""
        results = {}
        
        print("1. Analyzing Layer Activations...")
        activations = self.get_layer_activations(input_data)
        plot_activations(activations)
        results['activations'] = activations
        
        if target_class is not None:
            print("\n2. Computing Feature Attribution...")
            attributions = self.compute_feature_attribution(input_data, target_class)
            plot_attributions(attributions)
            results['attributions'] = attributions
            
            print("\n3. Generating Grad-CAM Visualization...")
            gradcam = self.compute_gradcam(input_data, target_class)
            plot_gradcam(input_data, gradcam)
            results['gradcam'] = gradcam
            
        return results

    def generate_interpretation_report(self, input_data, target_class):
        """
        Generate a comprehensive interpretation report with all analysis techniques
        """
        results = self.analyze_model(input_data, target_class)
        
        # Return detailed analysis dictionary
        report = {
            'model_summary': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'layers_analyzed': len(self.layer_outputs)
            },
            'layer_activations': {
                'shape': results['activations'].shape,
                'mean_activation': float(results['activations'].mean()),
                'max_activation': float(results['activations'].max())
            },
            'feature_attribution': {
                'total_attribution': float(results['attributions'].sum()),
                'mean_attribution': float(results['attributions'].mean()),
                'max_attribution': float(results['attributions'].max())
            },
            'gradcam_analysis': {
                'attention_mean': float(results['gradcam'].mean()),
                'attention_coverage': float((results['gradcam'] > 0.5).float().mean())
            }
        }
        
        return report