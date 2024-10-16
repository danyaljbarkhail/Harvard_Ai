import torch
import numpy as np

# Function to get the index of the [MASK] token in the input
def get_mask_token_index(mask_token_id, inputs):
    """
    Find the index of the token that matches the [MASK] token.
    
    Args:
        mask_token_id: The token ID that corresponds to the [MASK] token.
        inputs: The tokenized inputs which contain 'input_ids'.
        
    Returns:
        The index of the [MASK] token in the inputs, or None if not found.
    """
    token_ids = inputs['input_ids'][0].detach().cpu().numpy() if torch.is_tensor(inputs['input_ids'][0]) else inputs['input_ids'][0]
    for idx, token_id in enumerate(token_ids):
        if token_id == mask_token_id:
            return idx
    return None

# Function to convert attention score to a grayscale color value
def get_color_for_attention_score(attention_score):
    """
    Convert attention score to a grayscale color value.
    
    Args:
        attention_score: The attention score, typically between 0 and 1.
        
    Returns:
        A tuple representing the RGB color (gray_value, gray_value, gray_value).
    """
    gray_value = int((1 - min(max(attention_score, 0), 1)) * 255)
    return (gray_value, gray_value, gray_value)

# Example: Function to visualize attentions
def visualize_attentions(tokens, attentions):
    """
    Visualize the attentions for each layer and head.
    
    Args:
        tokens: List of tokens.
        attentions: The attention weights from the model.
        
    Returns:
        Visualization for each attention head and layer.
    """
    num_layers = len(attentions)
    for layer in range(num_layers):
        num_heads = attentions[layer][0].size(0)
        print(f"Layer {layer + 1} visualization:")
        for head in range(num_heads):
            print(f"  Head {head + 1}:")
            visualize_attention_for_head(layer + 1, head + 1, tokens, attentions[layer][0][head])

def visualize_attention_for_head(layer, head, tokens, attention_scores):
    """
    Visualizes attention for a specific head.
    
    Args:
        layer: The layer number.
        head: The head number.
        tokens: List of tokens.
        attention_scores: Attention scores for this head.
        
    Returns:
        None. Visualizes the attention diagram.
    """
    # Just printing attention matrix for simplicity
    print(f"Attention scores for layer {layer}, head {head}:")
    for i, token in enumerate(tokens):
        attention_for_token = attention_scores[i].detach().cpu().numpy() if torch.is_tensor(attention_scores[i]) else attention_scores[i]
        color = get_color_for_attention_score(np.mean(attention_for_token))
        print(f"  {token}: {attention_for_token} -> Color: {color}")

# Example use of get_mask_token_index
# mask_token_id = 103  # For BERT [MASK] token ID
# inputs = {"input_ids": torch.tensor([[101, 2003, mask_token_id, 102]])}
# mask_index = get_mask_token_index(mask_token_id, inputs)
# print(f"The [MASK] token is at index: {mask_index}")
