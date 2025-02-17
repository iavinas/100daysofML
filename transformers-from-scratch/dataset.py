import torch



def causal_mask(size):
    """
    Creates a causal mask for transformer decoder self-attention.
    
    Args:
        size (int): The size of the square mask matrix
        
    Returns:
        torch.Tensor: A boolean mask of shape (1, size, size) where True values 
                     indicate positions that should be attended to, and False values 
                     indicate positions that should be masked out.
                     
    The mask ensures each position can only attend to previous positions and itself,
    preventing information flow from future tokens. The resulting pattern is:
        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
    where 1 (True) indicates valid attention positions and 0 (False) indicates 
    masked positions.
    """
    
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    """ example output
        0 1 1 1
        0 0 1 1
        0 0 0 1
        0 0 0 0
    """
    return mask == 0 # invert the pattern