from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path


import torchtext.daatsets as datasets
import torch
from torch.utils.data import Dataset, DataLoader, random_split


import warnings
from tqdm import tqdm
import os
from pathlib import Path


# HF datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Performs greedy decoding to generate translations using a trained transformer model.
    
    This function implements a simple greedy decoding strategy for sequence generation in 
    transformer-based neural machine translation. At each decoding step, it selects the token 
    with the highest probability as the next token in the sequence. While this approach is 
    computationally efficient, it may not always produce the optimal translation as it doesn't 
    consider future implications of current choices.

    Args:
        model: The trained transformer model with encoder and decoder components.
               Expected to have encode(), decode(), and project() methods.
        
        source (torch.Tensor): Input sequence tensor to translate.
                              Shape: [batch_size, source_sequence_length]
        
        source_mask (torch.Tensor): Mask tensor for the source sequence.
                                   Used to handle padding in source sequence.
                                   Shape: [batch_size, 1, 1, source_sequence_length]
        
        tokenizer_src: Tokenizer for the source language.
                      Must implement token_to_id() method for special tokens.
        
        tokenizer_tgt: Tokenizer for the target language.
                      Must implement token_to_id() method for special tokens.
        
        max_len (int): Maximum length of the generated sequence.
                       Acts as a safety limit to prevent infinite generation.
        
        device: Computing device (CPU/GPU) for tensor operations.
                Example: 'cuda', 'cpu', or 'mps'

    Returns:
        torch.Tensor: A 1D tensor containing the indices of the generated tokens.
                     Shape: [generated_sequence_length]

    Process Details:
        1. Initialization:
           - Retrieves special token indices (SOS, EOS) from target tokenizer
           - Computes encoder output once for efficiency (will be reused)
           - Initializes decoder sequence with start-of-sequence token
        
        2. Main Generation Loop:
           a. At each step:
              - Creates appropriate causal mask for current decoder sequence
              - Passes encoder output and current decoder sequence through decoder
              - Projects decoder output to vocabulary space
              - Selects token with highest probability using torch.max
              - Concatenates new token to growing decoder sequence
           
           b. Loop continues until either:
              - End-of-sequence token is generated
              - Maximum length (max_len) is reached
        
        3. Output Processing:
           - Squeezes output to remove batch dimension
           - Returns final sequence of token indices

    Technical Implementation Details:
        - Uses torch.cat for sequence building
        - Maintains proper tensor shapes throughout generation
        - Handles device placement for all tensors
        - Implements autoregressive generation (each step depends on previous outputs)

    Example Usage:
        >>> # Assuming model and tokenizers are properly initialized
        >>> source_text = "Hello world"
        >>> source_tokens = tokenizer_src.encode(source_text)
        >>> source_tensor = torch.tensor(source_tokens).unsqueeze(0)  # Add batch dimension
        >>> source_mask = create_padding_mask(source_tensor)  # Create appropriate mask
        >>> output_tokens = greedy_decode(model, source_tensor, source_mask, 
                                        tokenizer_src, tokenizer_tgt, max_len=50, 
                                        device='cuda')
        >>> translated_text = tokenizer_tgt.decode(output_tokens)

    Limitations:
        1. Greedy Search Limitations:
           - May miss better translations by always taking locally optimal choices
           - Can't backtrack from suboptimal choices
           - Might generate repetitive or stuck sequences
        
        2. Technical Limitations:
           - Memory usage grows linearly with sequence length
           - No batch processing (generates one sequence at a time)
           - No beam search or sampling strategies

    Notes:
        - For better translation quality, consider implementing beam search
        - Function assumes model is in evaluation mode
        - Careful attention to device placement is important for efficiency
        - Consider adding temperature parameter for controlling generation diversity
    """

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:-1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Runs validation on a trained transformer model and computes various performance metrics.
    
    Args:
        model: The transformer model to validate
        validation_ds: DataLoader containing validation examples
        tokenizer_src: Tokenizer for source language
        tokenizer_tgt: Tokenizer for target language
        max_len (int): Maximum sequence length for generation
        device: Computing device (cuda/cpu)
        print_msg: Function for printing formatted messages
        global_step (int): Current training step (for logging)
        writer: TensorBoard writer object for metric logging
        num_examples (int, optional): Number of examples to validate. Defaults to 2.
    
    Process:
        1. Sets model to evaluation mode
        2. Iterates through validation dataset:
           - Processes input sequences
           - Generates translations using greedy decoding
           - Collects source, target, and predicted texts
           - Prints formatted comparisons
        3. If TensorBoard writer is provided:
           - Computes Character Error Rate (CER)
           - Computes Word Error Rate (WER)
           - Computes BLEU Score
           - Logs all metrics to TensorBoard
    
    Notes:
        - Uses no_grad() for efficient inference
        - Requires batch size of 1 for validation
        - Attempts to format output based on console width
        - Early stops after num_examples
        
    Metrics Logged:
        - Character Error Rate (CER): Character-level accuracy
        - Word Error Rate (WER): Word-level accuracy
        - BLEU Score: Standard machine translation quality metric
    
    Example Output Format:
        ----------------------------------------
        SOURCE:     [source text]
        TARGET:     [target text]
        PREDICTED:  [predicted text]
        ----------------------------------------
    """
    
    model.eval()
    count = 0

    source_texts = []
    excepted = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())


            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, excepted)
        write.add_scalar('validation cer', cer, global_step)
        writer.flush()


        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, excepted)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()


        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, excepted)
        writer.add_scalar('validation BLUE', bleu, global_step)
        writer.flush()




