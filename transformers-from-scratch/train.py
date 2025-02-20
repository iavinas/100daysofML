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
    Read Day 2 in Readme for more detailed explanation.
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


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizers(config, ds):
    """Build both tokenizers with guaranteed same IDs for special tokens"""
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    
    # Create or load tokenizers
    src_tokenizer_path = Path(config['tokenizer_file'].format(src_lang))
    tgt_tokenizer_path = Path(config['tokenizer_file'].format(tgt_lang))
    
    # Create new tokenizers if they don't exist
    if not Path.exists(src_tokenizer_path) or not Path.exists(tgt_tokenizer_path):
        # Initialize both tokenizers the same way
        src_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tgt_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        
        src_tokenizer.pre_tokenizer = Whitespace()
        tgt_tokenizer.pre_tokenizer = Whitespace()
        
        # Train both tokenizers
        src_trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=2)
        tgt_trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=2)
        
        src_tokenizer.train_from_iterator(get_all_sentences(ds, src_lang), trainer=src_trainer)
        tgt_tokenizer.train_from_iterator(get_all_sentences(ds, tgt_lang), trainer=tgt_trainer)
        
        # Manually ensure special tokens have the same IDs in both tokenizers
        # This is the key part - we modify the vocabulary mapping
        for i, token in enumerate(special_tokens):
            # Force special tokens to have IDs 0,1,2,3
            src_tokenizer.token_to_id_[token] = i
            tgt_tokenizer.token_to_id_[token] = i
            
            # Update the reverse mapping too
            src_tokenizer.id_to_token_[i] = token
            tgt_tokenizer.id_to_token_[i] = token
        
        # Save the tokenizers
        src_tokenizer.save(str(src_tokenizer_path))
        tgt_tokenizer.save(str(tgt_tokenizer_path))
    else:
        # Load existing tokenizers
        src_tokenizer = Tokenizer.from_file(str(src_tokenizer_path))
        tgt_tokenizer = Tokenizer.from_file(str(tgt_tokenizer_path))
        
        # Verify special tokens have the same IDs
        for token in special_tokens:
            src_id = src_tokenizer.token_to_id(token)
            tgt_id = tgt_tokenizer.token_to_id(token)
            if src_id != tgt_id:
                raise ValueError(f"Tokenizers have different IDs for {token}: {src_id} vs {tgt_id}")
    
    return src_tokenizer, tgt_tokenizer
    

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # build tokenizers
    tokenizer_src, tokenizer_tgt = get_or_build_tokenizer(config, ds_raw)

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocal_tgt_len):
    model = build_transformer(vocab_src_len, vocal_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model


def train(config):
    """
        Processes a batch of data through a transformer model's encoder-decoder architecture and calculates loss.
        
        This function implements the core forward pass of a sequence-to-sequence transformer model, 
        following the standard encoder-decoder architecture with self-attention mechanisms.
        
        Parameters:
        -----------
        model : nn.Module
            The transformer model containing encode, decode, and project methods
        encoder_input : torch.Tensor
            Input sequence to the encoder with shape (batch_size, seq_len)
            Contains token IDs of the source sequence (e.g., the sentence to translate)
        decoder_input : torch.Tensor
            Input sequence to the decoder with shape (batch_size, seq_len)
            Contains token IDs of the target sequence, typically right-shifted
            (start token followed by target sequence except last token)
        encoder_mask : torch.Tensor
            Attention mask for the encoder with shape (batch_size, 1, 1, seq_len)
            Prevents attention to padding tokens in the source sequence by setting their positions to 0
        decoder_mask : torch.Tensor
            Combined padding and causal mask for the decoder with shape (batch_size, 1, seq_len, seq_len)
            Contains both padding mask (to prevent attending to pad tokens) and
            causal/look-ahead mask (to prevent attending to future tokens during training)
        labels : torch.Tensor
            Ground truth token IDs with shape (batch_size, seq_len)
            Contains the expected next tokens at each position (target sequence shifted left)
        loss_fn : nn.Module
            Loss function, typically CrossEntropyLoss with label smoothing and padding token ignored
        device : torch.device
            Device (CPU, GPU, MPS) where computation is performed
            
        Returns:
        --------
        loss : torch.Tensor
            Scalar tensor containing the computed loss value
        proj_output : torch.Tensor
            Model predictions with shape (batch_size, seq_len, vocab_size)
            Contains logits over the vocabulary for each position
            
        Detailed Process:
        ----------------
        1. Encoder Stage:
            The encoder transforms the input sequence into contextual representations through
            several transformer encoder layers. Each encoder layer consists of:
            - Multi-head self-attention: Allows each position to attend to all positions in the input
            - Position-wise feed-forward network: Applies the same feed-forward network to each position
            - Layer normalization and residual connections: Stabilize training
            
            During encoding:
            - Input tokens are embedded into vectors (embedding lookup)
            - Positional encodings are added to provide position information
            - The encoder_mask ensures padding tokens don't contribute to attention calculations
            - Output shape: (batch_size, seq_len, d_model), where d_model is the model's hidden dimension
        
        2. Decoder Stage:
            The decoder processes the target sequence while attending to the encoder's output.
            Each decoder layer contains:
            - Masked multi-head self-attention: Allows each position to attend only to previous positions
            - Multi-head cross-attention: Attends to the encoder's output
            - Position-wise feed-forward network: Same as in encoder
            - Layer normalization and residual connections: Same as in encoder
            
            During decoding:
            - The decoder_mask prevents attention to:
                a) Future tokens (causal/look-ahead mask)
                b) Padding tokens in both source and target sequences
            - The decoder layers build target representations with knowledge of the source sequence
            - Output shape: (batch_size, seq_len, d_model)
        
        3. Projection Stage:
            The projection layer converts decoder outputs into vocabulary predictions:
            - Applies a linear transformation: d_model → vocab_size
            - Maps hidden representations to logits over the vocabulary
            - No activation function is applied (raw logits are used with CrossEntropyLoss)
            - Output shape: (batch_size, seq_len, vocab_size)
        
        4. Loss Calculation:
            The loss function measures prediction accuracy against ground truth:
            - Reshapes predictions to (batch_size×seq_len, vocab_size)
            - Reshapes labels to (batch_size×seq_len)
            - Computes cross-entropy loss between predictions and labels
            - Ignores loss contribution from padding tokens
            - Applies label smoothing to prevent overconfidence and improve generalization
            
        Why Cross-Entropy Loss for Language Modeling:
        --------------------------------------------
        Cross-entropy is the standard loss function for language modeling despite language being expressible 
        in multiple ways, for several important reasons:
        
        1. Probabilistic Nature of Language:
            - Language generation is inherently probabilistic - multiple valid continuations exist for any context
            - Cross-entropy measures the difference between predicted probability distributions and ground truth
            - It allows the model to assign probability mass to multiple plausible next tokens
        
        2. Mathematical Properties:
            - Minimizing cross-entropy is equivalent to maximizing the likelihood of the training data
            - It's the natural loss function derived from maximum likelihood estimation
            - It has beneficial gradients that become larger for more confident incorrect predictions
        
        3. Handling Multiple Valid Outputs:
            - Label smoothing (as used in this implementation) explicitly acknowledges multiple valid continuations
            - Instead of forcing 100% probability on one "correct" token, it reserves small probability for alternatives
            - This prevents the model from becoming overconfident and improves generalization
        
        4. Vocabulary as a Discrete Space:
            - Language modeling treats text generation as classification over a discrete vocabulary
            - Cross-entropy is the standard loss for multi-class classification problems
            - Alternative losses like MSE are inappropriate for discrete classification tasks
        
        5. Information Theory Connection:
            - Cross-entropy connects to information theory concepts of entropy and KL-divergence
            - It measures how many bits would be needed to encode the true distribution using the predicted distribution
            - This aligns with the goal of language modeling: finding the most probable continuation
        
        6. Practical Effectiveness:
            - Empirically, cross-entropy has proven highly effective for language modeling
            - It works well with teacher forcing (using ground truth as input during training)
            - It handles the sparse nature of language (most tokens are incorrect at any position)
        
        Alternative approaches like reinforcement learning or minimum risk training are sometimes used as 
        additional fine-tuning steps, but cross-entropy remains the foundation for basic language model training.
        
        Implementation Note:
        -------------------
        The cross-entropy loss combines both the softmax function and negative log-likelihood in one operation.
        It's numerically more stable than applying softmax and then calculating loss separately.
        
        Training Objective:
        ------------------
        The objective is to minimize this loss, which represents how well the model predicts
        the next token at each position given the source sequence and previous target tokens.
        This matches the autoregressive nature of sequence generation during inference.
    """
    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    device = torch.device(device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocal_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    preload = config['preload']

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No model preloaded, starting from scratch")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoc in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoc:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input '].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({
                "loss" : f"{loss.item():6.3f}"
            })

            writter.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step = global_step + 1

    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    model_filename = get_weights_file_path(config, f"{epoch:02d}")

    torch.save(
       {
        "epoch" : epoch, 
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "global_step" : global_step
       }, model_filename

    )

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
