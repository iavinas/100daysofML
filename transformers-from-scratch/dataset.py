import torch


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # both tokenizers have same id for all four special token, please see get_or_build_tokenizers in train.py for reference
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, idx):
        """
            Creates tokenized and padded sequences for encoder input, decoder input, and labels.

            For sequence-to-sequence models, we need three differently structured tensors:

            1. Encoder Input: The complete source sequence with boundary markers
            Structure: [SOS] + source_tokens + [EOS] + padding
            Padding calculation: seq_len - len(source_tokens) - 2
                                    (subtracting 2 for SOS and EOS tokens)

            2. Decoder Input: The target sequence shifted right (for teacher forcing)
            Structure: [SOS] + target_tokens + padding
            Padding calculation: seq_len - len(target_tokens) - 1
                                    (subtracting 1 for SOS token only)

            3. Labels: The target sequence shifted left (what to predict)
            Structure: target_tokens + [EOS] + padding
            Padding calculation: seq_len - len(target_tokens) - 1
                                    (subtracting 1 for EOS token only)
                                    
            This arrangement creates the teacher forcing mechanism where:
            - At position i, the decoder input contains the token at position i in the target
            - The corresponding label contains the token at position i+1
            - This trains the model to predict the next token given all previous tokens

            All sequences are padded to seq_len to enable efficient batching.
            If a sequence would exceed seq_len after adding special tokens, an error is raised.
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # sos, eos to be added so -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 # only sos to be added so -1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence os too long")
        
        # Add sos and eos
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,
                dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only sos
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only eos token

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )


        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input' : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqieeze(0).int(), # (1, 1, seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label" : label,
            "src_text" : src_text,
            "tgt_text" : tgt_text

        }






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