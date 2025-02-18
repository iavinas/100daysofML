DAY 2

** torchmetrics.CharErrorRate() **

Computes the Character Error Rate between predicted and target texts.

Formula: CER = (S + D + I) / N
- S: Number of character substitutions
- D: Number of character deletions
- I: Number of character insertions
- N: Total number of characters in reference

Example:
    >>> cer = torchmetrics.CharErrorRate()
    >>> pred = ["helllo world"]
    >>> target = ["hello world"]
    >>> cer(pred, target)  # Returns: 0.1 (1 insertion / 10 chars)

https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate.html

<br>

** torchmetrics.WordErrorRate() **

Computes the Word Error Rate between predicted and target texts.

Formula: WER = (S + D + I) / N
- S: Number of word substitutions
- D: Number of word deletions
- I: Number of word insertions
- N: Total number of words in reference

Example:
    >>> wer = torchmetrics.WordErrorRate()
    >>> pred = ["the quick brown fox jumps"]
    >>> target = ["the fast brown fox jumped"]
    >>> wer(pred, target)  # Returns: 0.4 (2 substitutions / 5 words)


Example:
    >>> from torchmetrics.text import WordErrorRate 
    >>> preds = ["there is an other sample"] 
    >>> target = ["there is another one"] 
    >>> wer = WordErrorRate() 
    >>> wer(preds, target) 
    tensor(0.7500)

**<contemplator>**

Let me break down how WordErrorRate is calculated for this specific example...

The input strings are:
- Prediction: "there is an other sample"
- Target: "there is another one"

First, let's split into words:
- Prediction: ["there", "is", "an", "other", "sample"]
- Target: ["there", "is", "another", "one"]

Now, let me count the words:
- Prediction: 5 words
- Target: 4 words

Let me identify the operations needed to transform prediction into target:
1. "there" - match (no operation)
2. "is" - match (no operation)
3. "an other" needs to become "another"
   - Delete "an"
   - Delete "other"
   - Insert "another"
4. "sample" needs to become "one"
   - Substitute "sample" with "one"

Let's count the operations:
- Substitutions (S) = 1 ("sample" → "one")
- Deletions (D) = 2 ("an" and "other" are deleted)
- Insertions (I) = 1 ("another" is inserted)

The formula is:
```
WER = (S + D + I) / N
Where N is the number of words in the target
```

Plugging in our numbers:
```
WER = (1 + 2 + 1) / 4
    = 4 / 4
    = 0.75
```

This is why we get:
```python
tensor(0.7500)
```

</contemplator>

**<final_answer>**

The Word Error Rate of 0.75 (75%) for the given example is calculated as follows:

1. Total Operations Needed:
   - Substitutions: 1 ("sample" → "one")
   - Deletions: 2 ("an" and "other")
   - Insertions: 1 ("another")
   - Total Operations: 4

2. Reference Length:
   - Target sentence has 4 words

3. Final Calculation:
   - WER = (S + D + I) / N
   - WER = (1 + 2 + 1) / 4
   - WER = 4 / 4
   - WER = 0.75 or 75%

The high error rate (0.75) indicates significant differences between the predicted and target sequences, which is accurate given that only the first two words match exactly.


https://lightning.ai/docs/torchmetrics/stable/text/word_error_rate.html


"""
torchmetrics.BLEUScore()

Computes the BLEU (Bilingual Evaluation Understudy) score for machine translation evaluation.

Key Components:
1. N-gram Precision:
   - Counts matching n-grams between prediction and reference
   - Usually uses n=1,2,3,4 (unigram to 4-gram)
   - Each n-gram level gets a precision score

2. Brevity Penalty:
   - Penalizes translations that are too short
   - BP = min(1, exp(1 - reference_length/prediction_length))

3. Final Score:
   - Geometric mean of n-gram precisions
   - Multiplied by brevity penalty
   - Range: 0 to 1 (higher is better)

Usage:
    >>> metric = torchmetrics.BLEUScore()
    >>> predictions = ["the cat sat on the mat"]
    >>> references = ["the cat sits on the mat"]
    >>> score = metric(predictions, references)

Options:
    - weights: Weights for different n-gram levels
    - smooth: Apply smoothing for better scores with short sequences
    - n_gram: Maximum n-gram length to consider

Advantages:
    - Correlates with human judgment
    - Language-independent
    - Supports multiple references
    - Fast computation
    - Industry standard for MT evaluation
"""
