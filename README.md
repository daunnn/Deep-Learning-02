# Deep-Learning_HW3

This repository contains a neural network for character-level language modeling.

## Code Structure

- `dataset.py`: Handles data processing and provides data to the model.
- `model.py`: Implements the vanilla RNN and LSTM models.
- `main.py`: Trains the models and monitors the training process using average loss values.
- `generate.py`: Generates characters using the trained models.

The generated results can be found in the `results` folder.

-------
## Plot the average loss values for training and validation 
<div align="center">
  <img src="https://github.com/daunnn/Deeplearning-HW3/blob/main/results/charrnn_loss.png" width="500"/>
  <img src="https://github.com/daunnn/Deeplearning-HW3/blob/main/results/charlstm_loss.png" width="500"/>
</div>

The performance is evaluated based on the loss values obtained on the validation dataset over the course of 20 training epochs.

Best CharRNN Model: Epoch 3, Validation Loss: 1.227

Best CharLSTM Model: Epoch 19, Validation Loss: 0.616

### Analysis

The CharLSTM achieves lower loss values overall compared to the CharRNN. This indicates that the CharLSTM is able to model the character dependencies and generate text more effectively.
The best performance for the CharRNN is obtained at epoch 3, with a validation loss of 1.2271. In contrast, the CharLSTM's best performance is at epoch 19, with a significantly lower validation loss of 0.6159. This suggests that the CharLSTM benefits from longer training and is able to capture more complex patterns in the text.

The training loss for both models decreases over the epochs, indicating that they are learning from the training data. 
However, the validation loss for the CharRNN starts to increase after epoch 3, implying overfitting. The CharLSTM's validation loss continues to decrease until around epoch 19, demonstrating better generalization.

The loss curves for the CharLSTM are smoother and more stable compared to the CharRNN. This suggests that the CharLSTM is less prone to instability and is able to learn more robust representations of the character sequences.
In conclusion, the **CharLSTM** outperforms the CharRNN in terms of language generation performance, as evidenced by the lower validation loss values. 


-----

### Generated Text Samples


- [CharLSTM Generated Texts](results/charlstm_generated_texts.txt)

```
**Seed Character: "Apple"
Temperature: 0.5**


Apple and thus fear
As frees before them,
And by thy ordination,
Must I cannot have them, and guilted so madam.

MENENENIUS:
No, to go.

VOLUMNIA:
Your than
```

```
**Temperature: 1.0**

Apple and thund the ladyship? what you have not
Of their toward. O, away?

CATFitions, and old device
On apber.

COMINIUS:
If you hear it out.

First may p
```


```
**Temperature: 1.5**


Apple are and nyel togeth. oo hosence to be ob strokes the us.
QUEEN ELIZABETH:
Then Anthus have passieve good me be
hons too of him: every thigh go.
Meth
```

```
**Seed Character: "University"
Temperature: 0.5
**

University close to get thee a monster of the ocent shall be you, if you do hold
therein show
range almost match it in my mind
He dreamt to the Capitol; I'll ta
```
```
**Temperature: 1.0**
University elor,
I cannot now thy hellected so far my son--
Thy life and purchase with all his thought upon 't; and they smart
To yield to him again.
I'll wine
```

```
**Temperature: 1.5**
University of the Volsces, to
sweet laid, I am barking
A place and leage
Most warlick, hear I know the canon an heart men thou stiff?
CORIOLANUS:
Cut it soject
```

Seed Character: "Seoul"
Temperature: 0.5
Seould no less to the people;
And by thy oratory farewell was forth
To tearing barrent cut one accuse, and less himsely not
How thy heigh.

First Stand thu
Temperature: 1.0
Seould no lectus on ants he dismings; and all out off, to ruin
Great complies nor falses from the held; and given
an our true them, and devies death.

QUEE
Temperature: 1.5
Seoul true accuse ourself
again.

HASTINGLEM:
Here go on:
Then shall high, am not sebjects:
Your lady; from sore young a very he
him one,
But by you show t
Seed Character: "Coffee"
Temperature: 0.5
Coffeed to the violence dins unto the other; but, as I do?

COMINIUS:
The best unto the other wife's worthy master on him fear'd by then as eas.

MENENIUS:
Temperature: 1.0
Coffeed to the Marcius;
I fear 'shall'd it
Intest of death, and before it were,
That will hear in great other, but it; and affections;
Or love.

Second Lord
Temperature: 1.5
Coffeed to the violence
Suckn hereto me
And himself a min'd out of
ingly returns
Of Phoein, worthing!
What, down to this,
Have taken of heaven furle is true
Seed Character: "Computer"
Temperature: 0.5
Computer speak, and go
Ine eyes now thy hein,
I must have redious supplies with spite one so many, I envy to get you for you all.

CORIOLANUS:
Saw I fear the
Temperature: 1.0
Computer sorrorn to seek him how second Lord:
His aught company my brother in the corse; indeed, report for my wife's in health! To him!

MENENIUS:
That Juliu
Temperature: 1.5
Computer to o' the comfort,
Untain Clarence,
That Marcius is come?

Againscould move
thee, gaven hours
At Pomfret, no doubt
Which o'erments,
And pardon blood

Upon examining the generated text samples from the CharLSTM model (Best Model: Epoch 19), we can observe the following:

### Report: Comparing Generated Text Samples

Upon examining the generated text samples, we can observe the following:

Stylistic Consistency: The generated text exhibits stylistic consistency with the training data, which in this case is Shakespeare's plays. The model captures the language patterns, vocabulary, and sentence structures commonly found in Shakespearean text, such as the use of archaic words, poetic expressions, and character names.

Coherence and Relevance: The generated text maintains a good level of coherence and relevance to the seed character, even at higher temperature values. 

Temperature Impact: The temperature value has a significant impact on the diversity and creativity of the generated text. At lower temperatures (e.g., 0.5), the model generates text that closely resembles the training data, with more predictable and conservative outputs. As the temperature increases (e.g., 1.0, 1.5), the generated text becomes more diverse, introducing novel word combinations, phrases, and ideas that deviate from the original text while still maintaining coherence.

Imaginative and Creative Output: At higher temperature values, the model generates highly imaginative and creative text. It combines words and phrases in unique ways, creating vivid and poetic expressions that evoke a sense of depth and emotion. 


## Summary

This project demonstrates the implementation and comparison of vanilla RNN and LSTM models for character-level language modeling. The CharLSTM model achieves better performance in terms of validation loss and generates more coherent and contextually relevant text compared to the CharRNN model.

## Reference

- https://karpathy.github.io/2015/05/21/rnn-effectiveness/





