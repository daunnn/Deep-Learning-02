# Deep-Learning_HW3

This repository contains neural network for character-level language modeling.

-------
## Plot the average loss values for training and validation 
<div align="center">
  <img src="https://github.com/daunnn/Deeplearning-HW3/blob/main/results/charrnn_loss.png" width="500"/>
  <img src="https://github.com/daunnn/Deeplearning-HW3/blob/main/results/charlstm_loss.png" width="500"/>
</div>

The performance is evaluated based on the loss values obtained on the validation dataset over the course of 20 training epochs.

Best CharRNN Model: Epoch 3, Validation Loss: 1.2270878601699393
Best CharLSTM Model: Epoch 19, Validation Loss: 0.6159207755533248

### Analysis

The CharLSTM achieves lower loss values overall compared to the CharRNN. This indicates that the CharLSTM is able to model the character dependencies and generate text more effectively.
The best performance for the CharRNN is obtained at epoch 3, with a validation loss of 1.2271. In contrast, the CharLSTM's best performance is at epoch 19, with a significantly lower validation loss of 0.6159. This suggests that the CharLSTM benefits from longer training and is able to capture more complex patterns in the text.

The training loss for both models decreases over the epochs, indicating that they are learning from the training data. 
However, the validation loss for the CharRNN starts to increase after epoch 3, implying overfitting. The CharLSTM's validation loss continues to decrease until around epoch 19, demonstrating better generalization.

The loss curves for the CharLSTM are smoother and more stable compared to the CharRNN. This suggests that the CharLSTM is less prone to instability and is able to learn more robust representations of the character sequences.
In conclusion, the **CharLSTM** outperforms the CharRNN in terms of language generation performance, as evidenced by the lower validation loss values. 


-----

## Summary


- Reference
  - https://karpathy.github.io/2015/05/21/rnn-effectiveness/






