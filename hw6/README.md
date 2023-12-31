# Transformer Implementation

This project is about creating a transformer model. The first requirement of this assignment is to create a multi-head attention module and a transformer block layer, the fundamental components of a transformer model. The second requirement is to implement tests that provide sufficient evidence of the transformer's functionality.

The multi-head attention module is influenced from the code presented in [1] and [2]. I was especially reliant on the code from [1] for implementing causal masking, which I found particularly difficult to implement using the TensorFlow library. I used a standard ReLU based feed-forward network in my transformer blocks, but I also implemented the SwiGLU layer from [3] that I could easily replace the ReLU feed-forward network with. I then made a decoder-only transformer which could be used as the basis of a GPT model.

For each component of the model, I implemented a trainability and dimensionality test to make sure everything was working as expected. For the multi-head attention and transformer block, I tested for trainability and dimensionality both when the modules were masked and when they were not.

I then found the Jacobian matrix with the input and output of a multi-head attention module using one head, effectively acting as single-head attention, when causal masking was enabled. This resulted in a lower triangular matrix, indicating that the module was causal as expected.

To show that the decoder-only model could be used in a real application like a large language model, I demonstrated the model's ability to overfit very simple training data. I put one sequence of 64-dimensional trainable embeddings into the model, which could theoretically represent tokens of text. The output of the model is compared to an identity matrix using cross-entropy softmax to get the loss value to minimize. The columns of the identity matrix can be thought of as the one-hot vector corresponding to the output token that the input token is supposed to be transformed into.

I started off testing the model's ability to overfit an input with 512 embeddings within 1000 iterations using AdamW. One issue is that the model is able to overfit the data without any transformer blocks, since there would still be trainable embeddings and a linear layer. However, the presence of transformer blocks decreases the loss of the model, indicating that the transformer blocks are contributing to the model's ability to overfit. With no transformer blocks the final loss is 1.42e-2, with one transformer block the loss goes down to 2.02e-4, with two transformer blocks the loss goes down to 1.79e-4, and with three transformer blocks the loss goes down to 1.77e-4. Adding more transformer blocks after this point increases the final loss, most likely due to the fact that the model becomes too large to train in 1000 iterations.

I then observed how well the two transformer block model performed as I increased the length of the input. With 1024 embeddings the loss increases to 1.99e-4, and with 2048 embeddings the loss increases to 2.96. I'm not really sure why the loss increases so rapidly when going up to an input length of 2048. I figured the issue may have been related to the difference between the embedding size and the number of possible tokens, limiting the ability of the linear layer to predict the output token. However, the loss goes down to 1.79e-2 with no transformer blocks and 4.91e-4 with one transformer block, so that doesn't appear to be the issue. It doesn't seem the issue has to do with the transformer block itself, since all the other overfitting tests worked, so I don't have a good guess of what the issue is.

The transformer model is capable of being causal and overfitting a small dataset, indicating that it was effectively implemented.

[1] https://github.com/Kyubyong/transformer/
[2] https://github.com/mistralai/mistral-src
[3] https://arxiv.org/abs/2002.05202