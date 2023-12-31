# Residual Network Assignment

For both CIFAR-10 and CIFAR-100 I attempted to implement the wide residual
networks in the article by Zagoruyko and Komodakis [1]. I used two residual
layers per block, a widening factor of six, thirty percent dropout within the
layers, and the same limited data augmentation in the article. However, I used
an Adam optimizer and ran for 20 epochs, while the article uses SGD with
momentum and ran for 200 epochs.

My results were significantly worse than those in the paper in comparison to a
sixteen layer model with a widening factor of eight from the article. My top-1
accuracy on CIFAR-10 was 72.49 percent compared to 95.19 percent accuracy in
the paper. My top-1 and top-5 accuracies respectively on CIFAR-100 were 40.12
and 68.86 percent compared to 77.93 percent for top-1 accuracy in the paper.

The difference in epoch count significantly limited the performance of my
model, but the models in the article still had loss which decreased
significantly faster than my model. In reading about optimization methods in
the super-convergence article by Smith and Topin [2], it became apparent that
optimizers with adaptive learning rates must use vastly different
hyperparameters than simpler stochastic gradient descent models, even with
regards to parameters related to regularization. I imagine my use of the Adam
optimizer without proper optimizers was the most significant issue with my
setup.

[1] https://arxiv.org/abs/1605.07146
[2] https://arxiv.org/abs/1708.07120