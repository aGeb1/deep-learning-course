# Multi-Layer Perceptron Assignment

Knowing that the way the data is classified is periodic relative to x and y, I
decided to use a sinusoidal output activation function. This created issues
since the range of sin is from -1 to 1, while the range should be between 0
and 1. I tried sin^2, sin^4, and sin^6, and found sin^4 to be the best. I also
tested the sigmoid function since it is a common activation function, but it
gave me worse results.

I used ReLU as the hidden layer activation function since it is a ubiquitous
choice. I got similar quality outputs using Leaky ReLU, but decided to go with
ReLU since it is more standard. GELU gave slightly worse results than ReLU, but
this may have been due to the fact that I already adjusted other aspects of the
model to work well with ReLU. Sigmoid produced poor results where the plotted
contour was just a straight line through the data. This could have been due to
vanishing gradient, but I'm not sure. I decided to see if sin worked as a
hidden layer activation function, considering it worked well as an output
activation function, and the results were decent but not as good as ReLU.

The results generally get better as the hidden layers become wider, with the
widest hidden layer I tested having 60 neurons. This made sense to me, at least
in the context of the first hidden layer, as I could imagine a straight line
being drawn inside the gaps between the spirals for each neuron in the hidden
layer, with each additional line providing a better approximation of the
spiral. I chose a width of 16 because it was the smallest value that gave
strong results, and I figured that a smaller width indicated a simpler model
that was more reasonably fit for the simplicity of the data. I found that four
to six hidden layers had the best results - any less would make the model too
simple to approximate a spiral, but any more began to show signs of
overfitting.

I ended up using relatively small values of lambda for regularization. The
datapoints from the two groups generally did not overlap, so there was less
risk the model overfiting in an attempt to account for edge cases.

Something I found interesting about my results is how the spiral in the contour
continues after the point where the data stops being generated. I found this
interesting because I didn't think a perceptron had any aspects that could help
with classifying data significantly different from data in the training set.

Considering how this model produced relatively poor results for a very simple
task relative to the abilities of a person, I am confused how other neural
networks produce much better results than humans for far more complicated
tasks. Even though other models have far greater training times, increasing the
computation time for this model only resulted in overfitting. I imagine other
models are able to do better because they have more data, use different model
architectures, and employ a variety of methods besides neural networks better
suited for specific tasks.