# AG News Classification Assignment

I intended to get results comparable to the state of the art models listed in
the article by Zhang, Zhao, and LeCun [1] on the AG News classification task.
Accuracies on the task ranged from 83.09 to 92.36 for classification task. To
determine a specific target, I used a zero shot classifier [2] and got an
accuracy of 89.7 percent.

To reach the 89.7 percent accuracy target, I trained a linear module on the
output of a sentence embedding model [3] and got an accuracy of 88.9 percent.
I found that relatively high weight decay and a large batch size helped
improved accuracy, but increasing the number of iterations and adding learning
rate decay did not.


[1] https://arxiv.org/abs/1509.01626
[2] https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1
[3] https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2