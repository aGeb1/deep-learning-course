from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("ag_news")
classifier = pipeline(
    "zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1"
)
candidate_labels = ["world", "sports", "business", "sci/tech"]


def category_to_label_number(s):
    if s == "world":
        return 0
    if s == "sports":
        return 1
    if s == "business":
        return 2
    if s == "sci/tech":
        return 3


size = len(dataset["test"])
correct = 0

for i in range(size):
    classifier_output = classifier(dataset["test"]["text"][i], candidate_labels)
    pred = category_to_label_number(classifier_output["labels"][0])
    if pred == dataset["test"]["label"][i]:
        correct += 1

correct /= size
print(f"Accuracy: {(100*correct):>0.1f}%")
