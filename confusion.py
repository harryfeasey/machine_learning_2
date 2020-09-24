from collections import namedtuple

class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        numbers = [self.true_positive, self.false_negative,
                   self.false_positive, self.true_negative]
        max_len = str(max(len(str(i)) for i in numbers))
        return (("{:>" + max_len + "} {:>" + max_len + "}\n" +
                 "{:>" + max_len + "} {:>" + max_len + "}").format(*numbers))


def confusion_matrix(classifier, dataset):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for xs, y in dataset:
        actual = classifier(xs)
        if actual:
            if actual == y:
                tp += 1
            else:
                fp += 1
        else:
            if actual == y:
                tn += 1
            else:
                fn += 1

    return ConfusionMatrix(true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn)

dataset = [
    ((0.8, 0.2), 1),
    ((0.4, 0.3), 1),
    ((0.1, 0.35), 0),
]
print(confusion_matrix(lambda x: 1, dataset))
print()
print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, dataset))
