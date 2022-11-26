from itertools import product

def generator(test_dict):
    for key in product(*test_dict.values()):
        yield {k:v for k, v in zip(test_dict.keys(), key)}