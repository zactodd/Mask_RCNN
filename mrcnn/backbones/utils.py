import math
import itertools


# TODO improve naming
def block_names(n):
    letters = math.ceil(math.log(n, 26))
    names = []
    for i, nums in enumerate(itertools.product(range(26), repeat=letters)):
        if i > n - 1:
            break
        else:
            names.append("".join(chr(num + 97) for num in nums))
    return names
