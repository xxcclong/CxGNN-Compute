import triton
from ..util import log


def compare(output1, output2):
    if triton.testing.allclose(output1, output2):
        print("CORRECT: Triton and Torch match")
    else:
        print("ERROR: Triton and Torch differ")
        print(output1[0], output2[0])
        exit()


def prof(task_name, method, func):
    output = triton.testing.do_bench(func)
    output2 = []
    for item in output:
        output2.append("{:.4f}".format(item))
    print(f"{task_name} {method}:", output2)
    return output
