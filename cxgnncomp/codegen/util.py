import triton


def compare(output1, output2):
    if triton.testing.allclose(output1, output2):
        print("CORRECT: Triton and Torch match")
    else:
        print("ERROR: Triton and Torch differ")
        print(output1[0], output2[0])
        exit()


def prof(task_name, method, func):
    print(f"{task_name} {method}:", triton.testing.do_bench(func))
