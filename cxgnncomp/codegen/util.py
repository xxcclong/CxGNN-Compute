import triton
import torch


def compare(output1, output2):
    assert output1.shape == output2.shape, f"{output1.shape} != {output2.shape}"
    if torch.allclose(output1, output2, atol=1e-2, rtol=1e-2):
        print("CORRECT: match")
    else:
        print("ERROR: differ\n==========")
        mask = ~torch.isclose(output1, output2, atol=1e-2, rtol=1e-2)
        # print(torch.where(mask))
        # print(output1[mask][:100], output2[mask][:100])
        print(
            f"ERROR: differ ratio: {torch.sum(mask) / torch.numel(mask)}\n=========="
        )
        # exit()


def prof(task_name, method, func, display=True):
    output = triton.testing.do_bench(func)
    output2 = []
    for item in output:
        output2.append("{:.4f}".format(item))
    if display:
        print(f"{task_name} {method}:", output2)
    return output
