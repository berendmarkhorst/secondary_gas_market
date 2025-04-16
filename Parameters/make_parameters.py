import numpy as np

input_file = "Data/OneScenario/SanityCheckBig.xlsx"

with open("Parameters/experiment2.txt", "w") as f:
    domain = np.linspace(0, 1, 21)
    for c1 in domain:
        for c2 in domain:
            if c2 <= c1:
            # if c1 == c2:
                output_file = "Results/Experiment2/c1_{:.2f}_c2_{:.2f}".format(c1, c2)
                print(f"python FodstadExample.py --c1 {c1} --c2 {c2} --input_file {input_file} --output_file {output_file} --equality_constraint 1", file=f)
                f.flush()

    # c1, c2 = None, None
    # output_file = "Results/Conservative_experiment/flexible".format(c1, c2)
    # print(f"python FodstadExample.py --c1 {c1} --c2 {c2} --input_file {input_file} --output_file {output_file}", file=f)
