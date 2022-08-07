import numpy as np

def expand_dims(input, conv_domain):
    r = int((conv_domain-1)/2)
    l = input.shape[0]
    n_input_vars = input.shape[1]
    output = np.zeros((l, conv_domain, n_input_vars))
    for i in range(l):
        for j in range(conv_domain):
            for k in range(n_input_vars):
                output[i,j,k] = input[min(i+j-r,l-1),k]
    return output