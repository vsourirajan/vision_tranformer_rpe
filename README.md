# Examining Various Relative Positional Encoding (RPE) Mechanisms for Vision Transformers (ViTs)

### RPE Mechanisms Explored
- General Learnable Function: $f_\Theta : \mathbb{R} \rightarrow \mathbb{R}$
- Monotonically Decreasing Function: $f = e^{-\alpha x}$
- Ratio of two polynomial functions: $f = \frac{h}{g}$

### Subvariants
We explore two subvariants for each of the mechanisms above:
- Each attention head of a layer uses the same parameters of the corresponding RPE mechanism
- Separate heads use separate parameters of the corresponding RPE mechamism

We also provide a comparison to the RPE-free baseline, which can be seen in `absolute_pos_encoding_vit.py`

Each mechanism is tested on MNIST, CIFAR datasets

### How to Run:
Note: If the Metal Performance Shaders backend is available, the model with train on the MacOS GPU. Otherwise it will train on the cpu
- You can run the absolute positional encoding model with the following command: `python3 absolute_pos_encoding_vit.py {dataset name}`
    - The dataset name can be one of the following:
        - "MNIST"
        - "CIFAR10"
        - eg. `python3 absolute_pos_encoding_vit.py MNIST` will train the RPE-free baseline on the MNIST dataset
- You can run the relative positional encoding model with the folliwing command: `python3 relative_pos_encoding_vit.py {dataset name} {rpe type} {subvariant}`
    - The dataset name follows the same requirements as above for the RPE-free baseline
    - The rpe type can be one of the following:
        - "general": General Learnable RPE Function
        - "monotonic": Monotonically Decreasing RPE Function
        - "ratio": Ratio of Polynomials RPE Function
    - The subvariant can be either "1" (same set of RPE parameters across all attention heads) or "2" (different parameters for different attention heads)
    - eg. `python3 relative_pos_encoding_vit MNIST general 1` will run General Learnable RPE Subvariant 1 on MNIST
### Results:
- Abolute Positional Encoding:
    - MNIST: 0.820 Test accuracy
    - CIFAR10: 0.546 Test accuracy
- Subvariant 1: Same paramters across all heads:
    - MNIST:
        - General Learnable Function: 0.975 Test Accuracy
        - Monotonically Decreasing Function: 0.368 Test Accuracy
        - Ratio of Two Polynomial Functions: 0.978 Test Accuracy
    - CIFAR10:
        - General Learnable Function: 0.510, 0.516 Test Accuracy
        - Monotonically Decreasing Function: 0.210 Test Accuracy
        - Ratio of Two Polynomial Functions: 0.479 Test Accuracy
- Subvariant 2: Different Parameters for Different Heads
    - MNIST:
        - General Learnable Function: 0.973 Test Accuracy
        - Monotonically Decreasing Function: 0.199 Test Accuracy
        - Ratio of Two Polynomial Functions: 0.655 Test Accuracy
    - CIFAR10:
        - General Learnable Function: 0.547 Test Accuracy
        - Monotonically Decreasing Function: 0.438 Test Accuracy
        - Ratio of Two Polynomial Functions: 0.247 Test Accuracy
