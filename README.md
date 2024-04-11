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

Each mechanism is tested on 