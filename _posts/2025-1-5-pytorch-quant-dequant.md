---
layout: post
title:  "Quantization and Dequantization in PyTorch: A Technical Overview"
author: Gurwinder
categories: [ AI ]
image: assets/images/vision-transformer-1.webp
featured: false
hidden: false
---

### Quantization and Dequantization in PyTorch: A Technical Overview

Quantization and dequantization are essential techniques in deep learning to optimize models for resource-constrained environments like mobile devices and edge computing platforms. These methods enable efficient inference by reducing model size and accelerating computations.

---

### 1. **Quantization Overview**

Quantization refers to mapping high-precision floating-point numbers (e.g., `float32`) into lower-precision formats such as integers (`int8`) or fixed-point numbers. This is done to minimize computational overhead and memory usage while maintaining acceptable accuracy.

In PyTorch, quantization can be implemented at various stages:
- **Static Quantization**: Applied after training using calibration.
- **Dynamic Quantization**: Quantization parameters are computed during inference.
- **Quantization-Aware Training (QAT)**: Simulates quantization during training for higher accuracy.

---

The **zero point** is a key parameter in quantization that helps map the range of floating-point numbers to integers in a way that minimizes the quantization error. It essentially adjusts the integer representation so that the original floating-point range aligns correctly with the integer range.

### Why is Zero Point Needed?
When quantizing, we map floating-point values (\(x\)) to integer values (\(q\)) using a scale factor (\(\text{scale}\)):

\[
q = \text{round}\left(\frac{x}{\text{scale}} + \text{zero\_point}\right)
\]

The **zero point** ensures that the quantized range properly represents the dynamic range of the original floating-point numbers, especially when the range doesn't start at zero. It acts as an offset, allowing negative or positive floating-point values to be correctly mapped into the integer domain.

### Definition of Zero Point
For a given floating-point range \([x_{\text{min}}, x_{\text{max}}]\) and an integer range \([q_{\text{min}}, q_{\text{max}}]\), the **zero point** is calculated as:

\[
\text{zero\_point} = \text{round}\left(q_{\text{min}} - \frac{x_{\text{min}}}{\text{scale}}\right)
\]

Where:
- \(x_{\text{min}}\): Minimum floating-point value.
- \(x_{\text{max}}\): Maximum floating-point value.
- \(q_{\text{min}}\): Minimum integer value (e.g., -128 for int8).
- \(q_{\text{max}}\): Maximum integer value (e.g., 127 for int8).
- \(\text{scale}\): The step size between representable quantized values.

### Intuition Behind Zero Point
- If \(x_{\text{min}}\) is 0, then \(\text{zero\_point}\) will align the floating-point value 0 to an integer value within the integer range.
- If \(x_{\text{min}}\) is not 0, the zero point ensures that the integer range can still represent values around the floating-point zero.

### Example of Zero Point Calculation
Let’s consider a floating-point range \([0.5, 2.0]\) and an integer range \([-128, 127]\):
1. Calculate the scale:
   \[
   \text{scale} = \frac{x_{\text{max}} - x_{\text{min}}}{q_{\text{max}} - q_{\text{min}}} = \frac{2.0 - 0.5}{127 - (-128)} = 0.005882
   \]
2. Compute the zero point:
   \[
   \text{zero\_point} = \text{round}(-128 - \frac{0.5}{0.005882}) = \text{round}(-213)
   \]

Here, the zero point shifts the floating-point range \([0.5, 2.0]\) into the integer range \([-128, 127]\) while maintaining alignment.

### Role in Dequantization
The zero point is also used when converting quantized values (\(q\)) back to floating-point approximations (\(x'\)):

\[
x' = (q - \text{zero\_point}) \times \text{scale}
\]

This ensures that the dequantized values are properly aligned to the original floating-point range.

### Summary of Zero Point's Function
- Aligns the integer representation with the floating-point range.
- Allows for asymmetric quantization, where the floating-point range does not have to be symmetric around zero.
- Ensures minimal quantization error during both quantization and dequantization.

In practice, **PyTorch's quantization tools** automatically compute the zero point, making it straightforward to use in model compression workflows.

### 2. **Quantization Workflow in PyTorch**

The quantization workflow typically involves three steps:
1. **Prepare**: Modify the model to be ready for quantization by adding quantization-specific modules.
2. **Calibrate**: Use representative input data to compute quantization parameters.
3. **Convert**: Replace floating-point modules with quantized versions.

---

### 3. **Key Concepts in PyTorch Quantization**

#### **Quantization Formula**

Quantization maps a floating-point value \( x \) to an integer \( q \):

\[
q = \text{round}\left(\frac{x - \text{zero\_point}}{\text{scale}}\right)
\]

- **Scale**: Determines the step size between representable values.
- **Zero Point**: Ensures that zero in floating-point maps to zero in quantized values.
- \( q \) lies in the range \([q_\text{min}, q_\text{max}]\).

#### **Dequantization Formula**

To recover the original value:

\[
x = q \cdot \text{scale} + \text{zero\_point}
\]

Dequantization is typically performed after computations to interpret quantized results.

---

### 4. **Quantization Types**

#### a. **Static Quantization**
- Weights and activations are quantized.
- Involves precomputing scale and zero-point values during calibration.
- Best for use cases with predictable inputs and stable computational patterns.

**Example in PyTorch**:
```python
import torch
import torch.quantization as quant

# Define a model
model = torch.nn.Sequential(
    torch.nn.Linear(4, 2),
    torch.nn.ReLU()
)

# Prepare for static quantization
model.qconfig = quant.get_default_qconfig('fbgemm')
prepared_model = quant.prepare(model)

# Calibrate with representative data
data = torch.randn(100, 4)
for sample in data:
    prepared_model(sample)

# Convert to quantized model
quantized_model = quant.convert(prepared_model)
```

---

#### b. **Dynamic Quantization**
- Only weights are quantized; activations remain in floating-point.
- Ideal for models dominated by linear layers, like transformers and RNNs.

**Example in PyTorch**:
```python
import torch

# Define a model
model = torch.nn.Linear(4, 2)

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

#### c. **Quantization-Aware Training (QAT)**
- Simulates quantized inference during training.
- Results in higher accuracy than static or dynamic quantization.

**Example in PyTorch**:
```python
import torch
import torch.quantization as quant

# Define a model
model = torch.nn.Sequential(
    torch.nn.Linear(4, 2),
    torch.nn.ReLU()
)

# Prepare for QAT
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
prepared_model = quant.prepare_qat(model)

# Train the model
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.01)
for epoch in range(10):
    optimizer.zero_grad()
    output = prepared_model(torch.randn(10, 4))
    loss = output.mean()
    loss.backward()
    optimizer.step()

# Convert to quantized model
quantized_model = quant.convert(prepared_model)
```

---

### 5. **Practical Considerations**

1. **Calibration Data**: The representativeness of calibration data directly impacts quantization accuracy.
2. **Hardware Support**: Quantized models must be deployed on hardware that supports integer arithmetic (e.g., ARM, NVIDIA, Intel).
3. **Accuracy vs. Efficiency**: Quantization may lead to accuracy loss. Techniques like QAT can mitigate this.

---

### 6. **Quantization with Custom Layers**

PyTorch allows custom handling of quantization for user-defined layers by overriding `quantize()` and `dequantize()` methods or providing custom observers.

**Example**:
```python
class CustomLayer(torch.nn.Module):
    def forward(self, x):
        return x * 2

# Define custom quantization observer
observer = torch.quantization.MinMaxObserver()
```

---

### Conclusion

Quantization in PyTorch is a powerful tool for optimizing models for deployment in real-world applications. By choosing the right quantization strategy (static, dynamic, or QAT) and leveraging PyTorch's comprehensive APIs, practitioners can balance the trade-off between efficiency and accuracy effectively.