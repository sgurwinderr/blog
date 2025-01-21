---
layout: post
title:  "Quantization and Dequantization in PyTorch: A Technical Overview"
author: Gurwinder
categories: [ AI ]
image: assets/images/vision-transformer-1.webp
featured: false
hidden: false
---

### Example Tensor  

Given weights tensor (2 channels):  
\[
\text{Weights} = 
\begin{bmatrix} 
1.0 & 2.0 & 3.0 \\ 
4.0 & 5.0 & 6.0 
\end{bmatrix}
\]  
Shape: \([2, 3]\)  
Integer range: \([-128, 127]\)  

---

### Per-Tensor Quantization  

#### Step 1: Compute Global Min/Max  
- Global Min: \(x_{\text{min}} = 1.0\)  
- Global Max: \(x_{\text{max}} = 6.0\)  

#### Step 2: Compute Scale and Zero Point  
$$  
\text{scale} = \frac{x_{\text{max}} - x_{\text{min}}}{q_{\text{max}} - q_{\text{min}}} = \frac{6.0 - 1.0}{255} = 0.01961  
$$  

$$  
\text{zero\_point} = \text{round}(-128 - \frac{x_{\text{min}}}{\text{scale}}) = \text{round}(-128 - \frac{1.0}{0.01961}) = \text{round}(-179.99) = -180  
$$  

#### Step 3: Quantize  
Apply the quantization formula:  
$$  
q = \text{round}\left(\frac{x}{\text{scale}} + \text{zero\_point}\right)  
$$  

Quantized values:  
$$  
\begin{aligned}  
q(1.0) &= \text{round}\left(\frac{1.0}{0.01961} - 180\right) = \text{round}(0) = 0 \\  
q(2.0) &= \text{round}\left(\frac{2.0}{0.01961} - 180\right) = \text{round}(51.02) = 51 \\  
q(3.0) &= \text{round}\left(\frac{3.0}{0.01961} - 180\right) = \text{round}(102.04) = 102 \\  
q(4.0) &= \text{round}\left(\frac{4.0}{0.01961} - 180\right) = \text{round}(153.06) = 153 \\  
q(5.0) &= \text{round}\left(\frac{5.0}{0.01961} - 180\right) = \text{round}(204.08) = 204 \\  
q(6.0) &= \text{round}\left(\frac{6.0}{0.01961} - 180\right) = \text{round}(255.0) = 255  
\end{aligned}  
$$  

#### Step 4: Dequantize  
$$  
\hat{x} = \text{scale} \cdot (q - \text{zero\_point})  
$$  

Dequantized values:  
$$  
\begin{aligned}  
\hat{x}(0) &= 0.01961 \cdot (0 - (-180)) = 1.0 \\  
\hat{x}(51) &= 0.01961 \cdot (51 - (-180)) = 2.0 \\  
\hat{x}(102) &= 0.01961 \cdot (102 - (-180)) = 3.0 \\  
\hat{x}(153) &= 0.01961 \cdot (153 - (-180)) = 4.0 \\  
\hat{x}(204) &= 0.01961 \cdot (204 - (-180)) = 5.0 \\  
\hat{x}(255) &= 0.01961 \cdot (255 - (-180)) = 6.0  
\end{aligned}  
$$  

#### Quantization Error  
Quantization error is:  
$$  
\text{Error} = x - \hat{x}  
$$  
For per-tensor quantization in this case, the error is **0 for all values**, as the range is perfectly covered.  

---

### Per-Channel Quantization  

#### Step 1: Compute Min/Max for Each Channel  
- Channel 1: \([1.0, 2.0, 3.0]\), Min = \(1.0\), Max = \(3.0\)  
- Channel 2: \([4.0, 5.0, 6.0]\), Min = \(4.0\), Max = \(6.0\)  

#### Step 2: Compute Scale and Zero Point for Each Channel  

**Channel 1**:  
$$  
\text{scale}_1 = \frac{3.0 - 1.0}{255} = 0.007843  
$$  
$$  
\text{zero\_point}_1 = \text{round}(-128 - \frac{1.0}{0.007843}) = \text{round}(-128 - 127.5) = -255  
$$  

**Channel 2**:  
$$  
\text{scale}_2 = \frac{6.0 - 4.0}{255} = 0.007843  
$$  
$$  
\text{zero\_point}_2 = \text{round}(-128 - \frac{4.0}{0.007843}) = \text{round}(-128 - 510) = -638  
$$  

#### Step 3: Quantize and Dequantize  

**Channel 1** Quantized Values:  
$$  
q(1.0) = \text{round}\left(\frac{1.0}{0.007843} + 255\right) = 127 \\  
q(2.0) = \text{round}\left(\frac{2.0}{0.007843} + 255\right) = 255 \\  
q(3.0) = \text{round}\left(\frac{3.0}{0.007843} + 255\right) = 383  
$$  

**Channel 1** Dequantized Values:  
$$  
\hat{x}(127) = 0.007843 \cdot (127 - 255) = 1.0 \\  
\hat{x}(255) = 0.007843 \cdot (255 - 255) = 2.0 \\  
\hat{x}(383) = 0.007843 \cdot (383 - 255) = 3.0  
$$  

**Channel 2** Quantized Values:  
$$  
q(4.0) = \text{round}\left(\frac{4.0}{0.007843} + 638\right) = 510 \\  
q(5.0) = \text{round}\left(\frac{5.0}{0.007843} + 638\right) = 766 \\  
q(6.0) = \text{round}\left(\frac{6.0}{0.007843} + 638\right) = 1022  
$$  

**Channel 2** Dequantized Values:  
$$  
\hat{x}(510) = 0.007843 \cdot (510 - 638) = 4.0 \\  
\hat{x}(766) = 0.007843 \cdot (766 - 638) = 5.0 \\  
\hat{x}(1022) = 0.007843 \cdot (1022 - 638) = 6.0  
$$  

#### Quantization Error  
In this case, the error is **0 for all values**, as the quantization was exact for each channel.  

---

### Comparison  

| Metric                     | Per-Tensor Quantization | Per-Channel Quantization |  
|----------------------------|-------------------------|--------------------------|  
| **Scale**                  | 0.01961                | [0.007843, 0.007843]     |  
| **Zero Point**             | -180                   | [-255, -638]            |  
| **Quantization Error**     | 0 for all values       | 0 for all values        |  

In this example, both methods resulted in no error due to perfect alignment of the tensor values with quantization levels. However, in practice:  
- **Per-tensor** can lead to larger errors when dynamic ranges vary significantly across channels.  
- **Per-channel** reduces errors for channels with different ranges, particularly in deep learning models.  

--- 

This version is ready for Jekyll with proper LaTeX rendering. Let me know if you'd like further refinements!

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