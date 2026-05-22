---
title: "Interactive Courses"
date: 2026-05-22T00:00:00Z
draft: false
---

# Interactive Courses

Learn complex technical concepts through beautiful, interactive single-page courses.

Each course transforms a codebase into an educational experience with:
- 📚 Scroll-based navigation through modules
- 💬 Animated conversations between components
- 🎯 Interactive quizzes and exercises
- 🔍 Plain-English code explanations
- 🎨 Beautiful visualizations

---

## Available Courses

### vLLM Optimization Deep Dive: Triton Kernels

Fast and efficient LLM inference is fundamentally about optimization—and that starts at the kernel level. This course takes you inside vLLM's Triton kernel implementations to show you exactly how to optimize attention, quantization, and memory patterns for production inference.

**[Launch Course →](/courses/vllm-triton-optimization/)**

**Duration:** 6 modules (2–3 hours)

**What You'll Learn:**
- How Triton kernels replace custom CUDA for faster inference
- Attention optimization patterns (fused kernels, memory-aware scheduling)
- Quantization kernels in vLLM (int8, fp8, mixed precision)
- Memory layout and bandwidth optimization
- Profiling and benchmarking inference workloads
- Real-world deployment patterns from production codebases

**Modules:**
1. Introduction to vLLM & Kernel Optimization
2. Fused Attention & Memory Efficiency
3. Quantization Kernels (Int8, FP8, Mixed Precision)
4. Paged Attention & Memory Management
5. Kernel Profiling & Performance Analysis
6. Production Optimization Patterns

**Prerequisites:**
- Basic understanding of CUDA / GPU programming
- Familiarity with PyTorch and model inference
- Interest in LLM systems

**Features:**
- 📖 Animated code walkthroughs (reading vLLM source directly)
- 🧮 Mathematical foundations for each optimization technique
- 🔬 Live profiler comparisons (optimized vs. naive implementations)
- 📊 Visualizations of memory access patterns
- 🎯 Interactive quizzes after each module

---

## How Courses Are Created

These courses are generated using the **codebase-to-course** methodology:

1. **Deep analysis** of a real codebase
2. **Curriculum design** (4-6 modules from beginner to advanced)
3. **Interactive elements** (animations, quizzes, diagrams)
4. **Plain-English translations** of technical code
5. **Self-contained HTML** (works offline, no backend needed)

Each course is a single HTML file in `/courses/` that runs entirely in your browser.

---

## Request a Course

Interested in a course on a specific topic? [Open an issue](https://github.com/sgurwinderr) or reach out on [LinkedIn](https://linkedin.com/in/sgurwinderr).
