# 📘 Forward–Momentum Scaling Law (AAAI 2026)

This repository provides a minimal, self-contained reference implementation of the **Forward–Momentum Scaling Law** proposed in:

**Scaling and Transferability of Annealing Strategies in Large Language Model Training**  
*AAAI 2026 (Main Track, Poster)*  
Siqi Wang, Zhengyu Chen, Teng Xiao, Zheqi Lv, Jinluan Yang, Xunliang Cai, Jingang Wang, Xiaomeng Li

---

## 🔍 Overview

This repository implements the key components introduced in our paper, including:

### ✔ **Forward Effect**
Computation of the cumulative *forward learning-rate effect*, characterized by the integral
```math
S = \int \eta(t) \, dt
```


### ✔ **Annealing Momentum**
A practical proxy for the *kinetic effect* of learning-rate decay, defined via a momentum-style update: 
```math
M = \sum_t \frac{m_t}{\sqrt{v_t}+\epsilon}
```
capturing both the **rate** and **magnitude** of decay during annealing.

### ✔ **Forward–Momentum Scaling Law** 
A unified scaling formulation: 
```math 
L = L_0 + \frac{\lambda_S}{S^{\alpha_S}} + \frac{\lambda_N}{N^{\alpha_N}} + \lambda_M \cdot M
```

### ✔ **Robust Curve Fitting**
Robust Huber-loss optimization with L-BFGS-B for stable estimation of scaling parameters.

---

## 📂 Repository Contents
- annealing_scaling_law.py
- README.md


This code is intentionally minimal. It provides only the components required to reproduce the **annealing-related scaling behavior** discussed in the paper.

---

## ▶️ Usage

```python
from annealing_scaling_law import compute_S, compute_M, fit_and_evaluate_lr_mom

S = compute_S(steps, learning_rates)
M = compute_M(steps, learning_rates)

y_fit, r2, mse, params = fit_and_evaluate_lr_mom(
    y=loss_values,
    x=S,
    t=M,
    n=model_size,
    initial_params=[...]
)
```



