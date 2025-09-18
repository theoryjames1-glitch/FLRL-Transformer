
# 🧩 **FLRL-Transformer (Fuzzy Logic Reinforcement Learning Transformer)**

### 🔑 Core Idea

* The **Transformer** learns to represent the environment state as **fuzzy concepts**.
* Instead of fixed membership functions, the **attention mechanism** maps observations → fuzzy degrees (e.g., "angle is negative 0.7, positive 0.2").
* These fuzzy representations drive **rule inference**, which selects actions and updates rewards.
* The fuzzy logic layer is **differentiable** so the Transformer can learn rules end-to-end.

---

## 1. Architecture

### **Inputs**

* Raw environment state vector \$s\_t\$ (e.g., CartPole: `[x, x_dot, angle, angle_dot]`).
* Optionally, history of last \$k\$ states (for recurrence).

### **Transformer Encoder**

* Projects \$s\_t\$ into fuzzy membership degrees:

  $$
  \mu_i = \text{Softmax}(W s_t)
  $$
* Attention layers allow the model to “focus” on fuzzy concepts (e.g., angle vs velocity).

### **Fuzzy Rule Layer**

* Implemented as a **differentiable fuzzy inference system (FIS)**:

  * Antecedents: learned membership functions.
  * Rules: parameterized weights (like attention heads).
  * Defuzzification: weighted sum → continuous action score.

### **Policy Head**

* Discretizes fuzzy action score into discrete environment action (`left/right` for CartPole).
* Optionally, outputs continuous force (for continuous environments).

---

## 2. Learning Mechanism

* **Transformer parameters + fuzzy memberships + rules** are updated via **policy gradient (PPO/DQN)**.
* Reward is environment return.
* Gradient flows through fuzzy inference, so the system *learns rules* instead of relying on hand-crafted ones.

---

## 3. Pseudocode (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FLRLTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(state_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # fuzzy membership generator
        self.membership_proj = nn.Linear(d_model, 16)  # 16 fuzzy concepts
        # rule weights
        self.rule_proj = nn.Linear(16, action_dim)

    def forward(self, state):
        # state: [B, state_dim]
        x = self.embed(state).unsqueeze(1)  # [B, 1, d_model]
        x = self.transformer(x)             # [B, 1, d_model]

        fuzzy_memberships = torch.sigmoid(self.membership_proj(x))  # [B, 1, 16]
        fuzzy_rules = self.rule_proj(fuzzy_memberships)             # [B, 1, action_dim]

        # defuzzify into policy logits
        logits = fuzzy_rules.squeeze(1)
        probs = F.softmax(logits, dim=-1)
        return probs
```

---

## 4. Training Loop (Gymnasium)

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
model = FLRLTransformer(state_dim=4, action_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for episode in range(200):
    state, _ = env.reset()
    done, total_reward = False, 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = model(state_t)
        action = torch.multinomial(probs, 1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # simple policy gradient update
        loss = -torch.log(probs[0, action]) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    print(f"Episode {episode} | Reward {total_reward}")
```

---

## 5. Extensions

* Replace the **fixed Softmax memberships** with **trainable Gaussian functions** (like scikit-fuzzy).
* Add **attention-based rule learning**: each attention head corresponds to a fuzzy rule.
* Scale up to **continuous control** with SAC/PPO + FLRL Transformer.
* Visualization: plot how membership functions evolve during training → “rules emerging”.

---

✅ This gives us a **differentiable fuzzy logic system inside a Transformer**, trained by reinforcement learning.
It’s basically *"Learnable Fuzzy Rules as Attention"*.

---

Would you like me to **add visualization code** (plotting learned fuzzy membership functions over time) so we can actually *see* the fuzzy rules evolving while training?
