# **Overview: Adaptive Antibiotic Control in a Simulated Bacterial Ecosystem**

> **Goal:**  
> Simulate a dynamic bacterial colony evolving antibiotic resistance via mutation and horizontal gene transfer (EA), and train a neural RL agent to adaptively apply antibiotics (and sequencing actions) to minimize infection and resistance spread over time.

---

## Environment Overview

The **environment** simulates a Petri dish or liquid culture with:

- A population of bacteria (each with its genome)
    
- A given **antibiotic concentration** (A(t))
    
- Optional **nutrient level** (N(t))
    
- A schedule for antibiotic exposure (constant, pulsed, or gradient)
    

The environment defines the **selection pressure**: which genomes survive and reproduce.

---

## Environment Variables

|Symbol|Meaning|Example|
|---|---|---|
|($A(t)$)|Antibiotic concentration|Constant 0.6 or time-dependent pulse|
|($N(t)$)|Nutrient availability|1.0 (full) → 0.5 (limited)|
|($k_d$)|Antibiotic toxicity constant|2.0|
|($k_g$)|Max growth rate (different per bacteria type)|1.0|
|($C_r$)|Cost of resistance mechanisms|0.2–0.5 penalty factor|

### Survival component

We model the _effective antibiotic concentration_ the cell experiences as reduced by its defenses:

$$
A_{\text{eff}} = A(t) \times (1 - \text{efflux weight} \cdot \text{efflux}) \times (1 - \text{enzyme weight} \cdot \text{enzyme}) \times (1 - \text{membrane weight} \cdot \text{membrane})  
$$

The **survival probability** decreases exponentially with damage, but is improved by **repair**:

$$ 
S = e^{-k_d , A_{\text{eff}} , (1 - \text{repair weight} \cdot \text{repair})}  
$$

Each weight for each resistance system will differ for each antibiotic.

Interpretation:

- Efflux pumps reduce internal concentration.
    
- Enzymes degrade antibiotics.
    
- Membrane changes lower permeability.
    
- Repair systems reduce damage impact.
    

---

### Growth component

Resistance has a **metabolic cost**, so growth slows as defenses strengthen:

$$
G = k_g , (1 - C_r \cdot (\text{membrane} + \text{efflux} + \text{enzyme} + 0.5 \cdot \text{repair}))  
$$

(Repair is weighted lower because it’s less costly than maintaining enzymes or pumps.)

You can also add **nutrient limitation** if desired:  
$$
G = G \times N(t)  
$$

## Agent Layer – _Adaptive Control via RL and Neural Inference_

### Agent objectives

Minimize:

- Total bacterial load
    
- Resistance prevalence
    
- Sequencing and antibiotic costs
    

while learning to:

- Infer hidden evolutionary parameters (mutation rates, resistances)
    
- Choose adaptive antibiotic dosing and sequencing strategies
    

---

### Partial Observability

The agent does **not** see the true genomes.  
Instead, it can obtain information through **sampling actions**:

| Observation           | Normally available | Requires sequencing          |
| --------------------- | ------------------ | ---------------------------- |
| Total bacteria count  | ✅                 | ✅                           |
| Growth rate           | ✅ (inferred)                | ✅ (inferred)      |
| Drug concentrations   | ✅                 | ✅                           |
| Mean resistance       | ❌                 | ✅ (meta)                    |
| Genotype distribution | ❌                 | ✅ (single-cell)             |
| Mutation rates        | ❌                 | ✅ (inferred or single-cell) |
| Per type density| ❌| ✅ (meta) |

---

### **Sequencing actions**

Two distinct exploration actions:

1. **Metagenomic sequencing (bulk):**
    
    - Returns _species frequencies_ and _mean gene expression values_
        
    - Cheap, fast, noisy
        
    - Cost `C_meta`, delay `L_meta`
        
2. **Single-cell sequencing:**
    
    - Returns per-cell genomes for a small sample
        
    - Expensive, slow, accurate
        
    - Cost `C_sc`, delay `L_sc`
        

Both provide optional **information gain reward**:  
$$ 
r_\text{info} = \lambda (H_\text{prior} - H_\text{posterior})  
$$

---

### **Action space**

|Type|Description|
|---|---|
|Antibiotic selection|Choose among N drugs|
|Dose level|Continuous [0, 1]|
|Wait / no action|Do nothing|
|Metagenomic sequencing|Information action|
|Single-cell sequencing|Information action|

Total action = `[drug_type, dose, sequencing_action]`.

---

### **Neural architecture**

A **recurrent actor–critic model (e.g., PPO or A2C)** with an **inference head**:

```
Input:
    - Recent bacterial counts
    - Growth rates
    - Previous actions (drug, dose)
    - Sequencing results (if available)

Network:
    Encoder (MLP + GRU)
    ├── Mutation estimator head → μ̂ (predicted mutation rate per species)
    ├── Critic head → V(s)
    └── Policy head → π(a | s, μ̂)
```

**Dual objective:**

- Control reward (infection + cost minimization)
    
- Auxiliary supervised loss for mutation estimation  
    (trained against ground truth in simulation)
    

---

## Training Dynamics

The full training loop combines:

- **Evolutionary simulation** (for the bacteria)
    
- **Reinforcement learning** (for the agent)
    

### Pseudocode

```python
for episode in range(num_episodes):
    env.reset()
    h = model.init_hidden()
    for t in range(T):
        obs = env.observe()
        action, mu_hat, h = model(obs, h)
        next_obs, reward, done, info = env.step(action)
        rl_agent.store(obs, action, reward, next_obs)
        if done: break
    rl_agent.update()  # PPO/A2C
```

---

## Rewards and Evaluation

**Reward function:**  
$$ 
r_t =

- w_1 \cdot N_t
    
- w_2 \cdot R_t
    
- w_3 \cdot \text{AB dose}
    
- w_4 \cdot \text{sequencing cost}
    

- w_5 \cdot r_\text{info}  
    $$
    

where:

- $N_t$: total population size
    
- $R_t$: resistant fraction
    
- $r_\text{info}$: optional entropy-based information reward
    

---

## Possible Research Experiments

|Experiment|Research question|
|---|---|
|🧪 Mutagenicity effect|How do high-mutagenicity antibiotics accelerate resistance evolution?|
|🎯 Adaptive dosing|Can an RL agent learn to suppress resistance by alternating antibiotics?|
|🔍 Information policy|When does the agent choose sequencing? Is it cost-efficient?|
|🧬 Estimation accuracy|Does the agent’s predicted mutation rate (μ̂) match the true one?|
|🧫 Spatial dynamics|Do local subpopulations evolve differently under diffusion gradients?|
|⚗️ Trade-offs|Compare single-cell vs metagenomic sequencing cost-benefit ratios|

## 🧭 Concept summary diagram

```
        ┌──────────────────────────────────────────────────┐
        │           Environment (Evolutionary EA)          │
        │──────────────────────────────────────────────────│
        │  Bacteria genomes → evolve via mutation, HGT     │
        │  Drug fields → diffusion, decay                  │
        │  Sequencing → noisy info return                  │
        └──────────────────────────────────────────────────┘
                               │
                        observations
                               ▼
        ┌──────────────────────────────────────────────────┐
        │          RL Agent (Neural Policy)                │
        │──────────────────────────────────────────────────│
        │  Inputs: counts, growth, seq results             │
        │  Internal inference: estimate μ̂                  │
        │  Outputs: drug type, dose, sequencing action     │
        └──────────────────────────────────────────────────┘
                               │
                        control actions
                               ▼
        ┌──────────────────────────────────────────────────┐
        │     Updated environment (new gen. of bacteria)   │
        └──────────────────────────────────────────────────┘
```

# Baseline work division
- Bacteria has its genome
	- membrane, efflux, enzyme, repair
	- different types of bacteria with different starting resistances
	- 


## Bacteria
- Has [[#**Genome Parameters Meaning**]]

## Environment
### Food
