# **Overview: Adaptive Antibiotic Control in a Simulated Bacterial Ecosystem**

> **Goal:**  
> Simulate a dynamic bacterial colony evolving antibiotic resistance via mutation and horizontal gene transfer (EA), and train a neural RL agent to adaptively apply antibiotics (and sequencing actions) to minimize infection and resistance spread over time.

---

## Environment Layer – _Bacterial Evolution & Ecology_

The environment is a **stochastic, spatially or well-mixed simulation** of bacterial species and antibiotics.

### Entities

#### **Bacteria**

Each bacterium has a _genome vector_:  
$$ 
G = [g_\text{membrane}, g_\text{efflux}, g_\text{enzyme}, g_\text{repair}]  
$$  
Each value ∈ [0, 1] represents **resistance gene expression or strength**.

#### **Genome Parameters Meaning**

|Gene|Function|Biological analogue|
|---|---|---|
|`membrane`|Reduces drug uptake|Outer membrane permeability|
|`efflux`|Pumps drug out|Efflux pump proteins|
|`enzyme`|Degrades drug|β-lactamases, etc.|
|`repair`|Repairs damage|SOS repair response, stress resistance|

---

### Evolutionary Dynamics (EA)

The bacterial population evolves via:

- **Asexual reproduction** (binary fission)
    
- **Mutation** per gene:  
    $$ 
    g_i' = g_i + \mathcal{N}(0, \sigma_i) \quad \text{with prob } \mu_{\text{eff}}  
    $$
    
- **Death rate** depending on antibiotic stress and resistance level:  
    $$ 
    p_\text{death} = f(\text{drug conc.}, G, \text{drug type})  
    $$
    
- **Horizontal Gene Transfer (optional)**: plasmid-like gene swaps between nearby cells. Could be seen as a form of crossover.
    

#### Mutation rate

$$ 
\mu_\text{eff} = \mu_\text{base} \cdot M_\text{drug}  
$$  
where $M_\text{drug}$ = drug-specific mutagenicity factor.

Example:

|Antibiotic|Mutagenicity factor|
|---|--:|
|Ciprofloxacin|×10|
|Ampicillin|×2|
|Colistin|×1|

Mutation rates can also be varied also for different bacteria types (e.g. E.coli has a different mutation rate vs ampicillin compared to  Salmonella)

---

### Environmental Fields

|Field|Description|
|---|---|
|`drug_concentration`|For each antibiotic type; diffuses over time|
|`nutrients`|Optional logistic growth control|
|`environmental stress`|Optional variables affecting growth rate (temperature, lack of liquid medium)|

---

### Fitness

$$ 
\text{Fitness}(G, A) = r_0 \cdot (1 - \text{drug effect}(G, A)) - c(G)  
$$  
where `c(G)` is metabolic cost of maintaining resistance (trade-off mechanism).

---

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

|Observation|Normally available|Requires sequencing|
|---|---|---|
|Total bacteria count|✅|✅|
|Growth rate|✅|✅|
|Drug concentrations|✅|✅|
|Mean resistance|❌|✅ (meta)|
|Genotype distribution|❌|✅ (single-cell)|
|Mutation rates|❌|✅ (inferred or single-cell)|

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
