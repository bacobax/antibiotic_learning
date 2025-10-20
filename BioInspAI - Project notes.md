# Antibiotic Microbial Resistance (AMR) simulation
## Simple model (no spatial, no gene transfer, no multi ab)
$$
\text{Genome} = [\text{membrane}, \text{efflux}, \text{enzyme}, \text{repair}]  
$$

Each gene is a continuous value in **[0, 1]**, representing the _strength or efficiency_ of that resistance mechanism.

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
|($k_g$)|Max growth rate|1.0|
|($C_r$)|Cost of resistance mechanisms|0.2–0.5 penalty factor|

---

## Fitness Function Design

The **fitness** combines:

1. **Survival** against antibiotic stress (affected by resistance mechanisms)
    
2. **Growth efficiency** (penalized by resistance costs)
    

$$
F = \text{Survival} \times \text{Growth}  
$$

---

### Survival component

We model the _effective antibiotic concentration_ the cell experiences as reduced by its defenses:

$$
A_{\text{eff}} = A(t) \times (1 - 0.4 \cdot \text{efflux}) \times (1 - 0.3 \cdot \text{enzyme}) \times (1 - 0.2 \cdot \text{membrane})  
$$

The **survival probability** decreases exponentially with damage, but is improved by **repair**:

$$ 
S = e^{-k_d , A_{\text{eff}} , (1 - 0.5 \cdot \text{repair})}  
$$

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

---

### Combined Fitness

Finally:  
$$
F = S \times G  
$$

or in code form:

```python
def evaluate(individual, A=0.6, kd=2.0, kg=1.0, Cr=0.3):
    membrane, efflux, enzyme, repair = individual

    # Effective antibiotic concentration reduced by defenses
    A_eff = A * (1 - 0.4 * efflux) * (1 - 0.3 * enzyme) * (1 - 0.2 * membrane)

    # Survival: exponential decay with repair compensation
    survival = np.exp(-kd * A_eff * (1 - 0.5 * repair))

    # Growth: penalized by resistance maintenance cost
    growth = kg * (1 - Cr * (membrane + efflux + enzyme + 0.5 * repair))

    fitness = survival * growth
    return (fitness,)
```

---

## Environment Dynamics
### Antibiotic dynamics

Pulsating antiobiotic propagation after $n$ generations.

## Observables

Track across generations:

- Mean and variance of each gene
    
- Mean fitness
    
- Survival rate after antibiotic introduction
    
- Population diversity (Shannon entropy)
    

This gives a clear biological narrative:

> “Over generations, efflux and enzyme activity increase after antibiotic exposure, while membrane changes plateau due to higher energetic costs.”

# Extensions
- Spatial simulation (2d grid)
	- Biofilm creation & quorum sensing
- Horizontal gene transfer (bacteria has a probability of transfering resistance genes): linked to a distance metric, either through visual simul or probability
- Implementation of different bacterial species (different resistances, weights and interactions with other bacteria)
- Reiforcement learning for antibiotics pharmacodynamics/pharmacokinetics (more resistence and ab combinations) of third party agent
- Sequence: takes a sample of the bacteria to discover the different species of bacteria present and their genome values. Sequencing has a cost.
	- Metagenomic: uncovers bacteria types and a mean of their genome values. Less informative but cheaper and faster.
	- Single cell: linked genotypes per cell of a sample. Is way more informative but much more costly and long to perform.
- Multi ab/multi resistance: different resistance genes, diffrent ab for each resistance. Can be done by simulating effective
- Persistor states (quiescent states of no microbial metabolism and increased resistance)