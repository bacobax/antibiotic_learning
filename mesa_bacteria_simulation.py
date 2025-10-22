"""
Bacteria simulation (Mesa model) with continuous space.
- ContinuousSpace for agents
- Nutrient field: 2D numpy array with diffusion and consumption
- Antibiotic: scalar concentration applied uniformly when user triggers
- Visualization: matplotlib animation of agents and nutrient/antibiotic overlays
- Control UI: simple Tkinter window to Pause/Resume and apply antibiotic doses (metagenomic sequencing not implemented here)

Run: python mesa_bacteria_simulation.py
Dependencies: mesa, numpy, scipy, matplotlib, tkinter
"""

import sys
import threading
import time
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter

from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.time import SimultaneousActivation

import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None

# -----------------------
# Configuration / Params
# -----------------------
WIDTH = 100.0  # continuous width
HEIGHT = 100.0
GRID_RES = 200  # resolution for nutrient & antibiotic fields (square grid)

INITIAL_BACTERIA = 120
FOOD_DIFFUSION_SIGMA = 1.0  # for gaussian_filter diffusion approximation
FOOD_DECAY = 0.0
FOOD_CONSUMPTION_PER_STEP = 0.2
BACTERIA_SPEED = 0.8
REPRODUCTION_ENERGY_THRESHOLD = 5.0
ENERGY_FROM_FOOD_SCALE = 1.0
MUTATION_STD = 0.03
HGT_RADIUS = 1.5  # horizontal gene transfer radius
HGT_PROB = 0.001

ANTIBIOTIC_DECAY = 0.05  # per sim step

CONTROL_INTERVAL = 20  # timesteps between control checks (UI applies when pressed)


# -----------------------
# Agent definition
# -----------------------
class Bacterium(Agent):
    def __init__(self, unique_id, model, pos, resistance=0.1):
        super().__init__(unique_id, model)
        self.pos = pos
        self.energy = random.uniform(1.0, 2.0)
        self.resistance = resistance  # value in [0,1]
        self.speed = BACTERIA_SPEED * random.uniform(0.8, 1.2)

    def step(self):
        # Movement: biased random walk toward nutrient gradient
        nx, ny = self.model.nutrient_to_field_coords(self.pos)
        grad = self.model.compute_gradient_at_field(nx, ny)
        # Normalize grad
        g = np.array(grad, dtype=float)
        if np.linalg.norm(g) > 1e-8:
            g = g / (np.linalg.norm(g) + 1e-9)
        else:
            g = np.zeros(2)
        rand_dir = np.random.normal(size=2)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-9
        # bias weight
        alpha = 0.8
        direction = alpha * g + (1 - alpha) * rand_dir
        direction /= np.linalg.norm(direction) + 1e-9

        # Move
        new_x = self.pos[0] + direction[0] * self.speed
        new_y = self.pos[1] + direction[1] * self.speed
        # Keep inside bounds
        new_x = min(max(new_x, 0.0), self.model.width)
        new_y = min(max(new_y, 0.0), self.model.height)
        self.model.space.move_agent(self, (new_x, new_y))

        # Consume food at location (sampled from field)
        fx, fy = self.model.nutrient_to_field_coords(self.pos)
        # Bilinear sample
        food_amount = self.model.sample_field(self.model.food_field, fx, fy)
        consumed = min(food_amount, FOOD_CONSUMPTION_PER_STEP)
        self.model.subtract_from_field(self.model.food_field, fx, fy, consumed)
        self.energy += consumed * ENERGY_FROM_FOOD_SCALE

        # Antibiotic effect: death probability depends on antibiotic concentration and resistance
        a_conc = self.model.sample_field(self.model.antibiotic_field, fx, fy)
        # survival probability model (simple): P_survive = exp(-k*(A - R)) clipped
        k_d = 2.0
        effective = max(0.0, a_conc - self.resistance)
        p_die = 1.0 - math.exp(-k_d * effective)
        if random.random() < p_die:
            # mark for removal by scheduler by setting a flag
            self.model.to_remove.add(self)
            return

        # Reproduction
        if self.energy >= REPRODUCTION_ENERGY_THRESHOLD:
            self.energy /= 2.0
            # Offspring inherits resistance with mutation
            new_res = self.resistance + random.gauss(0, MUTATION_STD)
            new_res = float(min(max(new_res, 0.0), 1.0))
            child = Bacterium(
                self.model.next_id(), self.model, pos=self.pos, resistance=new_res
            )
            self.model.new_agents.append(child)

    def advance(self):
        pass


# -----------------------
# Model definition
# -----------------------
class BacteriaModel(Model):
    def __init__(self, N=INITIAL_BACTERIA, width=WIDTH, height=HEIGHT):
        self.num_agents = N
        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)
        self.random = random.Random()
        self.schedule = SimultaneousActivation(self)

        # field grids (GRID_RES x GRID_RES)
        self.field_w = GRID_RES
        self.field_h = GRID_RES
        self.food_field = np.zeros((self.field_w, self.field_h), dtype=float)
        self.antibiotic_field = np.zeros_like(self.food_field)

        # initialize food with several gaussian patches
        for _ in range(6):
            cx = random.uniform(0, self.field_w - 1)
            cy = random.uniform(0, self.field_h - 1)
            sigma = random.uniform(6, 18)
            amplitude = random.uniform(2.0, 5.0)
            self.add_gaussian_patch(self.food_field, cx, cy, sigma, amplitude)

        self.to_remove = set()
        self.new_agents = []

        for i in range(self.num_agents):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            resistance = random.uniform(0.0, 0.2)
            a = Bacterium(i, self, (x, y), resistance=resistance)
            self.space.place_agent(a, (x, y))
            self.schedule.add(a)

        self.running = True
        self.step_count = 0

    # ---------------------
    # Field utilities
    # ---------------------
    def add_gaussian_patch(self, field, cx, cy, sigma, amplitude):
        X, Y = np.meshgrid(
            np.arange(self.field_w), np.arange(self.field_h), indexing="ij"
        )
        patch = amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
        field += patch

    def nutrient_to_field_coords(self, pos):
        # pos is continuous in [0,width]x[0,height]
        fx = (pos[0] / self.width) * (self.field_w - 1)
        fy = (pos[1] / self.height) * (self.field_h - 1)
        return fx, fy

    def sample_field(self, field, fx, fy):
        # bilinear interpolation
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        x1 = min(x0 + 1, self.field_w - 1)
        y1 = min(y0 + 1, self.field_h - 1)
        dx = fx - x0
        dy = fy - y0
        v00 = field[x0, y0]
        v10 = field[x1, y0]
        v01 = field[x0, y1]
        v11 = field[x1, y1]
        v = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        return v

    def subtract_from_field(self, field, fx, fy, amount):
        # subtract amount from nearest cell (simple)
        x = int(round(fx))
        y = int(round(fy))
        x = min(max(x, 0), self.field_w - 1)
        y = min(max(y, 0), self.field_h - 1)
        field[x, y] = max(0.0, field[x, y] - amount)

    def compute_gradient_at_field(self, fx, fy):
        # finite difference on nearby integer coords
        x = int(round(fx))
        y = int(round(fy))
        x0 = min(max(x - 1, 0), self.field_w - 1)
        x1 = min(max(x + 1, 0), self.field_w - 1)
        y0 = min(max(y - 1, 0), self.field_h - 1)
        y1 = min(max(y + 1, 0), self.field_h - 1)
        gx = self.food_field[x1, y] - self.food_field[x0, y]
        gy = self.food_field[x, y1] - self.food_field[x, y0]
        # convert to continuous scale
        gx *= self.field_w / self.width
        gy *= self.field_h / self.height
        return gx, gy

    # ---------------------
    # Antibiotic control
    # ---------------------
    def apply_antibiotic(self, amount):
        # amount is scalar added uniformly to 2D field
        if amount <= 0:
            return
        self.antibiotic_field += float(amount)

    # ---------------------
    # HGT: simple averaging of resistance when close
    # ---------------------
    def horizontal_gene_transfer(self):
        agents = list(self.schedule.agents)
        for i, a in enumerate(agents):
            neighbors = self.space.get_neighbors(
                a.pos, HGT_RADIUS, include_center=False
            )
            for nb in neighbors:
                if random.random() < HGT_PROB:
                    # simple model: exchange small fraction
                    mix = 0.5
                    new_res_a = a.resistance * (1 - mix) + nb.resistance * mix
                    new_res_b = nb.resistance * (1 - mix) + a.resistance * mix
                    a.resistance = float(min(max(new_res_a, 0.0), 1.0))
                    nb.resistance = float(min(max(new_res_b, 0.0), 1.0))

    # ---------------------
    # Step
    # ---------------------
    def step(self):
        # Update fields: diffuse nutrient and antibiotic
        # approximate diffusion using gaussian filter
        self.food_field = gaussian_filter(self.food_field, sigma=FOOD_DIFFUSION_SIGMA)
        # decay antibiotic
        self.antibiotic_field *= 1 - ANTIBIOTIC_DECAY

        # Schedule agents
        self.to_remove.clear()
        self.new_agents.clear()
        self.schedule.step()

        # remove dead
        for a in list(self.to_remove):
            try:
                self.space.remove_agent(a)
                self.schedule.remove(a)
            except Exception:
                pass

        # add newborns
        for child in self.new_agents:
            self.space.place_agent(child, child.pos)
            self.schedule.add(child)

        # horizontal gene transfer
        self.horizontal_gene_transfer()

        self.step_count += 1


# -----------------------
# Visualization + Control UI
# -----------------------
class SimulatorUI:
    def __init__(self, model):
        self.model = model
        self.paused = False
        self.latest_dose = 0.0

        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.scat = None
        self.im_food = None
        self.im_ab = None

        # Start Tk UI in another thread if available
        if tk is not None:
            self.root = tk.Tk()
            self.root.title("Control Panel")
            self.build_controls()
            threading.Thread(target=self.root.mainloop, daemon=True).start()
        else:
            self.root = None

    def build_controls(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid()
        ttk.Label(frm, text="Simulation Controls").grid(column=0, row=0, columnspan=2)

        self.pause_btn = ttk.Button(frm, text="Pause", command=self.toggle_pause)
        self.pause_btn.grid(column=0, row=1)
        ttk.Button(frm, text="Reset", command=self.reset_sim).grid(column=1, row=1)

        ttk.Label(frm, text="Antibiotic dose:").grid(column=0, row=2)
        self.dose_var = tk.DoubleVar(value=0.5)
        ttk.Entry(frm, textvariable=self.dose_var).grid(column=1, row=2)

        ttk.Button(frm, text="Apply antibiotic", command=self.apply_antibiotic_ui).grid(
            column=0, row=3, columnspan=2
        )

        ttk.Label(frm, text="Latest dose applied:").grid(column=0, row=4)
        self.latest_label = ttk.Label(frm, text="0.0")
        self.latest_label.grid(column=1, row=4)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Resume")
        else:
            self.pause_btn.config(text="Pause")

    def reset_sim(self):
        # not fully implemented
        print("Reset not implemented in this prototype")

    def apply_antibiotic_ui(self):
        val = float(self.dose_var.get())
        self.model.apply_antibiotic(val)
        self.latest_dose = val
        if self.root is not None:
            self.latest_label.config(text=f"{val:.3f}")

    def init_plot(self):
        self.ax.set_xlim(0, self.model.width)
        self.ax.set_ylim(0, self.model.height)
        self.ax.set_aspect("equal")
        # plot food as background
        food_img = np.rot90(self.model.food_field)
        self.im_food = self.ax.imshow(
            food_img,
            extent=[0, self.model.width, 0, self.model.height],
            alpha=0.6,
            cmap="Greens",
        )
        ab_img = np.rot90(self.model.antibiotic_field)
        self.im_ab = self.ax.imshow(
            ab_img,
            extent=[0, self.model.width, 0, self.model.height],
            alpha=0.35,
            cmap="Reds",
        )
        xs = [a.pos[0] for a in self.model.schedule.agents]
        ys = [a.pos[1] for a in self.model.schedule.agents]
        colors = [a.resistance for a in self.model.schedule.agents]
        self.scat = self.ax.scatter(xs, ys, c=colors, vmin=0, vmax=1, s=20)
        return (self.scat,)

    def update_plot(self, frame):
        if not self.paused:
            # run several model steps per frame for speed
            for _ in range(1):
                self.model.step()
        # update images and scatter
        food_img = np.rot90(self.model.food_field)
        self.im_food.set_data(food_img)
        ab_img = np.rot90(self.model.antibiotic_field)
        self.im_ab.set_data(ab_img)
        xs = [a.pos[0] for a in self.model.schedule.agents]
        ys = [a.pos[1] for a in self.model.schedule.agents]
        colors = [a.resistance for a in self.model.schedule.agents]
        if len(xs) == 0:
            self.scat.set_offsets(np.empty((0, 2)))
        else:
            self.scat.set_offsets(np.c_[xs, ys])
            self.scat.set_array(np.array(colors))
        self.ax.set_title(
            f"Step: {self.model.step_count}  Agents: {len(self.model.schedule.agents)}"
        )
        return (self.scat,)

    def run(self):
        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            init_func=self.init_plot,
            interval=200,
            blit=False,
        )
        plt.show()


# -----------------------
# Entrypoint
# -----------------------
def main():
    model = BacteriaModel(N=INITIAL_BACTERIA)
    ui = SimulatorUI(model)
    ui.run()


if __name__ == "__main__":
    main()
