"""
Updated Bacteria simulation (Mesa model) with continuous space.
- Reworked to avoid deprecated Mesa schedulers (manage agents manually)
- Proper Model and Agent initialization
- Toggleable horizontal gene transfer (HGT) from UI
- Tk UI event pumping integrated into Matplotlib animation (no separate Tk thread)
- cache_frame_data disabled for FuncAnimation to avoid unbounded cache warning

Run: python mesa_bacteria_simulation.py
Dependencies: mesa, numpy, scipy, matplotlib, tkinter (optional)
"""

import sys
import time
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter

from mesa import Model, Agent
from mesa.space import ContinuousSpace

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

INITIAL_BACTERIA = 20
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
    def __init__(self, model, pos, resistance=0.1):
        # Correct Agent initialization signature: Agent(unique_id, model)
        super().__init__(model)
        self.pos = pos
        self.energy = random.uniform(1.0, 2.0)
        self.resistance = float(resistance)  # value in [0,1]
        self.speed = BACTERIA_SPEED * random.uniform(0.8, 1.2)

    def step(self):
        # Movement: biased random walk toward nutrient gradient
        nx, ny = self.model.nutrient_to_field_coords(self.pos)
        grad = self.model.compute_gradient_at_field(nx, ny)
        g = np.array(grad, dtype=float)
        if np.linalg.norm(g) > 1e-8:
            g = g / (np.linalg.norm(g) + 1e-9)
        else:
            g = np.zeros(2)
        rand_dir = np.random.normal(size=2)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-9
        alpha = 0.8
        direction = alpha * g + (1 - alpha) * rand_dir
        direction /= np.linalg.norm(direction) + 1e-9

        # Move
        new_x = self.pos[0] + direction[0] * self.speed
        new_y = self.pos[1] + direction[1] * self.speed
        new_x = max(0, min(self.model.space.x_max, new_x))
        new_y = max(0, min(self.model.space.y_max, new_y))

        # Update position using ContinuousSpace API
        self.pos = (new_x, new_y)
        try:
            self.model.space.move_agent(self, self.pos)
        except Exception:
            # # Some Mesa versions may not provide move_agent; try place_agent
            # try:
            #     self.model.space.place_agent(self, self.pos)
            # except Exception:
            #     pass
            raise Exception("Agent movement failed")

        # Consume food at location (sampled from field)
        fx, fy = self.model.nutrient_to_field_coords(self.pos)
        food_amount = self.model.sample_field(self.model.food_field, fx, fy)
        consumed = min(food_amount, FOOD_CONSUMPTION_PER_STEP)
        self.model.subtract_from_field(self.model.food_field, fx, fy, consumed)
        self.energy += consumed * ENERGY_FROM_FOOD_SCALE

        # Antibiotic effect: death probability depends on antibiotic concentration and resistance
        a_conc = self.model.sample_field(self.model.antibiotic_field, fx, fy)
        k_d = 2.0
        effective = max(0.0, a_conc - self.resistance)
        p_die = 1.0 - math.exp(-k_d * effective)
        if random.random() < p_die:
            # mark for removal
            self.model.to_remove.add(self)
            return

        # Reproduction
        if self.energy >= REPRODUCTION_ENERGY_THRESHOLD:
            self.energy /= 2.0
            new_res = self.resistance + random.gauss(0, MUTATION_STD)
            new_res = float(min(max(new_res, 0.0), 1.0))
            child = Bacterium(self.model, pos=self.pos, resistance=new_res)
            # Defer adding to model until after stepping through all agents
            self.model.new_agents.append(child)

    def advance(self):
        # placeholder if later one wants two-phase updates
        pass


# -----------------------
# Model definition
# -----------------------
class BacteriaModel(Model):
    def __init__(self, N=INITIAL_BACTERIA, width=WIDTH, height=HEIGHT, enable_hgt=True):
        super().__init__()  # explicit model init to avoid FutureWarning
        self.num_agents = N
        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)
        self.random = random.Random()

        # agent container instead of deprecated scheduler
        self.agent_set = set()
        self._next_id = 0

        # fields
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

        # create agents
        for _ in range(self.num_agents):
            x, y = random.uniform(0, width), random.uniform(0, height)
            resistance = random.uniform(0.0, 0.2)
            a = Bacterium(self, (x, y), resistance=resistance)
            # # place and register
            # try:
            #     self.space.place_agent(a, (x, y))
            # except Exception:
            #     # fallback
            #     pass
            self.agent_set.add(a)

        self.running = True
        self.step_count = 0

        # HGT toggle
        self.enable_hgt = bool(enable_hgt)

    def next_id(self):
        nid = self._next_id
        self._next_id += 1
        return nid

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
        fx = (pos[0] / self.width) * (self.field_w - 1)
        fy = (pos[1] / self.height) * (self.field_h - 1)
        return fx, fy

    def sample_field(self, field, fx, fy):
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
        x = int(round(fx))
        y = int(round(fy))
        x = min(max(x, 0), self.field_w - 1)
        y = min(max(y, 0), self.field_h - 1)
        field[x, y] = max(0.0, field[x, y] - amount)

    def compute_gradient_at_field(self, fx, fy):
        x = int(round(fx))
        y = int(round(fy))
        x0 = min(max(x - 1, 0), self.field_w - 1)
        x1 = min(max(x + 1, 0), self.field_w - 1)
        y0 = min(max(y - 1, 0), self.field_h - 1)
        y1 = min(max(y + 1, 0), self.field_h - 1)
        gx = self.food_field[x1, y] - self.food_field[x0, y]
        gy = self.food_field[x, y1] - self.food_field[x, y0]
        gx *= self.field_w / self.width
        gy *= self.field_h / self.height
        return gx, gy

    # ---------------------
    # Antibiotic control
    # ---------------------
    def apply_antibiotic(self, amount):
        if amount <= 0:
            return
        self.antibiotic_field += float(amount)

    # ---------------------
    # HGT: simple averaging of resistance when close
    # ---------------------
    def horizontal_gene_transfer(self):
        agents = list(self.agent_set)
        for i, a in enumerate(agents):
            # use ContinuousSpace neighbors lookup
            try:
                neighbors = self.space.get_neighbors(
                    a.pos, HGT_RADIUS, include_center=False
                )
            except Exception:
                # fallback: brute force
                neighbors = [
                    b
                    for b in agents
                    if b is not a
                    and np.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) <= HGT_RADIUS
                ]
            for nb in neighbors:
                if random.random() < HGT_PROB:
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
        self.food_field = gaussian_filter(self.food_field, sigma=FOOD_DIFFUSION_SIGMA)
        self.antibiotic_field *= 1 - ANTIBIOTIC_DECAY

        # Prepare collections
        self.to_remove.clear()
        self.new_agents.clear()

        # Step each agent
        for a in list(self.agent_set):
            try:
                a.step()
            except Exception:
                # Avoid one agent failing stopping the sim
                pass

        # Remove dead agents
        for a in list(self.to_remove):
            try:
                # remove from space and agent set
                try:
                    self.space.remove_agent(a)
                except Exception:
                    pass
                if a in self.agent_set:
                    self.agent_set.remove(a)
            except Exception:
                pass

        # Add newborns
        for child in self.new_agents:
            # try:
            #     self.space.place_agent(child, child.pos)
            # except Exception:
            #     pass
            self.agent_set.add(child)

        # Horizontal gene transfer (toggleable)
        if self.enable_hgt:
            try:
                self.horizontal_gene_transfer()
            except Exception:
                pass

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

        # Start Tk UI if available, but DO NOT start mainloop in a separate thread.
        # We'll pump Tk events from the animation loop to avoid cross-thread Tcl calls.
        if tk is not None:
            try:
                self.root = tk.Tk()
                self.root.title("Control Panel")
                self.build_controls()
                # do NOT call self.root.mainloop() in another thread
            except Exception as e:
                print(f"Tk UI init failed: {e}")
                self.root = None
        else:
            self.root = None

    def build_controls(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid()
        ttk.Label(frm, text="Simulation Controls").grid(column=0, row=0, columnspan=2)

        self.pause_btn = ttk.Button(frm, text="Pause", command=self.toggle_pause)
        self.pause_btn.grid(column=0, row=1)
        ttk.Button(frm, text="Reset", command=self.reset_sim).grid(column=1, row=1)

        ttk.Label(frm, text="Antibiotic dose:").grid(column=0, row=2)
        self.dose_var = tk.DoubleVar(value=0.5)
        ttk.Entry(frm, textvariable=self.dose_var, width=8).grid(column=1, row=2)

        ttk.Button(frm, text="Apply antibiotic", command=self.apply_antibiotic_ui).grid(
            column=0, row=3, columnspan=2, pady=(4, 4)
        )

        ttk.Label(frm, text="Latest dose applied:").grid(column=0, row=4)
        self.latest_label = ttk.Label(frm, text="0.0")
        self.latest_label.grid(column=1, row=4)

        # HGT toggle
        self.hgt_var = tk.BooleanVar(value=self.model.enable_hgt)
        self.hgt_check = ttk.Checkbutton(
            frm, text="Enable HGT", variable=self.hgt_var, command=self.toggle_hgt
        )
        self.hgt_check.grid(column=0, row=5, columnspan=2)

    def toggle_pause(self):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")
        if self.paused:
            self.pause_btn.config(text="Resume")
        else:
            self.pause_btn.config(text="Pause")

    def reset_sim(self):
        print("Reset not implemented in this prototype")

    def apply_antibiotic_ui(self):
        try:
            val = float(self.dose_var.get())
        except Exception:
            val = 0.0
        self.model.apply_antibiotic(val)
        self.latest_dose = val
        if self.root is not None:
            try:
                self.latest_label.config(text=f"{val:.3f}")
            except Exception:
                pass

    def toggle_hgt(self):
        try:
            new_val = bool(self.hgt_var.get())
        except Exception:
            new_val = not self.model.enable_hgt
        self.model.enable_hgt = new_val

    def init_plot(self):
        self.ax.set_xlim(0, self.model.width)
        self.ax.set_ylim(0, self.model.height)
        self.ax.set_aspect("equal")
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
        xs = [a.pos[0] for a in self.model.agents]
        ys = [a.pos[1] for a in self.model.agents]
        colors = [a.resistance for a in self.model.agents]
        self.scat = self.ax.scatter(xs, ys, c=colors, vmin=0, vmax=1, s=20)
        return (self.scat,)

    def pump_tk(self):
        # Pump tkinter events from the main thread (safe) so we don't run mainloop in another thread
        if self.root is not None:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                # If the window has been closed or errors occur, ignore
                pass
            except Exception:
                pass

    def update_plot(self, frame):
        # pump tk events so controls are responsive without separate Tk thread
        self.pump_tk()

        if not self.paused:
            # run one model step per frame (or more, if desired)
            try:
                self.model.step()
            except Exception:
                pass

        # update images and scatter
        try:
            food_img = np.rot90(self.model.food_field)
            self.im_food.set_data(food_img)
            ab_img = np.rot90(self.model.antibiotic_field)
            self.im_ab.set_data(ab_img)
        except Exception:
            pass

        xs = [a.pos[0] for a in self.model.agents]
        ys = [a.pos[1] for a in self.model.agents]
        colors = [a.resistance for a in self.model.agents]
        if len(xs) == 0:
            self.scat.set_offsets(np.empty((0, 2)))
        else:
            self.scat.set_offsets(np.c_[xs, ys])
            self.scat.set_array(np.array(colors))
        self.ax.set_title(
            f"Step: {self.model.step_count}  Agents: {len(self.model.agents)}"
        )
        return (self.scat,)

    def run(self):
        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            init_func=self.init_plot,
            interval=200,
            blit=False,
            cache_frame_data=False,  # avoid unbounded cache warning
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
