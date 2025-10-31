# Segmentation Fault Fix - Complete Solution

## Root Cause Analysis

Your segmentation fault was caused by **fundamental incompatibility between matplotlib's and PyQt5's event loops**.

### The Problem Architecture

```
ORIGINAL CODE:
┌─────────────────────────────────────┐
│  Qt Application & Control Panel     │
│  - Has its own event loop           │
│  - Manages window events            │
└─────────────────────────────────────┘
              ❌ CONFLICT ❌
┌─────────────────────────────────────┐
│  plt.show() & Matplotlib Figure     │
│  - BLOCKS the entire program        │
│  - Creates its own window manager   │
│  - Has separate event handling      │
└─────────────────────────────────────┘
```

**When you click or resize:** Two event handlers compete, causing memory corruption and segfaults.

---

## What the Original Code Did Wrong

### Issue #1: `plt.show()` Blocks Everything ❌
```python
# ui.py - visualizer.show() calls plt.show()
def show(self):
    plt.show()  # BLOCKS! Creates separate event loop!
```

This is the **primary culprit**. `plt.show()` creates a blocking event loop that competes with PyQt5.

### Issue #2: No Matplotlib-PyQt5 Integration ❌
```python
# visualization.py
self.fig = plt.figure(figsize=(20, 10))  # Wrong! Unmanaged window
```

This creates a matplotlib window managed by matplotlib, not PyQt5.

### Issue #3: Direct Agent List Iteration ❌
```python
for bacterium in self.model.agent_set:  # Iterator corruption!
    # Model might modify agent_set while looping
```

### Issue #4: Unsafe Scatter Plot Operations ❌
```python
self.highlight_scat.remove()  # Can fail
self.highlight_scat = self.ax.scatter(...)  # Recreate
```

---

## The Fixed Solution

### Key Changes

#### 1. Use `Figure()` Instead of `plt.figure()`
```python
# BEFORE (WRONG):
self.fig = plt.figure(figsize=(20, 10))

# AFTER (CORRECT):
from matplotlib.figure import Figure
self.fig = Figure(figsize=(20, 10))  # No window manager!
```

#### 2. Embed Matplotlib in PyQt5 Window
```python
# NEW: visualization_fixed.py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class VisualizationWindow(QtWidgets.QMainWindow):
    def __init__(self, visualizer):
        super().__init__()
        self.canvas = FigureCanvas(visualizer.fig)  # Embed!
        self.setCentralWidget(self.canvas)
```

#### 3. Never Call `plt.show()`
```python
# visualization_fixed.py
def show(self):
    """Show the matplotlib figure"""
    pass  # Do nothing! PyQt5 shows it.
```

#### 4. Single Event Loop
```python
# main_fixed.py
app = QtWidgets.QApplication(sys.argv)  # ONE event loop
# ... setup ...
sys.exit(app.exec_())  # Block here ONLY
```

#### 5. Safe Iteration
```python
# visualization_fixed.py - _on_click()
try:
    agents = list(self.model.agent_set)  # Make copy!
except Exception:
    return

for bacterium in agents:  # Safe iteration
    # ...
```

---

## File Changes Summary

### NEW FILES
- ✅ `visualization_fixed.py` - Matplotlib integration without window manager
- ✅ `ui_fixed.py` - Proper PyQt5 + Matplotlib coordination
- ✅ `main_fixed.py` - Correct entry point with single event loop

### KEEP ORIGINAL
- ✅ `control_panel.py` - Works fine as-is (with minor improvements)
- ✅ `model.py` - No changes needed
- ✅ `bacterium.py` - No changes needed

---

## How to Use

### Run the Fixed Version
```bash
python main_fixed.py
```

### NOT This (It Will Crash)
```bash
python main.py  # Uses old broken ui.py
```

---

## Architecture Comparison

### BEFORE (Broken)
```
main.py
  ├─ Creates QApplication (Qt)
  └─ Creates SimulatorUI
      ├─ Creates ControlPanel (PyQt5 window)
      ├─ Creates SimulationVisualizer
      │   └─ plt.figure() → Matplotlib takes over!
      └─ ui.run()
          └─ visualizer.show() → plt.show() BLOCKS!
              └─ CONFLICT! Two event loops fighting
```

### AFTER (Fixed)
```
main_fixed.py
  ├─ Creates QApplication (Qt) - SINGLE EVENT LOOP
  ├─ Creates BacteriaModel
  ├─ Creates SimulatorUI with:
  │   ├─ SimulationVisualizer
  │   │   └─ Figure() → No window manager
  │   │   └─ Creates plots on abstract canvas
  │   ├─ VisualizationWindow (PyQt5)
  │   │   └─ FigureCanvas embeds Figure
  │   └─ ControlPanel (PyQt5)
  ├─ ui.run() → Creates animation
  ├─ Shows windows → PyQt5 handles
  └─ app.exec_() → SINGLE event loop runs everything
```

---

## Why This Works

✅ **Single Event Loop**: Qt manages EVERYTHING including matplotlib
✅ **No Blocking**: `plt.show()` not called
✅ **Proper Embedding**: Matplotlib Figure is PyQt5 widget
✅ **Safe Interaction**: All window events go through Qt
✅ **Safe Iteration**: Agent list copied before iteration
✅ **Graceful Degradation**: All operations have error handling

---

## Testing

Verify the fix works by:
1. ✅ Click on bacteria
2. ✅ Resize windows (main and control panel)
3. ✅ Resize the figure canvas
4. ✅ Highlight/switch between bacteria
5. ✅ Run for extended periods
6. ✅ Close windows during simulation

If any issues persist:
```bash
# Check for errors
python main_fixed.py 2>&1 | head -50
```

---

## If You Want to Preserve Original main.py

If you need `main.py` to work, simply edit it:

```python
# main.py (modified)
from model import BacteriaModel
from ui_fixed import SimulatorUI  # Use FIXED version
from PyQt5 import QtWidgets
import sys

def main():
    """Initialize and run the bacteria simulation"""
    app = QtWidgets.QApplication(sys.argv)
    model = BacteriaModel()
    ui = SimulatorUI(model)
    ui.run()
    ui.control_panel.window.show()
    ui.viz_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```

---

## Performance Notes

The fixed version should be:
- ✅ **More stable** - No segfaults
- ✅ **More responsive** - Single event loop handles all events
- ✅ **Same speed** - Animation rate unchanged
- ✅ **Lower memory** - No duplicate window managers

---

## Troubleshooting

### Still Getting Crashes?
```python
# main_fixed.py - uncomment for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Animation Not Updating?
Check that FigureCanvas is properly embedded - look at VisualizationWindow.__init__()

### Slow Performance?
Performance mode in control panel can help, but the fix itself is about stability, not speed.

### Individual Plotter Not Working?
Make sure tracking.py doesn't have its own matplotlib figures that conflict. They should also use `Figure()` not `plt.figure()`.

---

## Summary

**The Fix:** Embed matplotlib in PyQt5 instead of creating competing event loops.

**Files to Use:**
- `main_fixed.py` ← Run this
- `ui_fixed.py` ← New UI
- `visualization_fixed.py` ← Fixed visualization
- `control_panel.py` ← Unchanged
- `model.py` ← Unchanged

**What Changed:** Architecture and event loop management, NOT simulation logic.
