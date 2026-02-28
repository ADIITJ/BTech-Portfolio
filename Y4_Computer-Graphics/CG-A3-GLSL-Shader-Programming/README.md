<!-- PORTFOLIO CONTEXT -->
**Subject:** Computer Graphics — Assignment 3 (Y4) · IIT Jodhpur, Dept. of CSE & AI
**Problem:** 3D world building using OpenGL with custom GLSL vertex and fragment shaders. Implemented as a real-time 3D stacker game demonstrating perspective projection, collision detection, and shader-based visual feedback.
**Tech Stack:** Python, PyOpenGL, GLSL, GLFW, pyrr
<!-- END PORTFOLIO CONTEXT -->

---

# 3D Stacker Game

**Author:** Atharva Date (B22AI045)  
**Course Assignment:** 3D World Building

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install glfw PyOpenGL PyOpenGL_accelerate numpy pyrr

# Run the game
python main.py
```

## Controls

- **SPACEBAR** - Drop the cube
- **R** - Restart the game
- **ESC** - Close window

## Game Rules

1. A cube slides back and forth horizontally
2. Press spacebar to drop it
3. The cube must land at least partially on the previous cube to stack
4. If it completely misses, the cube falls and game ends
5. Stack as high as you can!

## Visual Hints

The active cube changes color to help you:
- **Bright Green** - Perfect alignment
- **Orange** - Okay placement
- **Red** - Very risky!

## Requirements

The following packages are needed (installed via the setup above):
- glfw
- PyOpenGL
- PyOpenGL_accelerate
- numpy
- pyrr

## Files

- `main.py` - Main game logic and rendering
- `basic.vert` - Vertex shader (3D transformations)
- `basic.frag` - Fragment shader (coloring)
- `DOCUMENTATION.md` - Complete technical documentation

## Features

✅ Smooth sine-wave sliding motion  
✅ Collision detection with partial overlap  
✅ 3D perspective camera view  
✅ Visual feedback system  
✅ Falling animation on game over  
✅ Score tracking  

Enjoy stacking! 🎮
