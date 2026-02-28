# 3D Stacker Game - Technical Documentation

**Name:** Atharva Date  
**Roll Number:** B22AI045  
**Assignment:** Assignment 3 - 3D World Building  
**Date:** February 24, 2026

---

## 1. Game Overview

This project implements a 3D Stacker game where players must time their drops to stack cubes accurately on top of each other. The game uses OpenGL for rendering, GLFW for window management, and implements various 3D transformations to create an engaging gameplay experience.

### Game Mechanics
- A cube slides horizontally across the screen following a smooth sine wave pattern
- Players press SPACEBAR to drop the cube at the current position  
- If the dropped cube overlaps with the previous cube (even partially), it stacks successfully
- If there's no overlap at all, the cube falls and the game ends
- The camera is positioned at an angle to provide a proper 3D perspective of the growing stack

---

## 2. Technical Implementation

### Task 1: Dynamic Sliding Movement (25 marks)

The sliding mechanism uses a trigonometric sine function to create smooth, predictable horizontal motion:

```python
active_cube_pos[0] = math.sin(current_time * SLIDE_SPEED) * SLIDE_RANGE
```

**How it works:**
- `current_time`: The elapsed time from `glfw.get_time()`, continuously increasing
- `SLIDE_SPEED`: Controls how fast the cube oscillates (set to 1.5 for moderate difficulty)
- `SLIDE_RANGE`: Determines the maximum distance the cube travels left and right (3.0 units)
- The sine function naturally oscillates between -1 and 1, creating smooth back-and-forth motion

**Why sine wave?**
The sine function provides:
- Smooth acceleration and deceleration at the turning points
- Predictable timing that players can learn
- Natural-looking motion that's visually appealing

The Y-coordinate is updated based on the current layer:
```python
active_cube_pos[1] = 0.5 + current_layer * LAYER_HEIGHT
```

This ensures each new cube appears one layer higher than the previous.

### Task 2: Interactive Drop System (30 marks)

The drop mechanism uses keyboard input detection with debouncing to prevent accidental multiple triggers:

```python
if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and not spacebar_pressed:
    spacebar_pressed = True
    # Lock the current position
    dropped_x = active_cube_pos[0]
    dropped_y = active_cube_pos[1]
    dropped_z = active_cube_pos[2]
```

**Key features:**
- **Debouncing:** The `spacebar_pressed` flag prevents multiple drops from holding the key
- **Position locking:** When dropped, the cube's exact position at that moment is saved
- **State transition:** The game checks collision, then either adds to stack or triggers game over

After a successful drop:
```python
cube_positions.append([dropped_x, dropped_y, dropped_z])
current_layer += 1
```

The cube position is stored in the `cube_positions` list, and a new sliding cube is prepared at the next level.

### Task 3: Collision Detection & Failure State (30 marks)

The collision system checks whether the dropped cube has any overlap with the previous cube using a simple distance-based algorithm:

```python
def check_overlap(new_x, prev_x):
    distance = abs(new_x - prev_x)
    return distance < CUBE_WIDTH
```

**Collision Logic Explained:**

For two cubes to overlap (even partially), the distance between their centers must be less than the cube width:

```
Cube Width = 1.0 unit
Each cube: -0.5 to +0.5 from its center

If distance >= 1.0: No overlap → cube falls
If distance < 1.0: Partial overlap → stack successful
```

**Visual Example:**
```
Previous cube: Center at X = 0, spans from -0.5 to +0.5
New cube: Center at X = 0.7, spans from 0.2 to 1.2
Distance = |0.7 - 0| = 0.7 < 1.0 → OVERLAP (stacks successfully)

New cube: Center at X = 1.2, spans from 0.7 to 1.7  
Distance = |1.2 - 0| = 1.2 >= 1.0 → NO OVERLAP (falls!)
```

When a cube misses the stack:
```python
if not check_overlap(dropped_x, prev_cube_x):
    game_over = True
    falling_cube = [dropped_x, dropped_y, dropped_z]
    fall_velocity = 0.0
```

The cube is saved to `falling_cube` and animated with gravity:
```python
fall_velocity += 0.01  # Gravity acceleration
falling_cube[1] -= fall_velocity
```

This creates a realistic falling effect where the cube accelerates downward.

**Visual Feedback System:**

To help players, the active cube changes color based on alignment:
- **Bright Green:** Distance < 30% of cube width (perfect alignment)
- **Orange:** Distance between 30-70% (okay placement)  
- **Red:** Distance > 70% (very risky)

This provides real-time feedback about whether the current position is safe.

---

## 3. 3D Transformation Matrices

### Translation Matrices

The game uses translation to position each cube in 3D space. The vertex shader applies the offset:

```glsl
vec4 worldPos = vec4(position + offset, 1.0);
```

This is equivalent to multiplying by a translation matrix:

```
T = [1  0  0  x]
    [0  1  0  y]
    [0  0  1  z]
    [0  0  0  1]
```

Where (x, y, z) is the offset vector sent from Python via `glUniform3f(offset_loc, x, y, z)`.

**Why this approach?**
Instead of building full 4x4 matrices in Python for each cube, we simply add the offset in the shader. This is more efficient and achieves the same result for pure translations.

### View Transformation

The view transformation moves the world relative to the camera:

```glsl
vec3 relativePos = worldPos.xyz - cameraPos;
```

Then applies a rotation matrix for camera yaw:

```glsl
mat4 viewRotate = mat4(
     cos(-yaw), 0, sin(-yaw), 0,
     0,         1, 0,         0,
    -sin(-yaw), 0, cos(-yaw), 0,
     0,         0, 0,         1
);
```

This is a rotation matrix around the Y-axis that allows the camera to orbit the scene.

### Perspective Projection

The perspective projection matrix makes objects further away appear smaller:

```python
projection = pyrr.matrix44.create_perspective_projection_matrix(
    45.0,   # Field of view (degrees)
    1.0,    # Aspect ratio 
    0.1,    # Near clipping plane
    100.0   # Far clipping plane
)
```

This creates the depth effect that makes the game feel truly 3D. Objects further from the camera appear smaller, adding difficulty to judging distances.

**Camera Positioning:**

The camera is positioned at `(4.0, 6.0, 8.0)` with a yaw of 0.6 radians. This gives an angled view that shows:
- The top of the stack (to judge placement)
- The side of the cubes (to see depth)
- The ground plane (for context)

This perspective was chosen after experimentation to provide the best gameplay visibility.

---

## 4. Game State Management

The game uses several state variables to track progress:

```python
cube_positions = []      # All successfully stacked cubes
active_cube_pos = []     # Current sliding cube position
current_layer = 0        # Height level (score)
is_sliding = True        # Whether active cube is moving
game_over = False        # Game over flag
falling_cube = None      # Position of falling cube (if any)
fall_velocity = 0.0      # For gravity animation
```

**State Flow:**
1. **Sliding State:** Cube moves back and forth
2. **Drop Trigger:** Player presses spacebar
3. **Collision Check:** Determine if cube overlaps previous
4. **Success:** Add to stack, increment layer, prepare next cube
5. **Failure:** Set game_over flag, animate falling cube
6. **Reset:** Press 'R' to start over

---

## 5. Rendering Pipeline

The rendering happens in this order each frame:

1. **Clear buffers:** `glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)`
2. **Draw ground plane:** Dark gray base at Y = -1
3. **Draw stacked cubes:** Loop through `cube_positions`, render each in green
4. **Draw active cube:** Render with color based on alignment (if still playing)
5. **Draw falling cube:** If exists, render in red with falling animation
6. **Swap buffers:** Display the rendered frame

Each cube is rendered by:
- Binding the cube VAO (vertex array object)
- Setting the offset uniform for position
- Setting the color uniform  
- Drawing 36 vertices (6 faces × 2 triangles × 3 vertices)

---

## 6. Key Features & Enhancements

### Difficulty Progression
- Early cubes are easy because the stack is low
- As the stack grows taller, the perspective makes timing harder
- The constant slide speed maintains consistent challenge

### Visual Feedback
- Color-coded active cube shows alignment quality
- Green stacked cubes indicate success
- Red falling cube clearly shows failure
- Dark background makes cubes stand out

### Player Guidance
- Console prints score after each successful drop
- Initial instructions printed at game start
- Clear "Game Over" message with final score
- Simple reset mechanism (press 'R')

### Polish Details
- Smooth sine-wave motion feels natural
- Gravity acceleration on falling cubes looks realistic
- Depth testing ensures proper 3D rendering
- VAO/VBO setup follows modern OpenGL practices

---

## 7. Challenges & Solutions

### Challenge 1: Perspective Distortion
**Problem:** The perspective projection made it hard to judge exact positions
**Solution:** Added color-coded visual feedback showing alignment in real-time

### Challenge 2: Input Handling
**Problem:** Holding spacebar triggered multiple drops
**Solution:** Implemented debouncing with `spacebar_pressed` flag

### Challenge 3: Camera Angle
**Problem:** Finding a view that shows both top and sides of cubes
**Solution:** Experimented with different positions, settled on (4, 6, 8) with slight yaw rotation

### Challenge 4: Collision Precision
**Problem:** Needed to allow partial overlap but detect complete misses
**Solution:** Distance-based check comparing cube centers against cube width

---

## 8. Code Structure

```
main.py (370 lines)
├── Initialization (GLFW, OpenGL, Shaders)
├── Geometry Setup (Cube vertices, VAO/VBO)
├── Game State Variables  
├── Camera & Projection Setup
├── Helper Functions (reset_game, check_overlap)
└── Main Game Loop
    ├── Input Handling
    ├── Physics Updates (sliding, falling)
    ├── Collision Detection
    └── Rendering

basic.vert (Vertex Shader)
├── Model Transform (offset addition)
├── View Transform (camera relative position)
└── Projection Transform (perspective)

basic.frag (Fragment Shader)  
└── Simple color output from uniform
```

---

## 9. How to Run

```bash
# Ensure dependencies are installed
pip install glfw PyOpenGL numpy pyrr

# Run the game
python main.py
```

**Controls:**
- `SPACEBAR`: Drop the cube
- `R`: Reset the game

---

## 10. Results & Demo

The game successfully implements all required features:
- ✅ Sine-wave sliding motion  
- ✅ Interactive spacebar drop mechanism
- ✅ Collision detection with partial overlap support
- ✅ Failure state with falling animation
- ✅ 3D perspective camera view
- ✅ Visual feedback for player guidance

A video demo showing gameplay, the collision logic in action, and the 3D perspective transformation is included with this submission.

**Demo Highlights:**
- Shows successful stacking with various overlap amounts
- Demonstrates game over when cube completely misses
- Shows the color feedback system guiding player timing  
- Displays the 3D perspective effect as stack grows
- Shows reset functionality

---

## 11. Conclusion

This project demonstrates understanding of:
- 3D transformation matrices (translation, view, projection)
- Real-time input handling and state management
- Collision detection algorithms  
- OpenGL rendering pipeline (shaders, VAO/VBO, uniforms)
- Game loop architecture
- User experience design (visual feedback, controls)

The implementation is clean, well-commented, and follows modern OpenGL practices. The game is playable, challenging, and visually clear.

---

## Appendix: Key Formulas

**Sliding Motion:**
```
x(t) = sin(t × speed) × range
```

**Overlap Detection:**
```
overlap = |x₁ - x₂| < width
```

**Gravity Simulation:**
```
v(t+1) = v(t) + g
y(t+1) = y(t) - v(t+1)
```

**Perspective Division:**
```
screen_pos = projection × view × model × vertex
```

---

*End of Documentation*
