import ctypes
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import math

# --- BOILERPLATE INITIALIZATION ---
glfw.init()
# Request OpenGL 4.1 Core Profile for macOS compatibility
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
window = glfw.create_window(1000, 1000, "Assignment 3: 3D Stacker Game", None, None)
glfw.make_context_current(window)
glEnable(GL_DEPTH_TEST)
glClearColor(0.1, 0.1, 0.15, 1.0)

# Load Shaders
vertex_shader = open("basic.vert").read()
fragment_shader = open("basic.frag").read()

# Compile shaders without validation (macOS compatibility)
shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    validate=False  # Skip validation until VAO is bound
)
glUseProgram(shader)

# --- GEOMETRY SETUP: CUBE VERTICES ---
# Each cube is 1.0 unit wide, centered at origin
cube_vertices = np.array([
    # Front face
    -0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    # Back face
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,
     0.5,  0.5, -0.5,
    -0.5,  0.5, -0.5,
    # Left face
    -0.5, -0.5, -0.5,
    -0.5, -0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5, -0.5, -0.5,
    -0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
    # Right face
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
     0.5, -0.5, -0.5,
     0.5,  0.5,  0.5,
     0.5,  0.5, -0.5,
    # Top face
    -0.5,  0.5, -0.5,
     0.5,  0.5, -0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    # Bottom face
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,
], dtype=np.float32)

# Ground plane vertices
plane_vertices = np.array([
    -5.0, 0.0, -5.0,
     5.0, 0.0, -5.0,
     5.0, 0.0,  5.0,
    -5.0, 0.0, -5.0,
     5.0, 0.0,  5.0,
    -5.0, 0.0,  5.0,
], dtype=np.float32)

# Setup VAO and VBO for cube
cube_vao = glGenVertexArrays(1)
cube_vbo = glGenBuffers(1)
glBindVertexArray(cube_vao)
glBindBuffer(GL_ARRAY_BUFFER, cube_vbo)
glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# Setup VAO and VBO for plane
plane_vao = glGenVertexArrays(1)
plane_vbo = glGenBuffers(1)
glBindVertexArray(plane_vao)
glBindBuffer(GL_ARRAY_BUFFER, plane_vbo)
glBufferData(GL_ARRAY_BUFFER, plane_vertices.nbytes, plane_vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# --- GAME STATE VARIABLES ---
cube_positions = []  # List of (x, y, z) tuples for dropped cubes
active_cube_pos = [0.0, 0.5, 0.0]  # Starting position of the sliding cube
is_sliding = True  # Whether the active cube is still moving
current_layer = 0  # Current height level
game_over = False  # Game over flag
falling_cube = None  # Position of cube that's falling (for animation)
fall_velocity = 0.0  # Vertical velocity for falling animation

# Starting platform (visual reference at ground level)
starting_platform_pos = [0.0, -0.5, 0.0]

# Game parameters
CUBE_WIDTH = 1.0  # Width of each cube
SLIDE_SPEED = 1.5  # Speed of horizontal sliding
SLIDE_RANGE = 3.0  # How far the cube slides left-right
LAYER_HEIGHT = 1.0  # Vertical spacing between cubes

# Key press tracking to prevent holding spacebar
spacebar_pressed = False

# --- UNIFORM LOCATIONS ---
model_loc = glGetUniformLocation(shader, "model")
view_loc = glGetUniformLocation(shader, "view")
projection_loc = glGetUniformLocation(shader, "projection")
color_loc = glGetUniformLocation(shader, "cubeColor")

# --- CAMERA AND PERSPECTIVE SETUP ---
# Camera offset from the look-at target
camera_offset = pyrr.Vector3([5.0, 3.0, 10.0])
camera_up = pyrr.Vector3([0.0, 1.0, 0.0])

# Perspective projection matrix (FOV, aspect ratio, near, far)
projection = pyrr.matrix44.create_perspective_projection_matrix(
    45.0,  # Field of view
    1.0,   # Aspect ratio (square window)
    0.1,   # Near clipping plane
    100.0  # Far clipping plane
)
glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

# --- HELPER FUNCTIONS ---
def reset_game():
    """Reset the game to initial state"""
    global cube_positions, active_cube_pos, current_layer, is_sliding, game_over, falling_cube, fall_velocity
    cube_positions = []
    active_cube_pos = [0.0, 0.5, 0.0]
    current_layer = 0
    is_sliding = True
    game_over = False
    falling_cube = None
    fall_velocity = 0.0
    print("Game Reset! Score: 0")

def check_overlap(new_x, prev_x):
    """
    Check if the new cube overlaps with the previous cube
    Returns True if there's any overlap, False if completely missed
    """
    # Calculate the distance between centers
    distance = abs(new_x - prev_x)
    # If the distance is greater than cube width, no overlap
    return distance < CUBE_WIDTH

# --- MAIN RENDER LOOP ---
print("=== 3D STACKER GAME ===")
print("Press SPACEBAR to drop the cube")
print("Stack as many cubes as you can!")
print("Press 'R' to restart anytime\n")

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore
    
    current_time = glfw.get_time()
    
    # Check for reset key
    if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
        reset_game()

    # --- TASK 1: SLIDING LOGIC ---
    # Use sine wave to create smooth back-and-forth motion
    if is_sliding and not game_over:
        active_cube_pos[0] = math.sin(current_time * SLIDE_SPEED) * SLIDE_RANGE
        active_cube_pos[1] = 0.5 + current_layer * LAYER_HEIGHT
        active_cube_pos[2] = 0.0
    
    # Update camera to follow the stack as it grows
    # Look at point moves upward with the active cube
    target_height = active_cube_pos[1] if not game_over else (cube_positions[-1][1] if cube_positions else 0.0)
    camera_target = pyrr.Vector3([0.0, target_height, 0.0])
    camera_pos = camera_target + camera_offset
    
    # Update view matrix each frame to follow the stack
    view = pyrr.matrix44.create_look_at(camera_pos, camera_target, camera_up)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    
    # --- TASK 2: DROP INPUT HANDLING ---
    # Detect spacebar press (with debouncing to prevent holding)
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and not spacebar_pressed:
        spacebar_pressed = True
        
        if is_sliding and not game_over:
            # Freeze the current X position
            dropped_x = active_cube_pos[0]
            dropped_y = active_cube_pos[1]
            dropped_z = active_cube_pos[2]
            
            # --- TASK 3: COLLISION & STACK CHECK ---
            # Check if this cube overlaps with the previous one
            can_stack = True
            
            if current_layer > 0:
                # Get the previous cube's X position
                prev_cube_x = cube_positions[-1][0]
                
                # Check for overlap
                if not check_overlap(dropped_x, prev_cube_x):
                    # No overlap - cube falls!
                    can_stack = False
                    game_over = True
                    falling_cube = [dropped_x, dropped_y, dropped_z]
                    fall_velocity = 0.0
                    print(f"GAME OVER! Final Score: {current_layer}")
                    print("Press 'R' to restart")
            else:
                # First cube - check against starting platform
                if not check_overlap(dropped_x, starting_platform_pos[0]):
                    # Missed the platform!
                    can_stack = False
                    game_over = True
                    falling_cube = [dropped_x, dropped_y, dropped_z]
                    fall_velocity = 0.0
                    print(f"GAME OVER! Missed the platform!")
                    print("Press 'R' to restart")
            
            if can_stack:
                # Successfully stacked!
                cube_positions.append([dropped_x, dropped_y, dropped_z])
                current_layer += 1
                print(f"Score: {current_layer}")
                
                # Prepare next cube at higher position
                active_cube_pos[1] = 0.5 + current_layer * LAYER_HEIGHT
            
    elif glfw.get_key(window, glfw.KEY_SPACE) == glfw.RELEASE:
        spacebar_pressed = False
    
    # --- HANDLE FALLING CUBE ANIMATION ---
    if falling_cube is not None:
        fall_velocity += 0.01  # Gravity acceleration
        falling_cube[1] -= fall_velocity
        
        # Stop animating when cube falls below ground
        if falling_cube[1] < -5.0:
            falling_cube = None
    
    # --- RENDERING ---
    # Draw ground plane
    glBindVertexArray(plane_vao)
    glUniform3f(color_loc, 0.3, 0.3, 0.35)  # Dark gray ground
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, -1.0, 0.0]))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    
    # Draw starting platform (visual reference)
    glBindVertexArray(cube_vao)
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3(starting_platform_pos))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniform3f(color_loc, 0.4, 0.4, 0.45)  # Slightly lighter gray for platform
    glDrawArrays(GL_TRIANGLES, 0, 36)
    
    # Draw all stacked cubes
    for pos in cube_positions:
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniform3f(color_loc, 0.2, 0.8, 0.3)  # Green for successfully stacked cubes
        glDrawArrays(GL_TRIANGLES, 0, 36)
    
    # Draw the active sliding cube (if still playing)
    if is_sliding and not game_over:
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(active_cube_pos))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        
        # Change color based on alignment (visual feedback)
        if current_layer > 0:
            prev_x = cube_positions[-1][0]
            distance = abs(active_cube_pos[0] - prev_x)
        else:
            # First cube - check against starting platform
            prev_x = starting_platform_pos[0]
            distance = abs(active_cube_pos[0] - prev_x)
        
        # Visual hint: green when close, orange when risky, red when too far
        if distance < CUBE_WIDTH * 0.3:
            glUniform3f(color_loc, 0.2, 1.0, 0.2)  # Bright green - perfect
        elif distance < CUBE_WIDTH * 0.7:
            glUniform3f(color_loc, 1.0, 0.7, 0.0)  # Orange - okay
        else:
            glUniform3f(color_loc, 1.0, 0.2, 0.2)  # Red - risky
        
        glDrawArrays(GL_TRIANGLES, 0, 36)
    
    # Draw falling cube if exists
    if falling_cube is not None:
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(falling_cube))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniform3f(color_loc, 1.0, 0.2, 0.2)  # Red for falling cube
        glDrawArrays(GL_TRIANGLES, 0, 36)
    
    glfw.swap_buffers(window)

glfw.terminate()