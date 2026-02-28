
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np

screenshot_saved = False

def save_screenshot():
    global screenshot_saved
    if not screenshot_saved:
        screenshot_saved = True
        # Read pixels from OpenGL framebuffer
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, 600, 600, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and reshape
        image_data = np.frombuffer(pixels, dtype=np.uint8).reshape((600, 600, 3))
        
        # Flip vertically (OpenGL reads from bottom-left)
        image_data = np.flipud(image_data)
        
        # Create PIL image and save
        img = Image.fromarray(image_data, 'RGB')
        img.save('question2.png')
        print("Question 2 screenshot saved as question2.png")

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Draw black square outline (from -1,-1 to 1,1)
    glBegin(GL_LINE_LOOP)
    glColor3f(1.0, 1.0, 1.0)  # White outline
    glVertex2f(-1.0, -1.0)
    glVertex2f(1.0, -1.0)
    glVertex2f(1.0, 1.0)
    glVertex2f(-1.0, 1.0)
    glEnd()
    
    # Draw green small square
    # One corner at center (0, 0), opposite corner at bottom-right (1, -1)
    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex2f(0.0, 0.0)      # Top-left (at center)
    glVertex2f(1.0, 0.0)      # Top-right
    glVertex2f(1.0, -1.0)     # Bottom-right
    glVertex2f(0.0, -1.0)     # Bottom-left
    glEnd()
    
    # Draw red right-angle triangle at top-left corner (drawn after square so it's visible)
    # Right angle at (-1, 1), other vertices at (-1, 0) and (0, 1)
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex2f(-1.0, 1.0)    # Right angle vertex (top-left)
    glVertex2f(-1.0, 0.0)    # Bottom of triangle
    glVertex2f(0.0, 1.0)     # Right of triangle
    glEnd()
    
    glFlush()
    save_screenshot()
    import sys
    sys.exit(0)


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"Question 2: Square with Triangle and Green Square")
    
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
    gluOrtho2D(-1, 1, -1, 1)
    glutDisplayFunc(display)
    glutMainLoop()


if __name__ == "__main__":
    main()
