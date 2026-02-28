
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import os

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
        img.save('question1.png')
        print("Question 1 screenshot saved as question1.png")

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Draw a filled triangle with color gradient from red (left) to green (right)
    glBegin(GL_TRIANGLES)
    
    # Bottom-left vertex - Red
    glColor3f(1.0, 0.0, 0.0)
    glVertex2f(-0.5, -0.4)
    
    # Top vertex - Green
    glColor3f(0.0, 1.0, 0.0)
    glVertex2f(0.0, 0.5)
    
    # Bottom-right vertex - Green
    glColor3f(0.0, 1.0, 0.0)
    glVertex2f(0.5, -0.4)
    
    glEnd()
    
    glFlush()
    save_screenshot()
    import sys
    sys.exit(0)


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"Question 1: Filled Triangle")
    
    glClearColor(0.2, 0.4, 0.7, 1.0)  # Blue background
    gluOrtho2D(-1, 1, -1, 1)
    glutDisplayFunc(display)
    glutMainLoop()


if __name__ == "__main__":
    main()
