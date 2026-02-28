#version 410 core

// Output color of the pixel
out vec4 color;

// Uniform to receive color data from the Python game logic
uniform vec3 cubeColor;

void main()
{
    // Sets the final pixel color (R, G, B) with full opacity (1.0)
    color = vec4(cubeColor, 1.0);
}