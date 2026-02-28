"""
Script to generate PNG screenshots for the assignments
"""
from PIL import Image, ImageDraw

# Question 1: Filled Triangle with Color Gradient on Blue Background
# Create image with blue background
img1 = Image.new('RGB', (600, 600), color=(51, 102, 179))  # Blue background
draw1 = ImageDraw.Draw(img1, 'RGBA')

# Draw a triangle with color gradient
# Vertices: bottom-left (red), top (green), bottom-right (yellow)
# Using barycentric coordinates for smooth gradient

for x in range(600):
    for y in range(600):
        # Triangle vertices (in pixel coordinates)
        x1, y1 = 150, 450  # Bottom-left (Red)
        x2, y2 = 300, 150  # Top (Green)
        x3, y3 = 450, 450  # Bottom-right (Yellow)
        
        # Calculate barycentric coordinates
        denom = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        if denom == 0:
            continue
            
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1 - a - b
        
        # Check if point is inside triangle
        if a >= 0 and b >= 0 and c >= 0:
            # Interpolate colors
            r = int(a * 255 + c * 255)  # Red at vertex 1, Yellow at vertex 3
            g = int(b * 255 + c * 255)  # Green at vertex 2, Yellow at vertex 3
            b_val = int(0)  # No blue in the gradient
            
            r = min(255, r)
            g = min(255, g)
            
            img1.putpixel((x, y), (r, g, b_val, 255))

img1.save('/Users/ashishdate/Documents/IITJ/4th year/CG_1/question1.png')
print("Question 1 screenshot saved")

# Question 2: Black square with red triangle and green small square
img2 = Image.new('RGB', (600, 600), color=(0, 0, 0))  # Black background
draw2 = ImageDraw.Draw(img2, 'RGBA')

# Draw black square outline (white for visibility)
draw2.rectangle([(0, 0), (599, 599)], outline=(255, 255, 255), width=2)

# Draw red right-angle triangle at top-left
# Triangle vertices in pixel coordinates: (0,0), (0,300), (300,0)
triangle_coords = [(0, 0), (0, 300), (300, 0)]
draw2.polygon(triangle_coords, fill=(255, 0, 0))

# Draw green small square
# From (0, 0) to (300, 300)
draw2.rectangle([(0, 0), (299, 299)], fill=(0, 255, 0))

img2.save('/Users/ashishdate/Documents/IITJ/4th year/CG_1/question2.png')
print("Question 2 screenshot saved")

print("Both PNG files created successfully!")
