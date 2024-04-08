from PIL import Image

# Open the image
image_path = "C:\\Users\\tomng\\Desktop\\logo_toronto_black.jpg"
image = Image.open(image_path)

# Convert the image to RGB mode
image = image.convert("RGB")

# Define the color to keep and the color to change other pixels to
keep_color = (53, 53, 65)
black = (255, 255, 255)

# Get the image dimensions
width, height = image.size

# Iterate over each pixel and check its color
for x in range(width):
    for y in range(height):
        pixel_color = image.getpixel((x, y))

        sqr = 0
        for i in pixel_color:
            sqr += i**2
        sqr = sqr ** (1/2)

        print(pixel_color, sqr)

        if sqr < 60:
            image.putpixel((x, y), black)  # Set pixel color to black

# Save the modified image
output_path = "C:\\Users\\tomng\\Desktop\\logo_toronto.jpg"
image.save(output_path)

print("Image processing complete.")
