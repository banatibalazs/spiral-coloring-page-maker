import numpy as np
import cv2


IMAGE_PATH = 'images/img.png'
SAVING_PATH = 'images/spiral.jpg'


def resize_image_with_aspect_ratio(image, width=None, height=None):
    h, w = image.shape[:2]
    dim = None
    if width is not None:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    elif height is not None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def get_outline_image(image):
    # Find the contours of the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank image
    outline_image = np.zeros_like(image)
    # Draw the contours on the blank image
    cv2.drawContours(outline_image, contours, -1, 255, 1)
    return cv2.bitwise_not(outline_image), contours


def get_spiral_image(num_rotations, scale_factor, width_factor, max_width, image, mode, resolution=4000):

    points_per_45_degrees = 100 / num_rotations
    theta = np.concatenate([np.linspace(np.pi*i/4, np.pi*(i+1)/4, int(points_per_45_degrees * (0.5 + (i/14)))) for i in range(num_rotations*8)])

    r = theta.copy()
    # Generate the spiral coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Normalize the coordinates to the range [0, 1]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Scale the coordinates to the desired image size
    x = (x * resolution).astype(np.int_) - 1
    y = (y * resolution).astype(np.int_) - 1

    # Create a blank image
    spiral_image = np.zeros((resolution, resolution, 1), dtype=np.uint8)
    resized_image = cv2.resize(image, (resolution, resolution))

    # Draw the spiral on the image
    for i in range(1, len(x)):
        color = resized_image[y[i], x[i]]
        width = scale_factor * (1 - (color / 255)) + width_factor # Calculate the width based on the grayscale value
        if width < 1:
            width = 1
        if mode == 0:
            cv2.line(spiral_image, (x[i-1], y[i-1]), (x[i], y[i]), 255, thickness=min(int(width),max_width))
        else:
            cv2.circle(spiral_image, (x[i], y[i]), 1, 255, thickness=min(int(width),max_width))

    spiral_image = cv2.bitwise_not(spiral_image)
    return spiral_image

# Create a window
cv2.namedWindow('Spiral')
cv2.createTrackbar('Rotations', 'Spiral', 45, 50, lambda x: None)
cv2.createTrackbar('Contrast', 'Spiral', 25, 70, lambda x: None)
cv2.createTrackbar('Line Width', 'Spiral', 20, 250, lambda x: None)
cv2.createTrackbar('Max Width', 'Spiral', 20, 250, lambda x: None)
cv2.createTrackbar('Resolution', 'Spiral', 3000, 3600, lambda x: None)
cv2.createTrackbar('Mode', 'Spiral', 0, 1, lambda x: None)


# Load the original image
original_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
original_width, original_height = original_img.shape
original_aspect_ratio = original_height / original_width

while True:
    # Get the current positions of the trackbars
    num_rotations = cv2.getTrackbarPos('Rotations', 'Spiral') + 1
    scale_factor = cv2.getTrackbarPos('Contrast', 'Spiral') + 1
    width_factor = cv2.getTrackbarPos('Line Width', 'Spiral') + 1
    max_width = cv2.getTrackbarPos('Max Width', 'Spiral') + 1
    resolution = cv2.getTrackbarPos('Resolution', 'Spiral') + 400
    mode = cv2.getTrackbarPos('Mode', 'Spiral')

    # Calculate the width as inversely proportional to the number of rotations
    width = width_factor * 5 / num_rotations
    orig_width, orig_height = original_img.shape
    w_ratio = resolution / orig_width
    h_ratio = resolution / orig_height

    # Draw the spiral with the current trackbar positions
    spiral_img = get_spiral_image(num_rotations, scale_factor, width, max_width, original_img, mode, resolution)
    outline_image, contours = get_outline_image(spiral_img)

    h, w = spiral_img.shape[:2]
    spiral_img = cv2.resize(spiral_img, (int(h * original_aspect_ratio), w))
    outline_image = cv2.resize(outline_image, (int(h * original_aspect_ratio), w))

    cv2.imshow('Preview Outline Image', resize_image_with_aspect_ratio(outline_image, height=800))
    cv2.imshow('Spiral', resize_image_with_aspect_ratio(spiral_img, height=800))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):

        # Save the images
        cv2.imwrite(SAVING_PATH, spiral_img)
        cv2.imwrite(SAVING_PATH[0:-4] + '_outline.jpg', outline_image)
        break


# Save the image
cv2.destroyAllWindows()