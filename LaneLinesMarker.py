import matplotlib.image as mpimg
import numpy as np
import cv2
import os


# const values
DEFAULT_LOW_CANNY_THRESHOLD = 50
DEFAULT_THICKNESS = 5
LEFT_BOUNDARY_RATIO = .104
PROCESSED_FILE = "processed_images/%s"
REGION_OF_INTEREST_LEFT_BOUND_RATIO = .458
REGION_OF_INTEREST_RIGHT_BOUND_RATIO = .542
REGION_OF_INTEREST_HEIGHT_RATIO = .611
RGB_RED = [255, 0, 0]
RIGHT_BOUNDARY_RATIO = .958
SHAPE_X_INDEX = 1
SHAPE_Y_INDEX = 0
TEST_FILE = "test_images/%s"
TEST_IMAGE_DIR = "test_images/"
THRESHOLD_RATIO = 3


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=RGB_RED, thickness=DEFAULT_THICKNESS):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Method for drawing extrapolated lane lines. It includes the following additional attributes compared to "draw_lines":
# width: The width of the image
# bottom_height: The Y value at the bottom of the image (Y height)
# top_height: The Y value at the top of the region of interest trapezoid.
def draw_extrapolated_lines(img, lines, width, bottom_height, top_height, color=RGB_RED, thickness=DEFAULT_THICKNESS):
    middle = int(width / 2)
    left_array = []
    right_array = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # There should be no instance where a line segment crosses between the middle of the image
            if x1 < middle:
                left_array.append((x1, y1))
                left_array.append((x2, y2))
            else:
                right_array.append((x1, y1))
                right_array.append((x2, y2))

    # Generate a y=mx+b formula
    left_formula = np.polyfit(np.array(left_array)[:, 0], np.array(left_array)[:, 1], 1)
    right_formula = np.polyfit(np.array(right_array)[:, 0], np.array(right_array)[:, 1], 1)

    left_m = left_formula[0]
    left_b = left_formula[1]

    right_m = right_formula[0]
    right_b = right_formula[1]

    # Calculate the x positions based on the parameter bottom / top heights.
    left_x_bottom = (bottom_height - left_b) / left_m
    left_x_top = (top_height - left_b) / left_m
    right_x_bottom = (bottom_height - right_b) / right_m
    right_x_top = (top_height - right_b) / right_m

    # Draw a line from the bottom of the image to the specified maximum height
    cv2.line(img, (int(left_x_bottom), bottom_height), (int(left_x_top), top_height), color, thickness)
    cv2.line(img, (int(right_x_bottom), bottom_height), (int(right_x_top), top_height), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, width, bottom_height, top_height):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_extrapolated_lines(line_img, lines, width, bottom_height, top_height)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Helper method for processing the image and applying the lines
def process_image(image):
    copied_image = np.copy(image)
    ysize = image.shape[SHAPE_Y_INDEX]
    xsize = image.shape[SHAPE_X_INDEX]

    # Apply grayscale, blur and canny edges
    gray = grayscale(copied_image)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, DEFAULT_LOW_CANNY_THRESHOLD, DEFAULT_LOW_CANNY_THRESHOLD*THRESHOLD_RATIO)

    leftmost = int(xsize * LEFT_BOUNDARY_RATIO)
    rightmost = int(xsize * RIGHT_BOUNDARY_RATIO)
    region_height = int(ysize * REGION_OF_INTEREST_HEIGHT_RATIO)
    region_left = int(xsize * REGION_OF_INTEREST_LEFT_BOUND_RATIO)
    region_right = int(xsize * REGION_OF_INTEREST_RIGHT_BOUND_RATIO)

    # Using a trapezoid instead of a triangle allows more evenness between
    # the left and right side lines across a video clip.
    trapezoid_array = np.array([[leftmost, ysize], [region_left, region_height], [region_right, region_height], [rightmost, ysize]])
    road_edges = region_of_interest(edges, [trapezoid_array])

    # Run Hough transform and apply line to the copied image
    line_drawn_image = hough_lines(road_edges, 1, np.pi/180, 15, 10, 20, xsize, ysize, region_height)
    result = weighted_img(line_drawn_image, copied_image)

    return result


#
# MAIN FUNCTION
#
files = os.listdir(TEST_IMAGE_DIR)
save_image = True
for file_name in files:
    # Retrieve image and get x/y size values
    filePath = TEST_FILE % file_name
    image = mpimg.imread(filePath)

    # Save image into the processed_image folder
    final_image = process_image(image)
    final_file_name = PROCESSED_FILE % file_name
    mpimg.imsave(final_file_name, final_image)



