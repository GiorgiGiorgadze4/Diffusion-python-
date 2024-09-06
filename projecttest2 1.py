import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


def resize_image(image, scale=0.5):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
# Function for Perona-Malik diffusion
def perona_malik_diffusion(u, windows, kappa, delta, method):
    # approximate gradients
    nabla = [ndimage.convolve(u, w) for w in windows]

    # approximate diffusion function
    diff = [np.exp(-(n/kappa)**2) for n in nabla]

    # update image using Adams method
    if method == 'bashforth':
        terms = [diff[i] * nabla[i] for i in range(4)]
        terms += [diff[i] * nabla[i] for i in range(4, 8)]
        u_next = u + delta * sum(terms)
    elif method == 'moulton':
        terms = [diff[i] * nabla[i] for i in range(4)]
        terms += [diff[i] * nabla[i] for i in range(4, 8)]
        u_next = u + delta * 0.5 * (sum(terms) + np.sum([ndimage.convolve(u, w) for w in windows]))
    return u_next

# Function to perform stability analysis
def stability_analysis(initial_image, iterations, delta, kappa, method):
    # Create a perturbation in the initial image
    perturbed_image = initial_image + np.random.normal(0, 5, initial_image.shape)
    u_initial = initial_image.copy()
    u_perturbed = perturbed_image.copy()

    for _ in range(iterations):
        u_initial = perona_malik_diffusion(u_initial, windows, kappa, delta, method)
        u_perturbed = perona_malik_diffusion(u_perturbed, windows, kappa, delta, method)
    
    # Calculate the difference between the images after diffusion
    difference = np.abs(u_initial - u_perturbed)
    return difference

# Load the image and convert to grayscale
image_file = 'circle5.png'
original_image = cv2.imread(image_file)
resized_image = resize_image(original_image, scale=0.5)  # Resize for faster processing
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
noisy_img = gray + np.random.normal(0, 50, gray.shape)

# Define 2D finite difference windows
windows = [
    np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64),  
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64),  
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64),  
    np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64), 
    np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64),  
    np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64),  
    np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64),  
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64)   
]

# Iteration count
iterations = 2  #NOTE YOU CAN DECREASE THIS for faster computation or INcrease for more precision

# Apply Perona-Malik with Adams methods (Iteratively)
u_bashforth = noisy_img.copy()
u_moulton = noisy_img.copy()

for _ in range(iterations):
    u_bashforth = perona_malik_diffusion(u_bashforth, windows, kappa=30, delta=0.5, method='bashforth')
    u_moulton = perona_malik_diffusion(u_moulton, windows, kappa=30, delta=0.5, method='moulton')

# Stability analysis
stability_diff_bashforth = stability_analysis(gray, iterations=2, delta=0.5, kappa=30, method='bashforth') #decrease iterations
stability_diff_moulton = stability_analysis(gray, iterations=2, delta=0.5, kappa=30, method='moulton') # NOTE decrease iterations for faster compile time increase for more precision

# Visualization
plt.figure(figsize=(18, 10))
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(u_bashforth, cmap='gray')
plt.title('Adams-Bashforth')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(u_moulton, cmap='gray')
plt.title('Adams-Moulton')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(stability_diff_bashforth, cmap='gray')
plt.title('Stability (Bashforth)')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(stability_diff_moulton, cmap='gray')
plt.title('Stability (Moulton)')
plt.colorbar()
plt.axis('off')

plt.show()
