#import os
import matplotlib.pyplot as plt
import pywt
import cv2  # Import OpenCV
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from scipy.signal import convolve2d



#------------------------------------------------------------------------------------------------------------------------------------------

def calculate_psnr(original_img, processed_img, max_pixel_value):
    mse = np.mean((original_img - processed_img) ** 2)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr




#Image Opening


# Define file paths for right and left eye images
right_eye_path = "D:\\Productivity\\Coding\\VSCode\\VSCode Projects\\Python Projects\\sean_righteye.jpeg"
left_eye_path = "D:\\Productivity\\Coding\\VSCode\\VSCode Projects\\Python Projects\\sean_lefteye.jpeg"


#------------------------------------------------------------------------------------------------------------------------------------------
#Sharpening Filters

# Unsharp Masking Function
def apply_unsharp_masking(image, kernel_size=(5, 5), sigma=0.5, alpha=2.5, beta=-1.5):
    # Apply Gaussian blur to create a blurred version of the image
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Calculate the unsharp mask
    unsharp = cv2.addWeighted(image, alpha, blurred, beta, 0)
    
    return unsharp


#------------------------------------------------------------------------------------------------------------------------------------------

#Wavelet Sharpen Function

def wavelet_sharpen(image):
    # Apply a wavelet transform to the image
    coeffs2 = pywt.dwt2(image, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    
    # Sharpen the LH, HL, and HH coefficients
    sharpened_LH = LH * 3.0  # You can adjust the sharpening factor
    sharpened_HL = HL * 3.0
    sharpened_HH = HH * 3.0
    
    # Perform inverse wavelet transform
    coeffs2_sharpened = LL, (sharpened_LH, sharpened_HL, sharpened_HH)
    sharpened_image = pywt.idwt2(coeffs2_sharpened, 'bior1.3')
    
    return sharpened_image



#------------------------------------------------------------------------------------------------------------------------------------------

# Kernel Function (e.g., a circular kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


#------------------------------------------------------------------------------------------------------------------------------------------

#Gamma Function

def adjust_gamma(image, gamma=1.0):
    # Ensure the image is of type uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)



#------------------------------------------------------------------------------------------------------------------------------------------

# Normalization Function

def normalize_image(image, min_value=0, max_value=255):
    image_min = np.min(image)
    image_max = np.max(image)
    
    # Normalize the image
    normalized_image = min_value + ((image - image_min) / (image_max - image_min)) * (max_value - min_value)
    
    return normalized_image


#------------------------------------------------------------------------------------------------------------------------------------------
#Noise Reduction Functions

# Gaussian Denoising Function

def gaussian_filter (image, kernel_size=(5, 5), sigma=0.5):
    return cv2.GaussianBlur(image, kernel_size, sigma)


#Bilateral Filter Function

def bilateral_filter(image):
    # Parameters: diameter, sigmaColor, sigmaSpace
    filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    return filtered_image


#------------------------------------------------------------------------------------------------------------------------------------------

# Wavelet Denoising Function

def wavelet_denoising(image):
    # Wavelet transformation (you can adjust the wavelet family and mode)
    coeffs2 = pywt.dwt2(image, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2

    # Thresholding (you can experiment with different thresholding methods)
    threshold = 0.05  # Adjust the threshold value
    LH_thr = pywt.threshold(LH, threshold, mode='soft')
    HL_thr = pywt.threshold(HL, threshold, mode='soft')
    HH_thr = pywt.threshold(HH, threshold, mode='soft')

    # Inverse wavelet transform
    coeffs2_thr = LL, (LH_thr, HL_thr, HH_thr)
    denoised_image = pywt.idwt2(coeffs2_thr, 'bior1.3')

    return denoised_image


#------------------------------------------------------------------------------------------------------------------------------------------

# LoG Filter Function (Laplacian of Gaussian)
def apply_log(image, kernel_size):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Apply Laplacian (LoG)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    
    return log



#------------------------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------------------------

#Opening Set Images

# Open the right eye image
right_eye_image = Image.open(right_eye_path)

# Convert the right eye image to a NumPy array for processing
right_eye_array = np.array(right_eye_image)

# Open the left eye image
left_eye_image = Image.open(left_eye_path)

# Convert the left eye image to a NumPy array for processing
left_eye_array = np.array(left_eye_image)


#------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------
#Applying Denoising Functions


# Apply Median Filter for noise reduction to the right eye
right_eye_median = cv2.medianBlur(right_eye_array, 5)  

# Apply Median Filter for noise reduction to the left eye
left_eye_median = cv2.medianBlur(left_eye_array, 5)  


# Apply bilateral filter to the right eye image
right_eye_bilateral = bilateral_filter(right_eye_median)
left_eye_bilateral = bilateral_filter(left_eye_median)


# Apply wavelet denoising to the right eye image
right_eye_denoised = wavelet_denoising(right_eye_bilateral)

# Apply wavelet denoising to the left eye image
left_eye_denoised = wavelet_denoising(left_eye_bilateral)


#------------------------------------------------------------------------------------------------------------------------------------------

# Convert the denoised images to 8-bit depth
right_eye_8bit = cv2.convertScaleAbs(right_eye_denoised)
left_eye_8bit = cv2.convertScaleAbs(left_eye_denoised)

# Convert the filtered right eye image to grayscale
right_eye_gray = cv2.cvtColor(right_eye_8bit, cv2.COLOR_BGR2GRAY)

# Convert the filtered left eye image to grayscale
left_eye_gray = cv2.cvtColor(left_eye_8bit, cv2.COLOR_BGR2GRAY)


#------------------------------------------------------------------------------------------------------------------------------------------
#Contrast Adjustment

# Create CLAHE objects
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE to enhance contrast for the right eye
right_eye_clahe = clahe.apply(right_eye_gray)

# Apply CLAHE to enhance contrast for the left eye
left_eye_clahe = clahe.apply(left_eye_gray)


#------------------------------------------------------------------------------------------------------------------------------------------
#Sharpening Filters

# Apply Unsharp Masking to the right eye image
right_eye_unsharp = apply_unsharp_masking(right_eye_clahe)

# Apply Unsharp Masking to the left eye image
left_eye_unsharp = apply_unsharp_masking(left_eye_clahe)



# Apply wavelet-based sharpening to the right eye image
right_eye_wavesharp = wavelet_sharpen(right_eye_unsharp)

# Apply wavelet-based sharpening to the left eye image
left_eye_wavesharp = wavelet_sharpen(left_eye_unsharp)


#------------------------------------------------------------------------------------------------------------------------------------------
#Thresgholding to see Edges

# Apply erosion to the right eye image
right_eye_eroded = cv2.erode(right_eye_wavesharp, kernel, iterations=2)

# Apply erosion to the left eye image
left_eye_eroded = cv2.erode(left_eye_wavesharp, kernel, iterations=2)


#------------------------------------------------------------------------------------------------------------------------------------------
#Gamma Correction

# Apply Gamma Correction to the right eye image
gamma_value = 1  # You can adjust this value
right_eye_gamma_corrected = adjust_gamma(right_eye_eroded, gamma_value)

# Apply Gamma Correction to the left eye image
left_eye_gamma_corrected = adjust_gamma(left_eye_eroded, gamma_value)


#------------------------------------------------------------------------------------------------------------------------------------------
#Normalization

# Apply pixel value normalization to the right eye image
right_eye_normalized = normalize_image(right_eye_gamma_corrected)

# Apply pixel value normalization to the left eye image
left_eye_normalized = normalize_image(left_eye_gamma_corrected)


#------------------------------------------------------------------------------------------------------------------------------------------


# Convert the normalized images to 8-bit unsigned integer (CV_8U) data type
right_eye_normalized_8u = cv2.convertScaleAbs(right_eye_gamma_corrected)
left_eye_normalized_8u = cv2.convertScaleAbs(left_eye_gamma_corrected)

'''# Apply Canny edge detection to the right eye image
right_eye_canny = cv2.Canny(right_eye_normalized_8u, 25, 80)  # You can adjust the threshold values

# Apply Canny edge detection to the left eye image
left_eye_canny = cv2.Canny(left_eye_normalized_8u, 25, 80)  # You can adjust the threshold values
'''


#------------------------------------------------------------------------------------------------------------------------------------------
#LoG Filter

# Apply LoG filter to the right eye image
right_eye_log = apply_log(right_eye_normalized, kernel_size=61)

# Apply LoG filter to the left eye image
left_eye_log = apply_log(left_eye_normalized, kernel_size=61)


# Apply Gaussian denoising to the right eye image
right_eye_loggaussian = gaussian_filter(right_eye_log, kernel_size=(5, 5), sigma=0.5)

# Apply Gaussian denoising to the left eye image
left_eye_loggaussian = gaussian_filter(left_eye_log, kernel_size=(5, 5), sigma=0.5)




#Threshholding to Define Edges

# Define a threshold value (adjust as needed)
threshold_value = 0.05  # Adjust this threshold value based on your requirements

# Apply thresholding
_, right_eye_thresholded = cv2.threshold(right_eye_loggaussian, threshold_value, 255, cv2.THRESH_BINARY)
_, left_eye_thresholded = cv2.threshold(left_eye_loggaussian, threshold_value, 255, cv2.THRESH_BINARY)



#------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------
#Displaying Images
'''
# Create subplots for original and enhanced images for right and left eyes
plt.figure(figsize=(12, 9))

# Display the original right eye image
plt.subplot(3, 2, 1)
plt.imshow(right_eye_array)
plt.title("Right Eye - Original")
plt.axis("off")

# Display the enhanced right eye image
plt.subplot(3, 2, 3)
plt.imshow(right_eye_normalized_8u, cmap='gray')
plt.title("Right Eye - Enhanced")
plt.axis("off")

# Display the enhanced right eye image with Frangi filter
plt.subplot(3, 2, 5)
plt.imshow(right_eye_thresholded, cmap='gray')
plt.title("Right Eye - Enhanced & Log Thresholded")
plt.axis("off")


#------------------------------------------------------------------------------------------------------------------------------------------


# Display the original left eye image
plt.subplot(3, 2, 2)
plt.imshow(left_eye_array)
plt.title("Left Eye - Original")
plt.axis("off")

# Display the enhanced left eye image
plt.subplot(3, 2, 4)
plt.imshow(left_eye_normalized, cmap='gray')
plt.title("Left Eye - Enhanced")
plt.axis("off")

# Display the enhanced left eye image with Frangi filter
plt.subplot(3, 2, 6)
plt.imshow(left_eye_thresholded, cmap='gray')
plt.title("Left Eye - Enhanced & Log Thresholded")
plt.axis("off")

'''

# Create subplots for the right eye with corrected order of techniques
plt.figure(figsize=(12, 9))

# Display the original right eye image
plt.subplot(4, 4, 1)
plt.imshow(left_eye_array)
plt.title("1. Original (Left Eye)")
plt.axis("off")

# Display the right eye image after Median Filter
plt.subplot(4, 4, 2)
plt.imshow(left_eye_median, cmap='gray')
plt.title("2. Median Filter")
plt.axis("off")

# Display the right eye image after Bilateral Filter
plt.subplot(4, 4, 3)
plt.imshow(left_eye_bilateral, cmap='gray')
plt.title("3. Bilateral Filter")
plt.axis("off")

# Ensure the image is of type uint8 and clip the values to the valid range
left_eye_denoised = np.clip(left_eye_denoised, 0, 255).astype(np.uint8)
# Display the right eye image after Wavelet Denoising
plt.subplot(4, 4, 4)
plt.imshow(left_eye_denoised, cmap='gray')
plt.title("4. Wavelet Denoising")
plt.axis("off")

# Display the right eye image after CLAHE
plt.subplot(4, 4, 5)
plt.imshow(left_eye_clahe, cmap='gray')
plt.title("5. CLAHE")
plt.axis("off")

# Display the right eye image after Unsharp Masking
plt.subplot(4, 4, 6)
plt.imshow(left_eye_unsharp, cmap='gray')
plt.title("6. Unsharp")
plt.axis("off")

# Display the right eye image after Wavelet Sharpening
plt.subplot(4, 4, 7)
plt.imshow(left_eye_wavesharp, cmap='gray')
plt.title("7. Wavelet Sharpening")
plt.axis("off")

# Display the right eye image after Erosion
plt.subplot(4, 4, 8)
plt.imshow(left_eye_eroded, cmap='gray')
plt.title("8. Erosion")
plt.axis("off")

# Display the right eye image after Gamma Correction
plt.subplot(4, 4, 9)
plt.imshow(left_eye_gamma_corrected, cmap='gray')
plt.title("9. Gamma Correction")
plt.axis("off")

# Display the right eye image after Normalization
plt.subplot(4, 4, 10)
plt.imshow(left_eye_normalized, cmap='gray')
plt.title("10. Normalization")
plt.axis("off")

# Display the right eye image after LoG filter
plt.subplot(4, 4, 11)
plt.imshow(left_eye_log, cmap='gray')
plt.title("11. LoG Filter")
plt.axis("off")

# Display the right eye image after Log Thresholding
plt.subplot(4, 4, 12)
plt.imshow(left_eye_thresholded, cmap='gray')
plt.title("12. Log Thresholded (Left Eye)")
plt.axis("off")


# Create subplots for the right eye and left eye - Normalization and Thresholding
plt.figure(figsize=(12, 12))

# Display the right eye image after Normalization
plt.subplot(2, 2, 1)
plt.imshow(right_eye_normalized, cmap='gray')
plt.title("Right Eye - Normalization")
plt.axis("off")

# Display the right eye image after Thresholding
plt.subplot(2, 2, 2)
plt.imshow(right_eye_thresholded, cmap='gray')
plt.title("Right Eye - Thresholding")
plt.axis("off")

# Display the left eye image after Normalization
plt.subplot(2, 2, 3)
plt.imshow(left_eye_normalized, cmap='gray')
plt.title("Left Eye - Normalization")
plt.axis("off")

# Display the left eye image after Thresholding
plt.subplot(2, 2, 4)
plt.imshow(left_eye_thresholded, cmap='gray')
plt.title("Left Eye - Thresholding")
plt.axis("off")

# Show the plots
plt.tight_layout()
plt.show()





#------------------------------------------------------------------------------------------------------------------------------------------
#Image Evaluations
#------------------------------------------------------------------------------------------------------------------------------------------
#PSNR Calculation

# Convert the filtered right eye image to grayscale
right_eye_grey = cv2.cvtColor(right_eye_8bit, cv2.COLOR_BGR2GRAY)

# Convert the filtered left eye image to grayscale
left_eye_grey = cv2.cvtColor(left_eye_8bit, cv2.COLOR_BGR2GRAY)


# Calculate PSNR for the right eye image
psnr_right_eye_norm = calculate_psnr(right_eye_grey, right_eye_normalized_8u, 255)  # 255 for 8-bit images

# Calculate PSNR for the left eye image
psnr_left_eye_norm = calculate_psnr(left_eye_grey, left_eye_normalized_8u, 255)  # 255 for 8-bit images

# Print or save the PSNR values
print(f"PSNR (Right Eye): {psnr_right_eye_norm:.2f} dB")
print(f"PSNR (Left Eye): {psnr_left_eye_norm:.2f} dB")

# Calculate PSNR for the right eye image
psnr_right_eye_thresh = calculate_psnr(right_eye_grey, right_eye_thresholded, 255)  # 255 for 8-bit images

# Calculate PSNR for the left eye image
psnr_left_eye_thresh = calculate_psnr(left_eye_grey, left_eye_thresholded, 255)

# Print or save the PSNR values
print(f"PSNR (Right Eye Thresh): {psnr_right_eye_thresh:.2f} dB")
print(f"PSNR (Left Eye Thresh): {psnr_left_eye_thresh:.2f} dB")

#------------------------------------------------------------------------------------------------------------------------------------------
#PSNR Visualization 

# PSNR values
psnr_right_sean = [31.12, 30.75, 31.79, 33.79, 30.73]
psnr_left_sean = [30.87, 31.13, 31.94, 33.39, 31.05]
eyes = ['Sean', 'Raffi', 'Hamid', 'Jobayer', 'Eye_13']

# Calculate the average PSNR values
average_psnr_right = sum(psnr_right_sean) / len(psnr_right_sean)
average_psnr_left = sum(psnr_left_sean) / len(psnr_left_sean)

# Bar width
bar_width = 0.35

# Set up positions for bars on X-axis
r1 = np.arange(len(eyes))
r2 = [x + bar_width for x in r1]

# Create the grouped bar chart
plt.bar(r1, psnr_left_sean, width=bar_width, label='Left Eye', color='blue')
plt.bar(r2, psnr_right_sean, width=bar_width, label='Right Eye', color='orange')

# Add labels, title, and legend
plt.xlabel('Eye Pairs')
plt.ylabel('PSNR (dB)')
plt.title('PSNR Comparison for Five Eye Pairs')
plt.xticks([r + bar_width/2 for r in range(len(eyes))], eyes)
plt.legend()

# Add a line for the average PSNR values
plt.axhline(average_psnr_left, color='blue', linestyle='--', label=f'Avg Left Eye: {average_psnr_left:.2f} dB')
plt.axhline(average_psnr_right, color='orange', linestyle='--', label=f'Avg Right Eye: {average_psnr_right:.2f} dB')

# Show the plot
plt.tight_layout()
plt.legend(loc='best')
plt.show()




#------------------------------------------------------------------------------------------------------------------------------------------
'''#SSIM Evaluation


# Calculate SSIM for the right eye image
ssim_right_eye = compare_ssim(right_eye_normalized_8u, right_eye_grey)

# Calculate SSIM for the left eye image
ssim_left_eye = compare_ssim(left_eye_normalized_8u, left_eye_grey)

print(f"SSIM (Right Eye): {ssim_right_eye:.4f}")
print(f"SSIM (Left Eye): {ssim_left_eye:.4f}")



# SSIM values
ssim_right_sean = [0.7155, 0.7111, 0.7797, 0.6335, 0.7082]
ssim_left_sean = [0.7183, 0.7073, 0.7779, 0.6282, 0.7060]
eyes = ['Sean', 'Raffi', 'Hamid', 'Jobayer', 'Eye_13']

# Calculate the average SSIM values
average_ssim_right = sum(ssim_right_sean) / len(ssim_right_sean)
average_ssim_left = sum(ssim_left_sean) / len(ssim_left_sean)

# Bar width
bar_width = 0.35

# Set up positions for bars on X-axis
r1 = np.arange(len(eyes))
r2 = [x + bar_width for x in r1]

# Create the grouped bar chart
plt.bar(r1, ssim_left_sean, width=bar_width, label='Left Eye', color='blue')
plt.bar(r2, ssim_right_sean, width=bar_width, label='Right Eye', color='orange')

# Add labels, title, and legend
plt.xlabel('Eye Pairs')
plt.ylabel('SSIM')
plt.title('SSIM Comparison for Five Eye Pairs')
plt.xticks([r + bar_width/2 for r in range(len(eyes))], eyes)
plt.legend()

# Add a line for the average SSIM values
plt.axhline(average_ssim_left, color='blue', linestyle='--', label=f'Avg Left Eye: {average_ssim_left:.4f}')
plt.axhline(average_ssim_right, color='orange', linestyle='--', label=f'Avg Right Eye: {average_ssim_right:.4f}')

# Show the plot
plt.tight_layout()
plt.legend(loc='best')
plt.show()
'''
#------------------------------------------------------------------------------------------------------------------------------------------
#MSE Evaluation

# Calculate the Mean Squared Error (MSE) for the right eye image
mse_right_eye = np.mean((right_eye_grey - right_eye_normalized_8u) ** 2)

# Calculate the Mean Squared Error (MSE) for the left eye image
mse_left_eye = np.mean((left_eye_grey - left_eye_normalized_8u) ** 2)

# Print the MSE values
print(f"MSE (Right Eye): {mse_right_eye:.2f}")
print(f"MSE (Left Eye): {mse_left_eye:.2f}")



# MSE values Visualization
mse_right_sean = [50.28, 54.69, 43.05, 27.14, 55.02]
mse_left_sean = [53.18, 50.13, 41.64, 29.77, 51.02]
eyes = ['Sean', 'Raffi', 'Hamid', 'Jobayer', 'Eye_13']

# Calculate the average MSE values
average_mse_right = sum(mse_right_sean) / len(mse_right_sean)
average_mse_left = sum(mse_left_sean) / len(mse_left_sean)

# Bar width
bar_width = 0.35

# Set up positions for bars on X-axis
r1 = np.arange(len(eyes))
r2 = [x + bar_width for x in r1]

# Create the grouped bar chart
plt.bar(r1, mse_left_sean, width=bar_width, label='Left Eye', color='blue')
plt.bar(r2, mse_right_sean, width=bar_width, label='Right Eye', color='orange')

# Add labels, title, and legend
plt.xlabel('Eye Pairs')
plt.ylabel('MSE')
plt.title('MSE Comparison for Five Eye Pairs')
plt.xticks([r + bar_width/2 for r in range(len(eyes))], eyes)
plt.legend()

# Add a line for the average MSE values
plt.axhline(average_mse_left, color='blue', linestyle='--', label=f'Avg Left Eye: {average_mse_left:.2f}')
plt.axhline(average_mse_right, color='orange', linestyle='--', label=f'Avg Right Eye: {average_mse_right:.2f}')

# Show the plot
plt.tight_layout()
plt.legend(loc='best')
plt.show()



#------------------------------------------------------------------------------------------------------------------------------------------
'''#VIF Evaluation

def calculate_vif(original_image, enhanced_image):
    # Convert images to float
    original_image = original_image.astype(np.float32)
    enhanced_image = enhanced_image.astype(np.float32)

    # Calculate image means
    mu1 = convolve2d(original_image, np.ones((8, 8)) / 64, mode='same')
    mu2 = convolve2d(enhanced_image, np.ones((8, 8)) / 64, mode='same')

    # Calculate image variances
    sigma1_sq = convolve2d(original_image**2, np.ones((8, 8)) / 64, mode='same') - mu1**2
    sigma2_sq = convolve2d(enhanced_image**2, np.ones((8, 8)) / 64, mode='same') - mu2**2
    sigma12 = convolve2d(original_image * enhanced_image, np.ones((8, 8)) / 64, mode='same') - mu1 * mu2

    # Constants for VIF calculation
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    # Calculate VIF
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    vif = num / den

    return np.mean(vif)

# Calculate VIF scores for the right and left eyes
vif_right_eye = calculate_vif(right_eye_grey, right_eye_normalized_8u)
vif_left_eye = calculate_vif(left_eye_grey, left_eye_normalized_8u)

print(f"VIF Score (Right Eye): {vif_right_eye:.4f}")
print(f"VIF Score (Left Eye): {vif_left_eye:.4f}")



#VIF Visualization

vif_right_sean = [0.7092, 0.7051, 0.7776, 0.6253, 0.7024]
vif_left_sean = [0.7119, 0.7012, 0.7755, 0.6308, 0.7003]
eyes = ['Sean', 'Raffi', 'Hamid', 'Jobayer', 'Eye_13']

# Calculate the average VIF values
average_vif_right = sum(vif_right_sean) / len(vif_right_sean)
average_vif_left = sum(vif_left_sean) / len(vif_left_sean)

# Bar width
bar_width = 0.35

# Set up positions for bars on X-axis
r1 = np.arange(len(eyes))
r2 = [x + bar_width for x in r1]

# Create the grouped bar chart
plt.bar(r1, vif_left_sean, width=bar_width, label='Left Eye', color='blue')
plt.bar(r2, vif_right_sean, width=bar_width, label='Right Eye', color= 'orange')

# Add labels, title, and legend
plt.xlabel('Eye Pairs')
plt.ylabel('VIF')
plt.title('VIF Comparison for Five Eye Pairs')
plt.xticks([r + bar_width/2 for r in range(len(eyes))], eyes)
plt.legend()

# Add a line for the average VIF values
plt.axhline(average_vif_left, color='blue', linestyle='--', label=f'Avg Left Eye: {average_vif_left:.4f}')
plt.axhline(average_vif_right, color='orange', linestyle='--', label=f'Avg Right Eye: {average_vif_right:.4f}')

# Show the plot
plt.tight_layout()
plt.legend(loc='best')
plt.show()



'''
#------------------------------------------------------------------------------------------------------------------------------------------
#Histogram Analysis

# Calculate histograms for the original and enhanced images
hist_original = cv2.calcHist([right_eye_grey], [0], None, [256], [0, 256])
hist_enhanced = cv2.calcHist([right_eye_normalized_8u], [0], None, [256], [0, 256])

# Plot histograms for visual comparison
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(hist_original, color='b')
plt.title('Histogram - Original')
plt.subplot(122)
plt.plot(hist_enhanced, color='r')
plt.title('Histogram - Enhanced')
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------