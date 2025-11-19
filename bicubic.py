import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim



def kernel(t, a=-0.5):
    t = np.abs(t).astype(np.float32)
    # Arreglo de salid en float32 con el mismo tamaño que `t`
    output = np.zeros_like(t, dtype=np.float32)
    # Caso 1: |t| <= 1
    mask1 = (t <= 1)
    output[mask1] = (a + 2) * (t[mask1] ** 3) - (a + 3) * (t[mask1] ** 2) + 1
    # Caso 2: 1 < |t| < 2
    mask2 = (t > 1) & (t < 2)
    # Fórmula: a|t|^3 - 5a|t|^2 + 8a|t| - 4a
    output[mask2] = a * (t[mask2] ** 3) - 5 * a * (t[mask2] ** 2) + 8 * a * t[mask2] - 4 * a
    # Caso 3: |t| >= 2 -> 0 (ya está inicializado)
    return output


def bicubic_interpolate(image, nw, nh):
    # imagen en escala de grises
    h, w = image.shape
    output = np.zeros((nh, nw), dtype=np.float32)
    # modificación para simular el comportamiento de OpenCV
    # Preparar imagen con padding tipo OpenCV reflect_101 para muestreo de bordes
    pad = 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    hp, wp = padded.shape

    for j in range(nh):
        # Mapeo centrado para Y similar a OpenCV
        y = (j + 0.5) * (h / nh) - 0.5
        l = int(np.floor(y))
        fy = y - l
        # Vecinos en Y: l-1, l, l+1, l+2 (en la imagen original)
        ls = np.array([l - 1, l, l + 1, l + 2])
        # Convertir a índices en la imagen padded
        ls_p = ls + pad
        ls_p = np.clip(ls_p, 0, hp - 1)
        # Pesos del kernel en Y 
        t_y = np.array([-1 - fy, -fy, 1 - fy, 2 - fy], dtype=np.float32)
        wy = kernel(t_y)

        for i in range(nw):
            # Mapeo centrado para X 
            x = (i + 0.5) * (w / nw) - 0.5
            k = int(np.floor(x))
            fx = x - k
            ks = np.array([k - 1, k, k + 1, k + 2])
            ks_p = ks + pad
            ks_p = np.clip(ks_p, 0, wp - 1)
            # Pesos del kernel en X 
            t_x = np.array([-1 - fx, -fx, 1 - fx, 2 - fx], dtype=np.float32)
            wx = kernel(t_x)

            # Extraer parche 4x4 desde la imagen padded y aplicar interpolación separable
            patch = padded[ls_p[:, None], ks_p[None, :]].astype(np.float32)
            inter = np.dot(patch, wx)
            value = np.dot(wy, inter)
            output[j, i] = value

    output = np.clip(output, 0, 255)
    return output.astype(image.dtype)



if __name__ == "__main__":
    # Ruta de la imagen
    img_path = r"C:\Users\yered\OneDrive\Escritorio\Bicubic\Imagenes\GT.jpg"
    reference_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {img_path}")

    # Redimensionar a la mitad para simular LR
    image_lr = cv2.resize(reference_image, (reference_image.shape[1] // 2, reference_image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

    h, w = image_lr.shape
    nh, nw = h * 2, w * 2


    bicubic_image = bicubic_interpolate(image_lr, nw, nh)
    # Usar OpenCV para comparar
    opencv_bicubic_image = cv2.resize(image_lr, (nw, nh), interpolation=cv2.INTER_CUBIC)

    # Métricas
    psnr_bicubic = cv2.PSNR(reference_image, bicubic_image)
    ssim_bicubic = ssim(reference_image, bicubic_image, data_range=255)
    mse_bicubic = np.mean((reference_image.astype(np.float64) - bicubic_image.astype(np.float64)) ** 2)

    psnr_opencv = cv2.PSNR(reference_image, opencv_bicubic_image)
    ssim_opencv = ssim(reference_image, opencv_bicubic_image, data_range=255)
    mse_opencv = np.mean((reference_image.astype(np.float64) - opencv_bicubic_image.astype(np.float64)) ** 2)

    print(f"Interpolación Bicúbica propia - PSNR: {psnr_bicubic:.4f}, SSIM: {ssim_bicubic:.4f}, MSE: {mse_bicubic:.4f}")
    print(f"Bicúbica OpenCV               - PSNR: {psnr_opencv:.4f}, SSIM: {ssim_opencv:.4f}, MSE: {mse_opencv:.4f}")

    # Mostrar imágenes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(reference_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Bicubic Interpolada")
    plt.imshow(bicubic_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Bicubic OpenCV")
    plt.imshow(opencv_bicubic_image, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.title("Diferencia Bicubic Interpolada")
    plt.imshow(np.abs(reference_image.astype(np.float64) - bicubic_image.astype(np.float64)), cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.title("Diferencia Bicubic OpenCV")
    plt.imshow(np.abs(reference_image.astype(np.float64) - opencv_bicubic_image.astype(np.float64)), cmap='gray')
    plt.axis('off')
    plt.show()
    