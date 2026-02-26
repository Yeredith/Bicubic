def epi2d(restored: np.ndarray, reference: np.ndarray, alpha: float = 0.5, eps: float = 1e-12) -> float:
    """
    Edge Preservation Index for 2D images
    
    Args:
        restored: Restored/predicted image
        reference: Reference/original image
        alpha: Parameter for Laplacian kernel (default 0.5)
        eps: Small value to avoid division by zero
    
    Returns:
        EPI value

    original code: https://www.mathworks.com/matlabcentral/fileexchange/65261-edge-preservation-index-new-file/files/EPI_new.m
    """
    k = _laplacian_kernel(alpha)
    d_ref = convolve2d(reference, k, mode='same', boundary='symm')
    d_res = convolve2d(restored, k, mode='same', boundary='symm')
    p1 = d_ref - d_ref.mean()
    p2 = d_res - d_res.mean()
    num = np.sum(p1 * p2)
    den = np.sum(p1 ** 2) * np.sum(p2 ** 2)
    return float(num / np.sqrt(den + eps))
