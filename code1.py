import numpy as np  # numerical arrays and matrix operations
from PIL import Image  # image loading and saving

def load_grayscale_image(path):
    """
    Load an image from 'path' and convert it to a grayscale NumPy array.
    Pixel values in [0, 255], dtype=uint8.
    """
    img = Image.open(path).convert("L")   # "L" = 8-bit grayscale
    return np.array(img)


def save_grayscale_image(arr, path):
    """
    Save a 2D NumPy array as a grayscale image.
    """
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


def partition_host_image(host_img):
    """
    Partition a square N x N image into four N/2 x N/2 subimages:
    ftl: top-left
    ftr: top-right
    fbl: bottom-left
    fbr: bottom-right
    """
    N_rows, N_cols = host_img.shape
    assert N_rows == N_cols, "Host image must be square (N x N)."
    N = N_rows
    assert N % 2 == 0, "N must be even to split into four equal parts."

    half = N // 2

    ftl = host_img[0:half, 0:half]      # top-left
    ftr = host_img[0:half, half:N]      # top-right
    fbl = host_img[half:N, 0:half]      # bottom-left
    fbr = host_img[half:N, half:N]      # bottom-right

    return ftl, ftr, fbl, fbr


def combine_subimages(ftl_w, ftr, fbl, fbr_w):
    """
    Recombine four N/2 x N/2 subimages into one N x N image.
    """
    half = ftl_w.shape[0]
    N = half * 2
    result = np.zeros((N, N), dtype=np.uint8)

    result[0:half,     0:half]   = ftl_w
    result[0:half,     half:N]   = ftr
    result[half:N,     0:half]   = fbl
    result[half:N,     half:N]   = fbr_w

    return result


def watermark_to_bits(wm_img):
    """
    Convert grayscale watermark to binary (0/1) using simple threshold.
    """
    return (wm_img >= 128).astype(np.uint8)


#  Permutation of Watermark
def permute_watermark(wm_img, key):
    """
    Permute (scramble) the watermark image using a secret key.
    Returns:
        wm_perm : permuted watermark (same shape)
    """
    flat = wm_img.flatten()
    num_pixels = flat.size

    seed = abs(hash(key)) % (2**32)
    rng = np.random.default_rng(seed)

    perm_indices = np.arange(num_pixels)
    rng.shuffle(perm_indices)

    perm_flat = flat[perm_indices]
    return perm_flat.reshape(wm_img.shape)


def depermute_watermark(wm_perm, key):
    """
    Reverse the permutation applied with permute_watermark(..., key).
    wm_perm: permuted watermark image (0/1 or 0/255)
    """
    flat = wm_perm.flatten()
    num_pixels = flat.size

    seed = abs(hash(key)) % (2**32)
    rng = np.random.default_rng(seed)

    perm_indices = np.arange(num_pixels)
    rng.shuffle(perm_indices)

    # Inverse permutation
    inv_perm = np.zeros_like(perm_indices)
    inv_perm[perm_indices] = np.arange(num_pixels)

    original_flat = flat[inv_perm]
    return original_flat.reshape(wm_perm.shape)


# =========================
#  Embedding in D (S matrix of ftl)
# =========================

def svd_blocks_and_collect_Dlarge(ftl, block_size, wm_shape):
    """
    - Split ftl into non-overlapping blocks of size (block_size x block_size)
    - For each block, compute SVD and store U, S, Vt
    - Collect the largest singular value (S[0]) from each block
      into a matrix Dlarge with shape = wm_shape
    Returns:
      U_list, S_list, Vt_list, Dlarge
    """
    H, W = ftl.shape
    assert H == W, "ftl must be square."
    assert H % block_size == 0, "ftl size must be divisible by block_size."

    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim
    assert num_blocks == (wm_shape[0] * wm_shape[1]), \
        "Number of blocks must match number of watermark pixels."

    U_list = []
    S_list = []
    Vt_list = []
    Dlarge_flat = np.zeros(num_blocks, dtype=float)

    idx = 0    # flatten index for Dlarge
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size
            block = ftl[y0:y0+block_size, x0:x0+block_size].astype(float)

            # SVD
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            U_list.append(U)
            S_list.append(S)
            Vt_list.append(Vt)

            # Largest singular value
            Dlarge_flat[idx] = S[0]
            idx += 1

    Dlarge = Dlarge_flat.reshape(wm_shape)
    return U_list, S_list, Vt_list, Dlarge


def modify_Dlarge_with_watermark(Dlarge, wm_bits, T):
    """
    Modify Dlarge according to watermark bits using bin-based embedding.
    Dlarge and wm_bits have the same shape.

    T = step size (bin width)
    """
    assert Dlarge.shape == wm_bits.shape, "Dlarge and watermark must match in shape."

    dmin = Dlarge.min()
    dmax = Dlarge.max()

    Dlarge_flat = Dlarge.flatten()
    wm_flat = wm_bits.flatten()
    modified_flat = np.zeros_like(Dlarge_flat, dtype=float)

    # Number of bins (ceil)
    num_bins = int(np.ceil((dmax - dmin) / T))

    for i, d in enumerate(Dlarge_flat):
        bit = wm_flat[i]

        # Bin index:
        bin_idx = int((d - dmin) // T)
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1

        d_low = dmin + bin_idx * T
        d_high = d_low + T
        mid = 0.5 * (d_low + d_high)

        if bit == 1:
            # Put in lower half center  (Range1)
            new_d = 0.5 * (d_low + mid)
        else:
            # Put in upper half center (Range2)
            new_d = 0.5 * (mid + d_high)

        modified_flat[i] = new_d

    return modified_flat.reshape(Dlarge.shape)


def reconstruct_ftl_from_svd(U_list, S_list, Vt_list, Dlarge_modified, block_size, ftl_shape):
    """
    Reconstruct ftl_w (watermarked ftl) from SVD components and modified D(1,1) values.
    """
    H, W = ftl_shape
    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim
    assert num_blocks == Dlarge_modified.size, "Number of blocks and Dlarge size must match."

    ftl_w = np.zeros((H, W), dtype=float)

    D_flat = Dlarge_modified.flatten()
    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            U = U_list[idx]
            S = S_list[idx].copy()
            Vt = Vt_list[idx]

            # Replace largest singular value
            S[0] = D_flat[idx]

            # Reconstruct block
            block_rec = U @ np.diag(S) @ Vt

            y0 = by * block_size
            x0 = bx * block_size
            ftl_w[y0:y0+block_size, x0:x0+block_size] = block_rec

            idx += 1

    # Clip to [0, 255] and convert to uint8
    ftl_w = np.clip(ftl_w, 0, 255).astype(np.uint8)
    return ftl_w


def embed_watermark_in_ftl(ftl, wm, block_size=8, T=5.0):
    """
    Embed a grayscale watermark wm into the top-left host subimage ftl
    using block-based SVD and bin quantization on the largest singular values.

    Returns:
      ftl_w : watermarked ftl subimage
    """
    wm_bits = watermark_to_bits(wm)

    U_list, S_list, Vt_list, Dlarge = svd_blocks_and_collect_Dlarge(
        ftl, block_size, wm_bits.shape
    )

    Dlarge_modified = modify_Dlarge_with_watermark(Dlarge, wm_bits, T)

    ftl_w = reconstruct_ftl_from_svd(
        U_list, S_list, Vt_list, Dlarge_modified, block_size, ftl.shape
    )

    return ftl_w


# =========================
#  Embedding in U (U matrix of fbr)
# =========================

def embed_watermark_in_fbr_U(fbr, wm, block_size=8, alpha=0.1):
    """
    Embed a watermark image into the U matrices of SVD blocks of the fbr subimage.

    fbr       : bottom-right host subimage (N/2 x N/2), uint8
    wm        : watermark image (grayscale)
    block_size: M (block is M x M)
    alpha     : embedding strength

    Returns:
        fbr_w : watermarked fbr subimage (uint8)
    """
    H, W = fbr.shape
    assert H == W, "fbr must be square."
    assert H % block_size == 0, "fbr size must be divisible by block_size."

    wm_bits = watermark_to_bits(wm)
    wm_flat = wm_bits.flatten()
    num_bits = wm_flat.size

    blocks_per_dim = H // block_size
    total_blocks = blocks_per_dim * blocks_per_dim
    assert total_blocks >= num_bits, (
        f"Not enough blocks ({total_blocks}) to embed {num_bits} watermark bits."
    )

    fbr_w = np.zeros_like(fbr, dtype=float)

    bit_idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size

            block = fbr[y0:y0+block_size, x0:x0+block_size].astype(float)

            # SVD
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            if bit_idx < num_bits:
                bit = wm_flat[bit_idx]
                bit_idx += 1

                u11 = U[0, 0]
                u21 = U[1, 0]
                diff = u11 - u21
                mean = 0.5 * (u11 + u21)

                # bit 1 => enforce u11 > u21 with |diff| >= alpha
                # bit 0 => enforce u11 < u21 with |diff| >= alpha
                if bit == 1:
                    if diff <= 0 or abs(diff) < alpha:
                        U[0, 0] = mean + alpha / 2.0
                        U[1, 0] = mean - alpha / 2.0
                else:  # bit == 0
                    if diff >= 0 or abs(diff) < alpha:
                        U[0, 0] = mean - alpha / 2.0
                        U[1, 0] = mean + alpha / 2.0

            block_w = U @ np.diag(S) @ Vt
            fbr_w[y0:y0+block_size, x0:x0+block_size] = block_w

    fbr_w = np.clip(fbr_w, 0, 255).astype(np.uint8)
    return fbr_w


# =========================
#  Extraction from D (ftlw)
# =========================

def svd_blocks_and_collect_D_from_ftlw(ftlw, block_size):
    """
    Apply block-based SVD on ftlw and collect D(1,1) from each block.

    Returns:
        Dlarge : 2D array of largest singular values,
                 shape = (blocks_per_dim, blocks_per_dim)
    """
    H, W = ftlw.shape
    assert H == W, "ftlw must be square."
    assert H % block_size == 0, "ftlw size must be divisible by block_size."

    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim
    Dlarge_flat = np.zeros(num_blocks, dtype=float)

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size
            block = ftlw[y0:y0+block_size, x0:x0+block_size].astype(float)

            # SVD
            _, S, _ = np.linalg.svd(block, full_matrices=False)

            # Largest singular value
            Dlarge_flat[idx] = S[0]
            idx += 1

    Dlarge = Dlarge_flat.reshape((blocks_per_dim, blocks_per_dim))
    return Dlarge


def extract_bits_from_Dlarge(Dlarge, T):
    """
    Given Dlarge (largest singular values) and step size T,
    extract watermark bits using bin partition and Range1/Range2 rule.

    Returns:
        wm_bits : 2D uint8 array with values 0 or 1
    """
    dmin = Dlarge.min()
    dmax = Dlarge.max()

    D_flat = Dlarge.flatten()
    bits_flat = np.zeros_like(D_flat, dtype=np.uint8)

    num_bins = int(np.ceil((dmax - dmin) / T))

    for i, d in enumerate(D_flat):
        # Bin index
        bin_idx = int((d - dmin) // T)
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1

        d_low = dmin + bin_idx * T
        d_high = d_low + T
        mid = 0.5 * (d_low + d_high)

        # Range1 = lower half  → bit = 1
        # Range2 = upper half  → bit = 0
        if d < mid:
            bits_flat[i] = 1
        else:
            bits_flat[i] = 0

    wm_bits = bits_flat.reshape(Dlarge.shape)
    return wm_bits


def extract_watermark_from_ftlw(ftlw, block_size=8, T=5.0):
    """
    Full extraction from D matrix of ftlw:
      - block SVD
      - collect D(1,1)
      - decide bits based on bins

    Returns:
        wm_bits : 0/1 array
        wm_img  : 0 or 255 grayscale image (uint8)
    """
    Dlarge = svd_blocks_and_collect_D_from_ftlw(ftlw, block_size)
    wm_bits = extract_bits_from_Dlarge(Dlarge, T)

    wm_img = (wm_bits * 255).astype(np.uint8)
    return wm_bits, wm_img


# =========================
#  Extraction from U (fbrw)
# =========================

def extract_bits_from_fbrw_U(fbrw, block_size=8):
    """
    Watermark extraction from U matrix of bottom-right watermarked subimage fbrw.

    For each MxM block of fbrw:
      - compute SVD: block = U @ S @ Vt
      - compare U[1,0] and U[0,0]
      - if U[1,0] > U[0,0] -> bit = 1
        else                 -> bit = 0

    Returns:
        wm_bits : 2D array of 0/1 bits, shape = (blocks_per_dim, blocks_per_dim)
        wm_img  : 0/255 grayscale uint8 image (same shape)
    """
    H, W = fbrw.shape
    assert H == W, "fbrw must be square."
    assert H % block_size == 0, "fbrw size must be divisible by block_size."

    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    bits_flat = np.zeros(num_blocks, dtype=np.uint8)

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size

            block = fbrw[y0:y0+block_size, x0:x0+block_size].astype(float)

            # SVD
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            u11 = U[0, 0]
            u21 = U[1, 0]

            # Paper rule:
            # w(i,j) = 1 if u21 > u11
            # w(i,j) = 0 otherwise
            if u21 > u11:
                bits_flat[idx] = 1
            else:
                bits_flat[idx] = 0

            idx += 1

    wm_bits = bits_flat.reshape((blocks_per_dim, blocks_per_dim))
    wm_img = (wm_bits * 255).astype(np.uint8)
    return wm_bits, wm_img


# =========================
#  MAIN: embedding + extraction demo
# =========================

if __name__ == "__main__":
    # ---- INPUT FILES ----
    host_path = "picture.png"
    wm_path   = "ivan.png"
    secret_key = "my_secret_key_123"  # used if you want permutation

    # ---- PARAMETERS ----
    block_size = 8
    T = 5.0
    alpha = 0.1

    # ---- LOAD IMAGES ----
    host = load_grayscale_image(host_path)
    wm_original = load_grayscale_image(wm_path)

    # ---- PARTITION HOST ----
    ftl, ftr, fbl, fbr = partition_host_image(host)
    H_ftl = ftl.shape[0]
    blocks_per_dim = H_ftl // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    # ---- RESIZE WATERMARK TO MATCH BLOCK CAPACITY ----
    # Watermark will be blocks_per_dim x blocks_per_dim
    wm_resized_img = Image.fromarray(wm_original.astype(np.uint8)).resize(
        (blocks_per_dim, blocks_per_dim),
        Image.BILINEAR
    )
    wm_resized = np.array(wm_resized_img)
    wm_pixels = wm_resized.size

    assert num_blocks >= wm_pixels, (
        f"ERROR: Watermark has {wm_pixels} pixels but only "
        f"{num_blocks} SVD blocks are available in ftl."
    )

    # ---- OPTIONAL: PERMUTE WATERMARK ----
    wm_perm = permute_watermark(wm_resized, secret_key)

    # ---- EMBEDDING ----
    ftl_w = embed_watermark_in_ftl(ftl, wm_perm, block_size, T=T)
    fbr_w = embed_watermark_in_fbr_U(fbr, wm_perm, block_size, alpha=alpha)

    watermarked_host = combine_subimages(ftl_w, ftr, fbl, fbr_w)
    save_grayscale_image(watermarked_host, "host_watermarked.png")

    # ---- EXTRACTION FROM WATERMARKED IMAGE ----
    F = load_grayscale_image("host_watermarked.png")
    ftlw_ex, ftr_w_ex, fbl_w_ex, fbrw_ex = partition_host_image(F)

    # From D matrix (top-left)
    wm_bits_D, wm_img_D_perm = extract_watermark_from_ftlw(ftlw_ex, block_size, T=T)
    # depermute
    wm_img_D_deperm = depermute_watermark(wm_bits_D, secret_key)
    wm_img_D_deperm = (wm_img_D_deperm * 255).astype(np.uint8)

    save_grayscale_image(wm_img_D_perm,   "extracted_from_D_perm.png")
    save_grayscale_image(wm_img_D_deperm, "extracted_from_D.png")

    # From U matrix (bottom-right)
    wm_bits_U, wm_img_U_perm = extract_bits_from_fbrw_U(fbrw_ex, block_size)
    wm_img_U_deperm = depermute_watermark(wm_bits_U, secret_key)
    wm_img_U_deperm = (wm_img_U_deperm * 255).astype(np.uint8)

    save_grayscale_image(wm_img_U_perm,   "extracted_from_U_perm.png")
    save_grayscale_image(wm_img_U_deperm, "extracted_from_U.png")
