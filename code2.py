from matplotlib.cbook import flatten
import numpy as np  # numerical arrays and matrix operations
from PIL import Image  # image loading and saving


#for preping images
def load_grayscale_image(path):
    img = Image.open(path).convert("L")
    return np.array(img)

def save_grayscale_image(arr, path):
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


#  A. Host Image Partition

# (1) and (2)
def partition_host_image(host_img):
    N = host_img.shape[0]
    half = N // 2

    ftl = host_img[0:half,     0:half]   # top-left
    ftr = host_img[0:half,     half:N]   # top-right
    fbl = host_img[half:N,     0:half]   # bottom-left
    fbr = host_img[half:N,     half:N]   # bottom-right

    return ftl, ftr, fbl, fbr

#=========================
#B. Watermark Embedding in D Matrix
#=========================

# (3) + (4)
# we are doing SVD on blocks of ftl and collecting Dlarge


def svd_blocks_and_collect_Dlarge(ftl, block_size, wm_shape):
    H = ftl.shape[0] #number of rows
    blocks_per_dim = H // block_size 
    num_blocks = blocks_per_dim * blocks_per_dim #total number of blocks

    # Lists to store U, S, Vt from SVD for each block
    U_list = [] 
    S_list = []
    Vt_list = []
    Dlarge_flat = np.zeros(num_blocks, dtype=float) #array to store largest singular values

    idx = 0
    for by in range(blocks_per_dim): #loop over block rows
        for bx in range(blocks_per_dim): #loop over block columns
            y0 = by * block_size
            x0 = bx * block_size
            block = ftl[y0:y0+block_size, x0:x0+block_size].astype(float)

            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            U_list.append(U)
            S_list.append(S)
            Vt_list.append(Vt)

            Dlarge_flat[idx] = S[0] #S is sorted in descending order, so it's the first element  
            idx += 1

    Dlarge = Dlarge_flat.reshape(wm_shape)  #reshape to match watermark shape
    return U_list, S_list, Vt_list, Dlarge

# (5) + (6)

def modify_Dlarge_with_watermark(Dlarge, wm_bits, T):
    dmin = Dlarge.min()
    dmax = Dlarge.max()

    Dlarge_flat = Dlarge.flatten() #convert to 1D for easier processing
    wm_flat = wm_bits.flatten()
    modified_flat = np.zeros_like(Dlarge_flat, dtype=float) 

    num_bins = int(np.ceil((dmax - dmin) / T))


# flatten + single loop --> faster
    for i, d in enumerate(Dlarge_flat):
        bit = wm_flat[i]

        # Determine bin index
        bin_idx = int((d - dmin) // T)

        # check bounds
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1

        # Calculate bin boundaries
        d_low = dmin + bin_idx * T
        d_high = d_low + T
        mid = 0.5 * (d_low + d_high)

        if bit == 1:
            # Put in lower half center (Range1)
            new_d = 0.5 * (d_low + mid)
        else:
            # Put in upper half center (Range2)
            new_d = 0.5 * (mid + d_high)

        modified_flat[i] = new_d

    return modified_flat.reshape(Dlarge.shape)


def embed_watermark_in_ftl(ftl, wm, block_size=8, T=5.0):
    wm_bits = (wm // 255).astype(np.uint8)

    U_list, S_list, Vt_list, Dlarge = svd_blocks_and_collect_Dlarge(
        ftl, block_size, wm_bits.shape
    )
    Dlarge_modified = modify_Dlarge_with_watermark(Dlarge, wm_bits, T)

    ftl_w = reconstruct_ftl_from_svd(
        U_list, S_list, Vt_list, Dlarge_modified, block_size, ftl.shape
    )
    return ftl_w

def combine_subimages(ftl_w, ftr, fbl, fbr_w):
    half = ftl_w.shape[0]
    N = half * 2
    result = np.zeros((N, N), dtype=np.uint8)

    result[0:half,     0:half]   = ftl_w
    result[0:half,     half:N]   = ftr
    result[half:N,     0:half]   = fbl
    result[half:N,     half:N]   = fbr_w

    return result

# (7)

def reconstruct_ftl_from_svd(U_list, S_list, Vt_list, Dlarge_modified, block_size, ftl_shape):

    H = ftl_shape[0]
    blocks_per_dim = H // block_size

    ftl_w = np.zeros((H, H), dtype=float)
    D_flat = Dlarge_modified.flatten()

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            U = U_list[idx]
            S = S_list[idx].copy()
            Vt = Vt_list[idx]

            S[0] = D_flat[idx]

            block_rec = U @ np.diag(S) @ Vt

            y0 = by * block_size
            x0 = bx * block_size
            ftl_w[y0:y0+block_size, x0:x0+block_size] = block_rec

            idx += 1

    ftl_w = np.clip(ftl_w, 0, 255).astype(np.uint8)
    return ftl_w

#permutation functions

def _rng_from_key(key):
    seed = abs(hash(key)) % (2**32)
    return np.random.default_rng(seed)

def permute_watermark(wm_img, key):
    flat = wm_img.flatten()
    num_pixels = flat.size

    rng = _rng_from_key(key)
    perm_indices = np.arange(num_pixels)
    rng.shuffle(perm_indices)

    perm_flat = flat[perm_indices]
    return perm_flat.reshape(wm_img.shape)


def depermute_watermark(wm_perm, key):
    flat = wm_perm.flatten()
    num_pixels = flat.size

    rng = _rng_from_key(key)
    perm_indices = np.arange(num_pixels)
    rng.shuffle(perm_indices)

    inv_perm = np.zeros_like(perm_indices)
    inv_perm[perm_indices] = np.arange(num_pixels)

    original_flat = flat[inv_perm]
    return original_flat.reshape(wm_perm.shape)


# (8) + (9)
def embed_watermark_in_fbr_U(fbr, wm, block_size=8, alpha=0.1):
    H = fbr.shape[0]

    wm_bits = (wm // 255).astype(np.uint8)
    wm_flat = wm_bits.flatten()
    num_bits = wm_flat.size

    blocks_per_dim = H // block_size

    fbr_w = np.zeros_like(fbr, dtype=float)

    bit_idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size

            block = fbr[y0:y0+block_size, x0:x0+block_size].astype(float)

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
                else:
                    if diff >= 0 or abs(diff) < alpha:
                        U[0, 0] = mean - alpha / 2.0
                        U[1, 0] = mean + alpha / 2.0

            block_w = U @ np.diag(S) @ Vt
            fbr_w[y0:y0+block_size, x0:x0+block_size] = block_w

    fbr_w = np.clip(fbr_w, 0, 255).astype(np.uint8)
    return fbr_w

#  Extraction from D (ftlw)

def svd_blocks_and_collect_D_from_ftlw(ftlw, block_size):
    H = ftlw.shape[0]
    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    Dlarge_flat = np.zeros(num_blocks, dtype=float)

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size
            block = ftlw[y0:y0+block_size, x0:x0+block_size].astype(float)

            _, S, _ = np.linalg.svd(block, full_matrices=False)
            Dlarge_flat[idx] = S[0]
            idx += 1

    Dlarge = Dlarge_flat.reshape((blocks_per_dim, blocks_per_dim))
    return Dlarge


def extract_bits_from_Dlarge(Dlarge, T):
    
    dmin = Dlarge.min()
    dmax = Dlarge.max()

    D_flat = Dlarge.flatten()
    bits_flat = np.zeros_like(D_flat, dtype=np.uint8)

    num_bins = int(np.ceil((dmax - dmin) / T))

    for i, d in enumerate(D_flat):
        bin_idx = int((d - dmin) // T)
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1

        d_low = dmin + bin_idx * T
        d_high = d_low + T
        mid = 0.5 * (d_low + d_high)

        bits_flat[i] = 1 if d < mid else 0

    wm_bits = bits_flat.reshape(Dlarge.shape)
    return wm_bits


def extract_watermark_from_ftlw(ftlw, block_size=8, T=5.0):
    Dlarge = svd_blocks_and_collect_D_from_ftlw(ftlw, block_size)
    wm_bits = extract_bits_from_Dlarge(Dlarge, T)
    wm_img = (wm_bits * 255).astype(np.uint8)
    return wm_bits, wm_img


# =========================
#  Extraction from U (fbrw)
# =========================

def extract_bits_from_fbrw_U(fbrw, block_size=8):
    H = fbrw.shape[0]
    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    bits_flat = np.zeros(num_blocks, dtype=np.uint8)

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size

            block = fbrw[y0:y0+block_size, x0:x0+block_size].astype(float)

            U, _, _ = np.linalg.svd(block, full_matrices=False)

            u11 = U[0, 0]
            u21 = U[1, 0]

            bits_flat[idx] = 1 if u21 > u11 else 0
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
    secret_key = "my_secret_key_123"  # used for permutation

    # ---- PARAMETERS ----
    block_size = 8
    T = 5.0
    alpha = 0.1

    # ---- LOAD IMAGES ----
    host = load_grayscale_image(host_path)
    wm_original = load_grayscale_image(wm_path)

    # ---- PARTITION HOST ----
    ftl, ftr, fbl, fbr = partition_host_image(host)

    # ---- RESIZE WATERMARK TO MATCH BLOCK CAPACITY ----
    H_ftl = ftl.shape[0]
    blocks_per_dim = H_ftl // block_size

    # watermark size = blocks_per_dim x blocks_per_dim
    # Use NEAREST so black/white stays 0/255 (no gray levels introduced)
    wm_resized_img = Image.fromarray(wm_original.astype(np.uint8)).resize(
        (blocks_per_dim, blocks_per_dim),
        Image.NEAREST
    )
    wm_resized = np.array(wm_resized_img)

    # ---- PERMUTE WATERMARK (scrambling with secret key) ----
    wm_perm = permute_watermark(wm_resized, secret_key)

    # ---- EMBEDDING ----
    ftl_w = embed_watermark_in_ftl(ftl, wm_perm, block_size, T=T)
    fbr_w = embed_watermark_in_fbr_U(fbr, wm_perm, block_size, alpha=alpha)

    watermarked_host = combine_subimages(ftl_w, ftr, fbl, fbr_w)
    save_grayscale_image(watermarked_host, "host_watermarked.png")

    # ---- EXTRACTION FROM WATERMARKED IMAGE ----
    F = load_grayscale_image("host_watermarked.png")
    ftlw_ex, ftr_w_ex, fbl_w_ex, fbrw_ex = partition_host_image(F)

    # ====== EXTRACTION FROM D (top-left) ======
    wm_bits_D, wm_img_D_perm = extract_watermark_from_ftlw(ftlw_ex, block_size, T=T)

    # de-permute to restore original watermark order
    wm_img_D_deperm = depermute_watermark(wm_bits_D, secret_key)
    wm_img_D_deperm = (wm_img_D_deperm * 255).astype(np.uint8)

    save_grayscale_image(wm_img_D_perm,   "extracted_from_D_perm.png")  # scrambled
    save_grayscale_image(wm_img_D_deperm, "extracted_from_D.png")       # descrambled

    # ====== EXTRACTION FROM U (bottom-right) ======
    wm_bits_U, wm_img_U_perm = extract_bits_from_fbrw_U(fbrw_ex, block_size)

    wm_img_U_deperm = depermute_watermark(wm_bits_U, secret_key)
    wm_img_U_deperm = (wm_img_U_deperm * 255).astype(np.uint8)

    save_grayscale_image(wm_img_U_perm,   "extracted_from_U_perm.png")  # scrambled
    save_grayscale_image(wm_img_U_deperm, "extracted_from_U.png")       # descrambled
