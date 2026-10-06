

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import cg

def downscale_2x2(X):
    
    return (
        X.reshape(-1, 28, 28)
         .reshape(-1, 14, 2, 14, 2)
         .mean(axis=(2, 4))
         .reshape(-1, 196)
    )

class kernelbasedMANDyClassifier:
    def __init__(self, alpha=0.59, reg=1e-4, dtype=np.float32):
        self.alpha = alpha
        self.reg = reg
        self.dtype = dtype
        self.X_train = None
        self.Z = None

    def _kernel(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        cos_vals = np.cos(self.alpha * diff, dtype=self.dtype)
        return np.prod(cos_vals, axis=2)
    
    def compute_gram_blockwise(self, X, block_size=1000):
        m, d = X.shape
        G = np.zeros((m, m), dtype=self.dtype)

        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)

            Xi = X[i:i_end]

            for j in range(0, m, block_size):
                j_end = min(j + block_size, m)

                Xj = X[j:j_end]

                diff = Xi[:, None, :] - Xj[None, :, :]
                K_block = np.prod(
                    np.cos(self.alpha * diff),
                    axis=2
                )

                G[i:i_end, j:j_end] = K_block

        return G

    def fit(self, X, y):
        X = X.astype(self.dtype)
        self.X_train = X

        m = X.shape[0]
        num_classes = np.max(y) + 1

        Y = np.zeros((m, num_classes), dtype=self.dtype)
        Y[np.arange(m), y] = 1

        print("Computing Gram matrix (blockwise)...")
        G = self.compute_gram_blockwise(X, block_size=1000)
        G += self.reg * np.eye(m, dtype=self.dtype)

        print("Solving system...")
        self.Z = np.zeros((m, num_classes), dtype=self.dtype)

        for c in range(num_classes):
            print(f"CG solve class {c}")
            z, info = cg(G, Y[:, c], maxiter=200)
            self.Z[:, c] = z

    def predict(self, X, block_size=500):
        X = X.astype(self.dtype)

        n_test = X.shape[0]
        m_train = self.X_train.shape[0]
        num_classes = self.Z.shape[1]

        scores = np.zeros((n_test, num_classes), dtype=self.dtype)

        for i in range(0, n_test, block_size):
            i_end = min(i + block_size, n_test)
            Xi = X[i:i_end]

            # compute kernel block
            diff = Xi[:, None, :] - self.X_train[None, :, :]
            K_block = np.prod(
                np.cos(self.alpha * diff),
                axis=2
            )

            scores[i:i_end] = K_block @ self.Z

        return np.argmax(scores, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)




class ARRClassifier:
    def __init__(self, d=196, n_basis=2, tt_rank=5, reg=1e-4, n_sweeps=5):
        self.d = d
        self.n_basis = n_basis
        self.r = tt_rank
        self.reg = reg
        self.n_sweeps = n_sweeps
        self.models = []  

    
    def feature_map(self, X, alpha=0.59):
        return np.stack(
            [np.cos(alpha * X), np.sin(alpha * X)],
            axis=2
        )  

    
    def init_tt(self):
        cores = []

        cores.append(np.random.randn(1, self.n_basis, self.r))

        for _ in range(self.d - 2):
            cores.append(np.random.randn(self.r, self.n_basis, self.r))

        cores.append(np.random.randn(self.r, self.n_basis, 1))

        return cores

    
    def right_orthogonalize(self, cores):
        for mu in reversed(range(1, self.d)):
            G = cores[mu]
            r1, n, r2 = G.shape

            Q, R = np.linalg.qr(G.reshape(r1, n * r2).T)
            Q = Q.T
            cores[mu] = Q.reshape(Q.shape[0], n, r2)

            cores[mu - 1] = np.einsum(
                'ijk,kl->ijl',
                cores[mu - 1],
                R.T
            )
        return cores


    def build_right_stack(self, Phi, cores):
        m = Phi.shape[0]
        Q_stack = [None] * self.d

        
        Q = np.ones((m, 1))

        for mu in reversed(range(self.d)):
            G = cores[mu]                  
            phi = Phi[:, mu, :]            

            
            tmp = np.einsum('mi,rij->mrj', phi, G)
            

            
            Q = np.einsum('mrj,mj->mr', tmp, Q)
            

            Q_stack[mu] = Q

        return Q_stack

    
    def compute_left_stack(self, Phi, cores):
        m = Phi.shape[0]
        P_stack = [None] * self.d

        P = np.ones((m, 1))
        for mu in range(self.d):
            G = cores[mu]
            tmp = np.einsum('mi,rij->mrj', Phi[:, mu, :], G)
            P = np.einsum('mr,mrj->mj', P, tmp)
            P_stack[mu] = P

        return P_stack

    
    def build_micromatrix(self, Phi, P_mu, Q_mu, mu):
        phi_mu = Phi[:, mu, :]  

        
        M = np.einsum('mi,mj,mk->mijk', P_mu, phi_mu, Q_mu)
        return M.reshape(P_mu.shape[0], -1)

    
    def solve_local(self, M, v, r_prev, r_next):
        A = M.T @ M + self.reg * np.eye(M.shape[1])
        b = M.T @ v

        w = np.linalg.solve(A, b)

        W = w.reshape(r_prev * self.n_basis, r_next)

        U, S, Vt = np.linalg.svd(W, full_matrices=False)

        rank = min(self.r, len(S))
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]

        return U, S, Vt

    
    def train_one_class(self, Phi, v):
        cores = self.init_tt()
        cores = self.right_orthogonalize(cores)

        for sweep in range(self.n_sweeps):

            
            Q_stack = self.build_right_stack(Phi, cores)

            
            P = np.ones((Phi.shape[0], 1))

            for mu in range(self.d - 1):
                M = self.build_micromatrix(
                    Phi, P, Q_stack[mu + 1], mu
                )

                r_prev = cores[mu].shape[0]
                r_next = cores[mu].shape[2]

                U, S, Vt = self.solve_local(
                    M, v, r_prev, r_next
                )

                cores[mu] = U.reshape(
                    r_prev, self.n_basis, -1
                )

                cores[mu + 1] = np.einsum(
                    'ij,jkl->ikl',
                    np.diag(S) @ Vt,
                    cores[mu + 1]
                )

                
                tmp = np.einsum('mi,rij->mrj', Phi[:, mu, :], cores[mu])
                P = np.einsum('mr,mrj->mj', P, tmp)

            
            for mu in reversed(range(1, self.d)):
               
                pass  

        return cores

    def fit(self, X, y):
        Phi = self.feature_map(X)

        n_classes = np.max(y) + 1

        for i in range(n_classes):
            print(f"Training class {i}")

            v = (y == i).astype(float)

            cores = self.train_one_class(Phi, v)

            self.models.append(cores)

    
    def predict(self, X):
        Phi = self.feature_map(X)
        scores = []

        for cores in self.models:
            result = np.ones((X.shape[0], 1))
            for mu in range(self.d):
                G = cores[mu]
                tmp = np.einsum('mi,rij->mrj', Phi[:, mu, :], G)
                result = np.einsum('mr,mrj->mj', result, tmp)
            scores.append(result.squeeze())

        scores = np.stack(scores, axis=1)
        return np.argmax(scores, axis=1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

import numpy as np
import matplotlib.pyplot as plt
def plot_misclassified(clf, X_test_down, X_test_full, y_test, max_images=20):
    y_pred = clf.predict(X_test_down)

    y_test = np.asarray(y_test)  

    mis_idx = np.where(y_pred != y_test)[0][:max_images]

    plt.figure(figsize=(10, 6))

    for i, idx in enumerate(mis_idx):
        plt.subplot(4, 5, i + 1)
        plt.imshow(X_test_full[idx].reshape(28, 28), cmap="gray_r")
        plt.title(f"{y_pred[idx]}", color="red")
        plt.xlabel(f"{y_test[idx]}", color="green")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

from sklearn.metrics import confusion_matrix
def plot_confusion(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_test = np.asarray(y_test)

    misc = y_pred != y_test
    y_true_mis = y_test[misc]
    y_pred_mis = y_pred[misc]
    cm = confusion_matrix(y_true_mis, y_pred_mis, labels=np.arange(10))

    plt.figure(2,figsize=(6, 5))
    plt.imshow(cm, cmap="viridis")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.show()


mnist = fetch_openml('mnist_784', version=1)  # original MNIST
#mnist = fetch_openml(data_id=40996)  # fashion MNIST 

X = mnist.data.to_numpy() / 255.0
#X = downscale_2x2(X)
y = mnist.target.astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=5000, random_state=42
)

# ARR se može pokrenut i za svih 60000 primjera iz trening seta, odnosno 10000
# iz test seta (točnost za fashion MNIST ispada 88.66%), ali je presporo pa uzimamo
# 10000 za trening i 5000 za testiranje, točnost za fashion MNIST je 85.96%
# za originalni MNIST je 96.56%
X_train = X_train_full[:10000]
y_train = y_train_full[:10000]
# za kernelbasedMANDyClassifier isto uzimamo 10000 primjera (točnost za originalni
# MNIST je 97.3%, a za fashion MNIST 86.68%), nisam isprobavo za više od 10000


X_train_downscaled = downscale_2x2(X_train)
X_test_downscaled = downscale_2x2(X_test)
clf = kernelbasedMANDyClassifier(alpha=0.59, reg=1e-3)
#clf = ARRClassifier(d=196, n_basis=2, tt_rank=5, reg=1e-2, n_sweeps=5)
clf.fit(X_train_downscaled, y_train)

print("Accuracy:", clf.score(X_test_downscaled, y_test))

plot_misclassified(clf, X_test_downscaled, X_test, y_test)

plot_confusion(clf, X_test_downscaled, y_test)