import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import einops as eo


def constant(j: int):
    return np.ones((2 ** j, 2 ** j))


def horizontal(j: int):
    return np.concatenate(
        (np.ones((2 ** j, 2 ** (j - 1))), -np.ones((2 ** j, 2 ** (j - 1)))), axis=1
    )


def vertical(j: int):
    return np.concatenate(
        (np.ones((2 ** (j - 1), 2 ** j)), -np.ones((2 ** (j - 1), 2 ** j))), axis=0
    )


def checker(j: int):
    return np.concatenate(
        (
            np.concatenate(
                (
                    np.ones((2 ** (j - 1), 2 ** (j - 1))),
                    -np.ones((2 ** (j - 1), 2 ** (j - 1))),
                ),
                axis=0,
            ),
            np.concatenate(
                (
                    -np.ones((2 ** (j - 1), 2 ** (j - 1))),
                    np.ones((2 ** (j - 1), 2 ** (j - 1))),
                ),
                axis=0,
            ),
        ),
        axis=1,
    )


def haar2d_unnormalized(j: int, top_level: bool = True):
    if top_level:
        result = np.stack((constant(j), horizontal(j), vertical(j), checker(j)), axis=0)
        freq = np.zeros(4)
    else:
        result = np.stack((horizontal(j), vertical(j), checker(j)), axis=0)
        freq = np.zeros(3)
    if j > 1:
        sub, subfreq = haar2d_unnormalized(j - 1, top_level=False)
        size = sub.shape[0]
        zeros = np.zeros((size * 4, 2 ** j, 2 ** j))
        zeros[:size, : 2 ** (j - 1), : 2 ** (j - 1)] = sub
        zeros[size : 2 * size, 2 ** (j - 1) :, : 2 ** (j - 1)] = sub
        zeros[2 * size : 3 * size, : 2 ** (j - 1), 2 ** (j - 1) :] = sub
        zeros[3 * size :, 2 ** (j - 1) :, 2 ** (j - 1) :] = sub
        result = np.concatenate((result, zeros), axis=0)
        freq = np.concatenate(
            (freq, subfreq + 1, subfreq + 1, subfreq + 1, subfreq + 1), axis=0
        )
    return result, freq


def haar2d(j: int):
    H_unnormalized, freq = haar2d_unnormalized(j)
    M_unnormalized = H_unnormalized.reshape(2 ** j * 2 ** j, 2 ** j * 2 ** j)
    M = M_unnormalized / np.expand_dims(np.linalg.norm(M_unnormalized, axis=1), axis=1)
    H = M.reshape(2 ** j * 2 ** j, 2 ** j, 2 ** j)
    return H, freq


def sample_image(j: int, decay: float):
    n = 2 ** j
    N = n * n
    z1 = np.random.randn(N) / decay ** freq
    z2 = np.random.randn(N) / decay ** freq
    z3 = np.random.randn(N) / decay ** freq
    I1 = np.einsum("abc,a->bc", H, z1)
    I2 = np.einsum("abc,a->bc", H, z2)
    I3 = np.einsum("abc,a->bc", H, z3)
    I = np.stack((I1, I2, I3), axis=0)
    return I


def show_image(I):
    # Preprocess image
    I = (I - I.min()) / (I.max() - I.min())
    I = eo.rearrange(I, "c w h -> w h c")
    # Visualize
    fig, ax = plt.subplots()
    ax.imshow(I)
    plt.axis("off")
    st.pyplot(fig)


st.write("# Sampling in Haar Basis")

j = st.slider("Image size", 2, 10, 7)

H, freq = haar2d(j)

decay = st.slider("Decay", 0.5, 5.0, 1.0)

col1, col2, col3 = st.columns(3)

with col1:
    show_image(sample_image(j, decay))

with col2:
    show_image(sample_image(j, decay))

with col3:
    show_image(sample_image(j, decay))
