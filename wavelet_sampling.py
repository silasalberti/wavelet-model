import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import einops as eo


def constant(j: int):
    return torch.ones(2 ** j, 2 ** j)


def horizontal(j: int):
    return torch.cat(
        (torch.ones(2 ** j, 2 ** (j - 1)), -torch.ones(2 ** j, 2 ** (j - 1))), dim=1
    )


def vertical(j: int):
    return torch.cat(
        (torch.ones(2 ** (j - 1), 2 ** j), -torch.ones(2 ** (j - 1), 2 ** j)), dim=0
    )


def checker(j: int):
    return torch.cat(
        (
            torch.cat(
                (
                    torch.ones(2 ** (j - 1), 2 ** (j - 1)),
                    -torch.ones(2 ** (j - 1), 2 ** (j - 1)),
                ),
                dim=0,
            ),
            torch.cat(
                (
                    -torch.ones(2 ** (j - 1), 2 ** (j - 1)),
                    torch.ones(2 ** (j - 1), 2 ** (j - 1)),
                ),
                dim=0,
            ),
        ),
        dim=1,
    )


def haar2d_unnormalized(j: int, top_level: bool = True):
    if top_level:
        result = torch.stack(
            (constant(j), horizontal(j), vertical(j), checker(j)), dim=0
        )
        freq = torch.zeros(4)
    else:
        result = torch.stack((horizontal(j), vertical(j), checker(j)), dim=0)
        freq = torch.zeros(3)
    if j > 1:
        sub, subfreq = haar2d_unnormalized(j - 1, top_level=False)
        size = sub.shape[0]
        zeros = torch.zeros(size * 4, 2 ** j, 2 ** j)
        zeros[:size, : 2 ** (j - 1), : 2 ** (j - 1)] = sub
        zeros[size : 2 * size, 2 ** (j - 1) :, : 2 ** (j - 1)] = sub
        zeros[2 * size : 3 * size, : 2 ** (j - 1), 2 ** (j - 1) :] = sub
        zeros[3 * size :, 2 ** (j - 1) :, 2 ** (j - 1) :] = sub
        result = torch.cat((result, zeros), dim=0)
        freq = torch.cat(
            (freq, subfreq + 1, subfreq + 1, subfreq + 1, subfreq + 1), dim=0
        )
    return result, freq


def haar2d(j: int):
    H_unnormalized, freq = haar2d_unnormalized(j)
    M_unnormalized = H_unnormalized.reshape(2 ** j * 2 ** j, 2 ** j * 2 ** j)
    M = M_unnormalized / torch.unsqueeze(torch.norm(M_unnormalized, dim=1), dim=1)
    H = M.reshape(2 ** j * 2 ** j, 2 ** j, 2 ** j)
    return H, freq


def sample_image(j: int, decay: float):
    n = 2 ** j
    N = n * n
    z1 = torch.randn(N) / decay ** freq
    z2 = torch.randn(N) / decay ** freq
    z3 = torch.randn(N) / decay ** freq
    I1 = torch.einsum("abc,a->bc", H, z1)
    I2 = torch.einsum("abc,a->bc", H, z2)
    I3 = torch.einsum("abc,a->bc", H, z3)
    I = torch.stack((I1, I2, I3), dim=0)
    return I


def show_image(I):
    # Preprocess image
    I = (I - I.min()) / (I.max() - I.min())
    I = eo.rearrange(I, "c w h -> w h c")
    I = I.numpy()
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
