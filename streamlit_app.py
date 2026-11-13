import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Image Segmentation App", layout="wide")
st.title("Homework6 박현우 : Object Counting")

# 최상단에 객체 수를 표시할 자리 (결과는 아래에서 계산 후 채움)
count_placeholder = st.empty()

# ---------------------------------------------------------------
# Sidebar: 입력 및 파라미터
# ---------------------------------------------------------------
st.sidebar.header("설정")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
kernel_size = st.sidebar.slider("Median filter 커널 크기 (홀수)", 1, 31, 7, step=2)
th_val = st.sidebar.slider("Threshold 값", 0, 255, 128)
dark_obj = st.sidebar.checkbox("어두운 객체 (밝은 배경)", value=True)
fill_holes = st.sidebar.checkbox("객체 내부 구멍 채우기", value=True)
min_area = st.sidebar.slider("최소 면적 (pixel) 미만 제거", 0, 5000, 2000, step=50)
cmap_label = st.sidebar.selectbox("Label colormap", ["Spectral_r", "nipy_spectral", "tab20", "viridis"], index=0)

if uploaded is None:
    st.info("왼쪽 사이드바에서 이미지를 업로드하세요.")
    st.stop()


# ---------------------------------------------------------------
# 처리 함수 (캐싱)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def denoise(f: np.ndarray, k: int) -> np.ndarray:
    return signal.medfilt2d(f, k)


def show_image(img, title, cmap="gray", colorbar=False, figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, cmap=cmap)
    ax.axis("off")
    ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------
# 1. 이미지 로드
# ---------------------------------------------------------------
I = Image.open(uploaded).convert("L")
f = np.array(I)

# ---------------------------------------------------------------
# 2. Denoising
# ---------------------------------------------------------------
f_denoised = denoise(f, kernel_size)

# ---------------------------------------------------------------
# 3. Thresholding + 후처리 + Labeling
# ---------------------------------------------------------------
# 객체가 어두우면 th 이하를 객체로, 밝으면 th 초과를 객체로
f_mask_raw = (f_denoised <= th_val) if dark_obj else (f_denoised > th_val)

# 객체 안쪽의 밝은 반점(구멍)을 채워서 하나의 객체로 만들기
f_mask = nd.binary_fill_holes(f_mask_raw) if fill_holes else f_mask_raw

labels0, nums0 = nd.label(f_mask)

# 최소 면적 미만(잡음, 모서리에 잘린 조각 등) 제거 후 label을 1..N으로 재정렬
labels = np.zeros_like(labels0)
nums = 0
for i in range(1, nums0 + 1):
    region = labels0 == i
    if region.sum() >= min_area:
        nums += 1
        labels[region] = nums

# 최상단 객체 수 표시
count_placeholder.metric("검출된 객체 수 (object count)", int(nums))

st.subheader("1. 원본 / 노이즈 제거")
c1, c2 = st.columns(2)
with c1:
    show_image(f, "Input image (grayscale)", colorbar=True)
with c2:
    show_image(f_denoised, f"Median filtered (k={kernel_size})", colorbar=True)

# ---------------------------------------------------------------
# 4. 히스토그램
# ---------------------------------------------------------------
st.subheader("2. 히스토그램")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(f_denoised.ravel(), bins=256, range=(0, 256), color="gray")
ax.axvline(th_val, color="red", linestyle="--", label=f"threshold = {th_val}")
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Frequency")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("3. 이진화 / 레이블링")
c1, c2, c3 = st.columns(3)
with c1:
    show_image(f_mask_raw, f"Binary mask (th={th_val})")
with c2:
    show_image(f_mask, "After hole filling")
with c3:
    show_image(labels, f"Labeled components ({nums}개)", cmap=cmap_label)

if nums == 0:
    st.warning("검출된 component가 없습니다. Threshold 값이나 최소 면적을 조절하세요.")
    st.stop()

# ---------------------------------------------------------------
# 5. Component 크기 계산
# ---------------------------------------------------------------
cc = np.zeros((nums, 2), dtype=int)
for i in range(1, nums + 1):
    cc[i - 1, 0] = i
    cc[i - 1, 1] = np.sum(labels == i)

st.subheader("4. Component별 픽셀 수")
st.dataframe(
    {"index": cc[:, 0], "number of pixels": cc[:, 1]},
    use_container_width=True,
    height=min(35 * (nums + 1), 400),
)

ind_max = int(np.argmax(cc[:, 1]))
ind_min = int(np.argmin(cc[:, 1]))
label_max = int(cc[ind_max, 0])
label_min = int(cc[ind_min, 0])

m1, m2 = st.columns(2)
m1.metric("가장 큰 component", f"index {label_max}", f"{cc[ind_max, 1]} pixels")
m2.metric("가장 작은 component", f"index {label_min}", f"{cc[ind_min, 1]} pixels")

# ---------------------------------------------------------------
# 6. 결과 비교
# ---------------------------------------------------------------
st.subheader("5. 결과 비교")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(f, cmap="gray")
axes[0].set_title("input image")
axes[1].imshow(labels == label_max, cmap=cmap_label)
axes[1].set_title("the largest component")
axes[2].imshow(labels == label_min, cmap=cmap_label)
axes[2].set_title("the smallest component")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
