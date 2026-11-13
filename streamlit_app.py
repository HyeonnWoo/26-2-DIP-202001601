import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Homework 5 박현우 : Object Counting")

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.header("1. 입력 / 노이즈 제거")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
kernel_size = st.sidebar.slider("Median filter 커널 크기 (홀수)", 1, 31, 7, step=2)

st.sidebar.header("2. 이진화")
method = st.sidebar.radio("Threshold 방식", ["Otsu (자동)", "Fixed (수동)", "Adaptive (국소 평균)"])
th_val = 128
block = 51
offset = 5
if method == "Fixed (수동)":
    th_val = st.sidebar.slider("Threshold 값", 0, 255, 128)
elif method == "Adaptive (국소 평균)":
    block = st.sidebar.slider("Block size (홀수)", 3, 201, 51, step=2)
    offset = st.sidebar.slider("Offset C", -30, 30, 5)
polarity = st.sidebar.radio("객체 밝기", ["밝은 객체 (배경 어두움)", "어두운 객체 (배경 밝음)"])

st.sidebar.header("3. 후처리")
open_iter = st.sidebar.slider("Opening 반복 (작은 잡음 제거)", 0, 10, 1)
close_iter = st.sidebar.slider("Closing 반복 (끊어진 부분 연결)", 0, 10, 0)
fill_holes = st.sidebar.checkbox("내부 구멍 채우기", value=True)
min_area = st.sidebar.slider("최소 면적 (pixel) 미만 제거", 0, 2000, 30)
clear_border = st.sidebar.checkbox("이미지 경계에 걸친 객체 제외", value=False)

st.sidebar.header("4. 붙은 객체 분리")
use_ws = st.sidebar.checkbox("Watershed로 분리", value=False)
min_dist = st.sidebar.slider("객체 중심 간 최소 거리 (pixel)", 3, 100, 15, disabled=not use_ws)

cmap_label = st.sidebar.selectbox("Label colormap", ["Spectral_r", "nipy_spectral", "tab20", "viridis"], index=0)

if uploaded is None:
    st.info("왼쪽 사이드바에서 이미지를 업로드하세요.")
    st.stop()


# ---------------------------------------------------------------
# 처리 함수
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def denoise(f, k):
    return signal.medfilt2d(f, k)


def otsu_threshold(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.nanargmax(sigma_b))


def binarize(img, method, th_val, block, offset, bright_obj):
    x = img.astype(np.float32)
    if method == "Adaptive (국소 평균)":
        local = nd.uniform_filter(x, size=block, mode="reflect")
        mask = x > (local + offset) if bright_obj else x < (local - offset)
        return mask, None
    th = otsu_threshold(img) if method == "Otsu (자동)" else th_val
    mask = img > th if bright_obj else img <= th
    return mask, th


def postprocess(mask, open_iter, close_iter, fill_holes):
    if open_iter > 0:
        mask = nd.binary_opening(mask, iterations=open_iter)
    if close_iter > 0:
        mask = nd.binary_closing(mask, iterations=close_iter)
    if fill_holes:
        mask = nd.binary_fill_holes(mask)
    return mask


def split_touching(mask, min_dist):
    """Distance transform + watershed (scipy only)."""
    dist = nd.distance_transform_edt(mask)
    if dist.max() == 0:
        return nd.label(mask)[0]
    local_max = dist == nd.maximum_filter(dist, size=2 * min_dist + 1)
    peaks = local_max & mask & (dist > 1)
    markers, n = nd.label(peaks)
    if n == 0:
        return nd.label(mask)[0]
    d8 = (255 * (1 - dist / dist.max())).astype(np.uint8)
    lab = nd.watershed_ift(d8, markers.astype(np.int16))
    lab[~mask] = 0
    return lab


def filter_components(labels, min_area, clear_border):
    """작은 객체 / 경계 객체 제거 후 label을 1..N으로 재정렬."""
    n = labels.max()
    if n == 0:
        return labels
    areas = nd.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = areas >= min_area
    if clear_border:
        border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        for b in border:
            if b > 0:
                keep[b - 1] = False
    new = np.zeros_like(labels)
    new_id = 1
    for i in np.where(keep)[0]:
        new[labels == i + 1] = new_id
        new_id += 1
    return new


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
# 파이프라인
# ---------------------------------------------------------------
f = np.array(Image.open(uploaded).convert("L"))
f_denoised = denoise(f, kernel_size)

bright_obj = polarity.startswith("밝은")
mask_raw, th_used = binarize(f_denoised, method, th_val, block, offset, bright_obj)
mask = postprocess(mask_raw, open_iter, close_iter, fill_holes)

if use_ws:
    labels0 = split_touching(mask, min_dist)
else:
    labels0, _ = nd.label(mask)
labels = filter_components(labels0, min_area, clear_border)
nums = int(labels.max())

# ---------------------------------------------------------------
# 결과 표시
# ---------------------------------------------------------------
st.metric("검출된 객체 수", nums)

st.subheader("1. 원본 / 노이즈 제거")
c1, c2 = st.columns(2)
with c1:
    show_image(f, "Input image (grayscale)", colorbar=True)
with c2:
    show_image(f_denoised, f"Median filtered (k={kernel_size})", colorbar=True)

st.subheader("2. 히스토그램")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(f_denoised.ravel(), bins=256, range=(0, 256), color="gray")
if th_used is not None:
    ax.axvline(th_used, color="red", linestyle="--", label=f"threshold = {th_used}")
    ax.legend()
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Frequency")
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)
plt.close(fig)

st.subheader("3. 이진화 / 후처리")
c1, c2 = st.columns(2)
with c1:
    title = f"Binary mask ({method}" + (f", th={th_used})" if th_used is not None else ")")
    show_image(mask_raw, title)
with c2:
    show_image(mask, "After morphology / hole fill")

st.subheader("4. 레이블링 결과")
if nums == 0:
    st.warning("검출된 객체가 없습니다. Threshold 방식, 객체 밝기, 최소 면적을 확인하세요.")
    st.stop()

areas = np.array(nd.sum(np.ones_like(labels), labels, index=np.arange(1, nums + 1))).astype(int)
centers = nd.center_of_mass(np.ones_like(labels), labels, index=np.arange(1, nums + 1))

c1, c2 = st.columns(2)
with c1:
    show_image(labels, f"Labeled components ({nums})", cmap=cmap_label)
with c2:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(f, cmap="gray")
    ax.contour(labels > 0, levels=[0.5], colors="red", linewidths=0.8)
    for i, (cy, cx) in enumerate(centers, start=1):
        ax.text(cx, cy, str(i), color="yellow", fontsize=8, ha="center", va="center")
    ax.set_title("Overlay with index")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("5. Component별 픽셀 수")
c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(
        {"index": np.arange(1, nums + 1), "number of pixels": areas},
        use_container_width=True,
        height=min(35 * (nums + 1), 400),
    )
with c2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(areas, bins=min(30, max(5, nums)), color="steelblue")
    ax.axvline(np.median(areas), color="red", linestyle="--", label=f"median = {int(np.median(areas))}")
    ax.set_xlabel("Area (pixels)")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("면적이 비정상적으로 큰 객체는 여러 개가 붙은 것일 가능성이 높습니다. → Watershed 사용 권장")

ind_max = int(np.argmax(areas))
ind_min = int(np.argmin(areas))
m1, m2 = st.columns(2)
m1.metric("가장 큰 component", f"index {ind_max + 1}", f"{areas[ind_max]} pixels")
m2.metric("가장 작은 component", f"index {ind_min + 1}", f"{areas[ind_min]} pixels")

st.subheader("6. 결과 비교")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(f, cmap="gray")
axes[0].set_title("input image")
axes[1].imshow(labels == ind_max + 1, cmap=cmap_label)
axes[1].set_title("the largest component")
axes[2].imshow(labels == ind_min + 1, cmap=cmap_label)
axes[2].set_title("the smallest component")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
