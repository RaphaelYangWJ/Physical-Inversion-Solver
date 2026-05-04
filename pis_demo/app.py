import streamlit as st
import os
from PIL import Image

# --- 页面配置 ---
st.set_page_config(page_title="PIS Inversion Solver")

# --- 1. 标题与介绍 ---
st.title("Physical Inversion Solver (PIS) Sampling Trajectory")
st.markdown("""
Explore the deterministic straight-path probability flow of the **Set-Conditioned Flow Matching** framework.
The GIFs below show the full generation process, while the interactive sections allow for step-by-step inspection.
""")

# --- 2. 新增：标题下方的三个并排 GIF ---
# 请确保这三个 GIF 文件存在于你的 images 文件夹或根目录下
gif_paths = {
    "Subsurface": "images/subsurface_demo.gif",
    "Helmholtz": "images/helmholtz_demo.gif",
    "SHM": "images/shm_demo.gif"
}

st.markdown("### 🎬 Full Generation Demos")
gif_cols = st.columns(3)

for i, (name, path) in enumerate(gif_paths.items()):
    with gif_cols[i]:
        try:
            # Streamlit 直接支持展示 GIF
            st.image(path, caption=f"{name} Process", use_container_width=True)
        except Exception:
            st.warning(f"GIF not found at: {path}")

st.markdown("---")  # 分割线

# --- 3. 定义场景和对应的文件夹路径 (交互部分) ---
scenarios = {
    "Subsurface Characterization (Darcy Flow)": "images/darcyflow",
    "Wave-based Characterization (Helmholtz)": "images/helmholtz",
    "Structural Health Monitoring (SHM)": "images/shm"
}

max_step = 90  # 对应 frame_000 到 frame_090
preview_steps = [0, 22, 45, 67, 90]

# --- 4. 遍历并渲染交互场景 ---
for name, folder in scenarios.items():
    st.header(name)

    # 创建上方主图的占位符
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        main_image_placeholder = st.empty()

    # 在下方创建滑动条
    step_idx = st.slider(
        f"Drag to control sampling time ($t$) for {name}",
        min_value=0,
        max_value=max_step,
        value=0,
        format="Frame %d",
        key=f"slider_{name}"
    )

    # 计算并显示当前的 t 值
    current_t = step_idx / max_step
    st.markdown(f"**Current State:** $t = {current_t:.2f}$ ({step_idx}/{max_step})")

    # 更新上方占位符中的图片
    img_name = f"frame_{step_idx:03d}.png"
    img_path = os.path.join(folder, img_name)
    try:
        img = Image.open(img_path)
        main_image_placeholder.image(img, caption=f"{name} at t={current_t:.2f}", use_container_width=True)
    except FileNotFoundError:
        main_image_placeholder.error(f"⚠️ Image not found: `{img_path}`")

    # 底部 5 张静态缩略图
    st.markdown("##### Sampling Trajectory Preview")
    preview_cols = st.columns(5)

    for i, p_step in enumerate(preview_steps):
        p_img_name = f"frame_{p_step:03d}.png"
        p_img_path = os.path.join(folder, p_img_name)
        p_t = p_step / max_step

        with preview_cols[i]:
            try:
                p_img = Image.open(p_img_path)
                st.image(p_img, caption=f"t = {p_t:.2f}", use_container_width=True)
            except FileNotFoundError:
                st.error(f"Missing `{p_img_name}`")

    st.markdown("---")