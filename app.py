import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt

# ================== 字体加载（放在 set_page_config 之后） ==================
font_path = "static/NotoSansCJKsc-Regular.otf"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    
except FileNotFoundError:
    st.warning(" 字体文件未找到，图表将使用默认英文字体")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 页面配置与标题
# ============================================
st.set_page_config(page_title="最小二乘法：矩阵运算可视化", layout="wide")
st.title("📊 普通最小二乘法（OLS）：矩阵运算可视化与推导")
st.markdown("""
本应用通过**交互式图表**和**逐步数学推导**，展示如何利用矩阵运算求解线性回归参数，
并直观呈现 **“残差平方和（RSS）最小化”** 的过程。  
👉 左侧控制面板可拖动参数或播放动画，观察拟合直线、残差、RSS曲面与梯度变化。
""")

# ============================================
# 数据准备（与原始代码一致）
# ============================================
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 4])
X = np.c_[np.ones_like(x), x]          # 设计矩阵 X (4x2)

# 最优解（正规方程精确解）
XtX = X.T @ X
Xty = X.T @ y
beta_opt = np.linalg.solve(XtX, Xty)   # [1.5, 0.8]

# 初始猜测（动画起点）
beta_init = np.array([0.0, 0.0])

# 定义残差平方和函数（RSS）
def rss(beta, X, y):
    y_pred = X @ beta
    return np.sum((y - y_pred) ** 2)

# 定义梯度 ∇RSS = -2 X^T (y - X β)
def gradient(beta, X, y):
    residuals = y - X @ beta
    return -2 * X.T @ residuals

# 预计算用于绘制 RSS 曲面的网格
b0_range = np.linspace(-1.0, 3.5, 80)   # β0 范围
b1_range = np.linspace(-0.5, 2.0, 80)   # β1 范围
B0, B1 = np.meshgrid(b0_range, b1_range)
Z = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        Z[i, j] = rss([B0[i, j], B1[i, j]], X, y)

# ============================================
# 右侧辅助区域：矩阵运算与理论推导（静态展示）
# ============================================
with st.sidebar:
    st.header("🧠 矩阵运算与推导精要")
    with st.expander("📐 1. 模型矩阵表示", expanded=True):
            st.latex(r"\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}")
            st.markdown(f"""
            **设计矩阵 X** (形状 4×2)：  
            `[[1, 1],`  
            ` [1, 2],`  
            ` [1, 3],`  
            ` [1, 4]]`  
            **观测向量 y**：{y.tolist()}ᵀ
            """)

    with st.expander("📉 2. 残差平方和 (RSS) 矩阵形式", expanded=True):
            st.latex(
                r"S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})")
            st.markdown("展开得：")
            st.latex(
                r"S(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}")

    with st.expander("⚙️ 3. 最小化推导 (求导 = 0)", expanded=True):
            # 修正：添加 r 前缀
            st.markdown(r"对 $\boldsymbol{\beta}$ 求梯度并令其为零：")
            st.latex(
                r"\frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0")
            st.latex(r"\Rightarrow \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}")
            st.markdown("**正规方程** 的解即为最小二乘估计量：")
            st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}")

    with st.expander("🔢 4. 当前数据正规方程数值解", expanded=True):
            # 使用 st.latex 分别显示，避免换行问题
            st.latex(rf"\mathbf{{X}}^T\mathbf{{X}} = {np.round(XtX, 2).tolist()}")
            st.latex(rf"\mathbf{{X}}^T\mathbf{{y}} = {np.round(Xty, 2).tolist()}")
            st.latex(rf"\hat{{\boldsymbol{{\beta}}}} = [{beta_opt[0]:.4f},\ {beta_opt[1]:.4f}]^T")
            st.success("当参数取上述值时，梯度为零，RSS 达到全局最小值。")

# ============================================
# 左侧控制栏：参数调节与动画
# ============================================
col_left, col_right = st.columns([0.35, 0.65], gap="medium")

with col_left:
    st.header("🎮 交互控制台")
    control_type = st.radio(
        "参数调节方式",
        ["🎚️ 直接拖动 β", "▶️ 动画演示 (从初始到最优)"],
        index=0
    )

    if control_type == "🎚️ 直接拖动 β":
        beta0 = st.slider("截距 β₀", -1.0, 3.5, 0.0, 0.05, format="%.2f")
        beta1 = st.slider("斜率 β₁", -0.5, 2.0, 0.0, 0.05, format="%.2f")
        beta_current = np.array([beta0, beta1])
        progress = None   # 无动画进度

    else:  # 动画模式 - 仅使用滑动条手动控制进度，移除自动播放避免无限重运行
        progress = st.slider("动画进度 (从初值 → 最优)", 0.0, 1.0, 0.0, 0.01)
        beta_current = beta_init + (beta_opt - beta_init) * progress
        st.metric("当前 β₀", f"{beta_current[0]:.3f}", delta=f"{beta_current[0]-beta_opt[0]:+.3f}")
        st.metric("当前 β₁", f"{beta_current[1]:.3f}", delta=f"{beta_current[1]-beta_opt[1]:+.3f}")

    # 显示当前RSS和梯度信息
    current_rss = rss(beta_current, X, y)
    grad = gradient(beta_current, X, y)
    grad_norm = np.linalg.norm(grad)

    col1, col2 = st.columns(2)
    col1.metric("残差平方和 (RSS)", f"{current_rss:.4f}", delta=f"{current_rss - rss(beta_opt, X, y):+.4f}")
    col2.metric("梯度范数 ||∇RSS||", f"{grad_norm:.4f}", delta="→0 为最优" if grad_norm < 0.1 else "远离最优")

    st.markdown("**当前残差向量 e = y - Xβ**")
    y_pred = X @ beta_current
    residuals = y - y_pred
    st.write(np.round(residuals, 4))
    st.caption(f"梯度 = -2 Xᵀ e = {np.round(grad, 4).tolist()}")  # 修改此行

    # 辅助理解：矩阵运算RSS的另一个计算形式
    with st.expander("📐 当前β下的矩阵计算细节"):
        st.markdown(f"**ŷ = Xβ** = {np.round(y_pred, 4).tolist()}")
        st.markdown(f"**残差 e = y - ŷ** = {np.round(residuals, 4).tolist()}")
        # 修正：使用原始 f-string
        st.latex(rf"RSS = e^T e = {current_rss:.4f}")

# ============================================
# 右侧可视化区域：拟合图、残差、RSS曲面与路径
# ============================================
with col_right:
    tab1, tab2, tab3 = st.tabs(["📈 拟合直线与残差", "🗺️ RSS 曲面与梯度下降路径", "📜 矩阵运算流程推导"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(x, y, color='blue', s=80, zorder=5, label='观测数据')
        x_line = np.linspace(0, 5, 100)
        y_line = beta_current[0] + beta_current[1] * x_line
        ax1.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'拟合线: y = {beta_current[0]:.2f} + {beta_current[1]:.2f}x')
        # 绘制残差线
        for xi, yi, ypi in zip(x, y, y_pred):
            ax1.plot([xi, xi], [yi, ypi], 'k--', alpha=0.7, linewidth=1.2)
        ax1.scatter(x, y_pred, color='orange', s=60, marker='x', zorder=4, label='拟合值 ŷ')
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 6)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'数据点与拟合直线 (RSS = {current_rss:.3f})')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        st.caption("黑色虚线表示残差（误差）长度，RSS 即为所有虚线长度平方之和。")

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        contour = ax2.contour(B0, B1, Z, levels=20, cmap='viridis', alpha=0.7)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.plot(beta_opt[0], beta_opt[1], 'r*', markersize=12, label='最优解 $\\hat{{\\beta}}$')
        ax2.plot(beta_current[0], beta_current[1], 'bo', markersize=8, label='当前 $\\beta$')
        # 若动画模式且进度>0，绘制从初始到当前的路径线
        if control_type == "▶️ 动画演示" and progress is not None and progress > 0:
            path_betas = [beta_init + (beta_opt - beta_init) * t for t in np.linspace(0, progress, 30)]
            path_b0 = [b[0] for b in path_betas]
            path_b1 = [b[1] for b in path_betas]
            ax2.plot(path_b0, path_b1, 'c--', linewidth=2, alpha=0.8, label='参数移动路径')
        ax2.set_xlabel('$\\beta_0$ (截距)')
        ax2.set_ylabel('$\\beta_1$ (斜率)')
        ax2.set_title('残差平方和 RSS(β) 等高线图\n颜色越深 RSS 越小')
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        st.markdown(r"""
        **解释**：RSS 是一个凸二次函数，其最小值点正是正规方程的解。  
        梯度 $\nabla RSS$ 指向 RSS 增长最快的方向，其相反方向即为下降方向。  
        最优解处梯度为零（红五角星）。
        """)

    with tab3:
        st.markdown(r"""
        #### 🧩 从矩阵运算到正规方程 (一步一步推导)
        1. **将线性模型写为矩阵形式**  
           $$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$
        2. **残差平方和**  
           $$S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$
        3. **展开得到**  
           $$S(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$
        4. **对 $\boldsymbol{\beta}$ 求导 (梯度)**  
           $$\frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$
        5. **令导数为零，得到正规方程**  
           $$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$
        6. **若 $\mathbf{X}^T\mathbf{X}$ 可逆，则最小二乘估计量为**  
           $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
        """)
        # 展示当前数据矩阵运算中间结果
        st.markdown("#### 🧮 本例中的矩阵数值计算过程")
        st.code(f"""
X = {X.tolist()}
y = {y.tolist()}

X^T X = {np.round(XtX, 2).tolist()}
X^T y = {np.round(Xty, 2).tolist()}

求解正规方程: (X^T X) β = X^T y
β_hat = {np.round(beta_opt, 4).tolist()}
""", language='python')
        st.success("上述解使得残差平方和达到最小，且梯度范数为零。")

# ============================================
# 页脚 & 说明
# ============================================
st.markdown("---")
st.markdown("""
**💡 使用提示**  
- **直接拖动 β**：实时观察拟合直线变化，RSS 和梯度的变化会立刻反映。  
- **动画演示模式**：从 (0,0) 线性移动到最优解，可在 RSS 等高线中看到参数轨迹（滑动进度条）。  
- 右侧**矩阵运算流程推导**展示了普通最小二乘估计量的完整代数推导，配合数值实例加深理解。  
- 梯度范数接近 0 时表示当前参数已接近最小二乘解。
""")
st.caption("✨ 可视化基于最小二乘法的矩阵运算核心思想：通过正规方程一步得到最优参数，并使残差平方和最小。")
