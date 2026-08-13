"""推理板块 UI"""
import os

import numpy as np
import streamlit as st
from PIL import Image

from ..data.image_io import list_images, read_image, save_image, to_uint8
from ..inference.inferencer import Inferencer
from ..utils.metrics import evaluate_pair


def _find_local_models(base_dir: str) -> list:
    """扫描本地已训练模型 (.pth)"""
    if not base_dir or not os.path.isdir(base_dir):
        return []
    found = []
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith((".pth", ".pt")):
                found.append(os.path.join(root, f))
    return found


def _display_comparison(input_img: np.ndarray, output_img: np.ndarray,
                        truth_img: np.ndarray = None, caption: str = ""):
    c1, c2, c3 = st.columns(3 if truth_img is not None else 2)
    c1.image(to_uint8(input_img), caption="输入（平板源，含混叠）", use_container_width=True)
    c2.image(to_uint8(output_img), caption="输出（去混叠）", use_container_width=True)
    if truth_img is not None:
        c3.image(to_uint8(truth_img), caption="真值（点源）", use_container_width=True)


def render_infer_page():
    st.header("模型推理")
    st.caption("加载本地/外部模型，对输入图片进行去混叠处理")

    # ---------- 模型加载 ----------
    st.subheader("1. 加载模型")
    src_type = st.radio("模型来源", ["框架训练输出模型", "外部模型 (.py + 权重)", "TorchScript", "ONNX"],
                        horizontal=True)

    inferencer = Inferencer()
    model_loaded = False

    if src_type == "框架训练输出模型":
        default_dir = os.path.join("outputs", "models")
        mdir = st.text_input("模型目录", value=default_dir)
        local_models = _find_local_models(mdir)
        if not local_models:
            st.warning("该目录下没有找到 .pth 模型文件，请检查路径或先到「模型训练」板块训练。")
        else:
            model_path = st.selectbox("选择模型", local_models,
                                      format_func=lambda p: os.path.basename(p))
            if st.button("加载模型", use_container_width=True):
                try:
                    name = inferencer.load_local_model(model_path)
                    st.session_state["_inferencer"] = inferencer
                    model_loaded = True
                    st.success(f"模型加载成功：{name}")
                except Exception as e:
                    st.error(f"模型加载失败：{e}")

    elif src_type == "外部模型 (.py + 权重)":
        st.caption("外部模型：提供一个含模型定义的 .py 文件（类需继承 nn.Module，可用 @register_model 注册），"
                   "以及可选的权重文件 (.pth)。")
        py_path = st.text_input("模型定义 .py 路径")
        weights_path = st.text_input("权重文件路径（可选）", value="")
        class_name = st.text_input("模型类名（未注册时需填写，可选）", value="")
        registry_name = st.text_input("注册名（可选，若 .py 中用了 @register_model）", value="")
        if st.button("导入并加载外部模型", use_container_width=True):
            if not py_path or not os.path.isfile(py_path):
                st.error("请填写有效的模型定义文件路径")
            else:
                try:
                    name = inferencer.load_external_model(
                        py_path, class_name=class_name or None,
                        weights_path=weights_path or None,
                        registry_name=registry_name or None)
                    st.session_state["_inferencer"] = inferencer
                    model_loaded = True
                    st.success(f"外部模型加载成功：{name}")
                except Exception as e:
                    st.error(f"外部模型加载失败：{e}")

    elif src_type == "TorchScript":
        ts_path = st.text_input("TorchScript 模型路径 (.pt)")
        if st.button("加载 TorchScript 模型", use_container_width=True):
            if not ts_path or not os.path.isfile(ts_path):
                st.error("请填写有效的模型路径")
            else:
                try:
                    inferencer.load_torchscript(ts_path)
                    st.session_state["_inferencer"] = inferencer
                    model_loaded = True
                    st.success("TorchScript 模型加载成功")
                except Exception as e:
                    st.error(f"加载失败：{e}")

    elif src_type == "ONNX":
        onnx_path = st.text_input("ONNX 模型路径 (.onnx)")
        if st.button("加载 ONNX 模型", use_container_width=True):
            if not onnx_path or not os.path.isfile(onnx_path):
                st.error("请填写有效的 ONNX 模型路径")
            else:
                try:
                    inferencer.load_onnx(onnx_path)
                    st.session_state["_inferencer"] = inferencer
                    model_loaded = True
                    st.success("ONNX 模型加载成功（需已安装 onnxruntime）")
                except Exception as e:
                    st.error(f"加载失败：{e}")

    # ---------- 推理 ----------
    st.divider()
    st.subheader("2. 图片处理")
    inferencer = st.session_state.get("_inferencer")
    if inferencer is None:
        st.info("请先在上方加载模型")
        return

    st.success(f"当前模型：`{inferencer.model_name}`")

    mode = st.radio("处理方式", ["单张图片", "批量处理文件夹"], horizontal=True)

    if mode == "单张图片":
        up = st.file_uploader("上传图片（PNG/TIFF/JPG）", type=["png", "tif", "tiff", "jpg", "jpeg", "bmp"])
        truth_up = st.file_uploader("可选：上传真值图片用于评估 PSNR/SSIM", type=["png", "tif", "tiff", "jpg", "jpeg", "bmp"])
        save16 = st.checkbox("保存 16 位 TIFF", value=False)
        if st.button("执行推理", type="primary", use_container_width=True):
            if up is None:
                st.warning("请先上传图片")
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as tf:
                    tf.write(up.getbuffer())
                    tmp_path = tf.name
                try:
                    arr = read_image(tmp_path, size=inferencer.image_size)
                    out = inferencer.infer_array(arr)
                    st.success("推理完成！")
                    truth_arr = None
                    if truth_up is not None:
                        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(truth_up.name)[1], delete=False) as tf2:
                            tf2.write(truth_up.getbuffer())
                            truth_arr = read_image(tf2.name, size=inferencer.image_size)
                        m = evaluate_pair(truth_arr, out)
                        st.metric("PSNR (dB)", m["PSNR (dB)"])
                        st.metric("SSIM", m["SSIM"])
                    _display_comparison(arr, out, truth_arr)
                    save_dir = st.text_input("保存目录", value=os.path.join("outputs", "results"))
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, f"dealiased_{up.name}")
                    save_image(out, out_path, bit16=save16)
                    st.success(f"已保存：`{out_path}`")
                except Exception as e:
                    st.error(f"推理失败：{e}")
                finally:
                    os.unlink(tmp_path)

    else:  # 批量
        in_dir = st.text_input("输入文件夹路径")
        out_dir = st.text_input("输出文件夹路径", value=os.path.join("outputs", "results"))
        save16 = st.checkbox("保存 16 位 TIFF", value=False)
        if st.button("开始批量处理", type="primary", use_container_width=True):
            if not in_dir or not os.path.isdir(in_dir):
                st.error("请输入有效的输入文件夹路径")
            else:
                files = list_images(in_dir)
                if not files:
                    st.warning("输入文件夹中没有找到图像文件")
                else:
                    progress = st.progress(0.0)
                    status = st.empty()
                    results = {}
                    for i, f in enumerate(files):
                        status.write(f"处理中 ({i+1}/{len(files)}): {f}")
                        try:
                            arr = read_image(os.path.join(in_dir, f), size=inferencer.image_size)
                            out = inferencer.infer_array(arr)
                            save_path = os.path.join(out_dir, f)
                            save_image(out, save_path, bit16=save16)
                            results[f] = save_path
                        except Exception as e:
                            st.warning(f"{f} 处理失败：{e}")
                        progress.progress((i + 1) / len(files))
                    status.success(f"批量处理完成：成功 {len(results)} / {len(files)} 张")
                    st.success(f"结果已保存到：`{out_dir}`")
