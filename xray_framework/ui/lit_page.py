"""文献导入与算法迭代请求页面

流程：
  1. 导入文献（多文件上传 / 粘贴文本）→ 存入 literature/ 并建立索引
  2. 在文献库中查看并预览自动提取的代码块
  3. 添加外部模型到框架（上传 .py 定义文件，持久注册）
  4. 创建算法迭代请求：
     - 改进现有模型（选基类）
     - 新建模型（基于引用文献从零设计）
  5. 由 CodeBuddy 读取请求，实现新算法并注册到框架
"""
import os
import tempfile

import streamlit as st

from ..literature import (
    list_literature,
    list_iteration_requests,
    save_literature,
    save_literature_text,
    create_iteration_request,
    mark_request_status,
    get_literature_path,
)
from ..literature.extract import extract_text, extract_code_blocks
from ..models.registry import list_models, discover, add_external_model

REQ_STATUS = {"pending": "待实现", "done": "已完成"}
REQ_TYPE = {"improve": "改进现有模型", "new": "新建模型"}


def render():
    st.header("文献导入与算法迭代")
    st.caption(
        "导入参考文献（可多篇）→ 自动提取其中的代码 → 选择「改进现有模型」或「新建模型」，"
        "生成算法迭代请求。CodeBuddy 读取请求后即会实现新算法并注册到框架。"
    )

    _render_import()
    _render_library()
    _render_extract()
    _render_external_model()
    _render_request_form()
    _render_request_list()


# ---------------------------------------------------------------------------
def _render_import():
    st.subheader("1️⃣ 导入文献")
    with st.expander("上传文献文件（支持 PDF / TXT / MD / PY，可多选）", expanded=True):
        files = st.file_uploader(
            "文献文件",
            type=["pdf", "txt", "md", "py"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="lit_files",
        )
        if st.button("保存上传的文献", disabled=not files):
            ok, fail = 0, 0
            for f in files or []:
                try:
                    save_literature(f.name, f.getbuffer().tobytes())
                    ok += 1
                except Exception as e:
                    st.error(f"保存 {f.name} 失败：{e}")
                    fail += 1
            if ok:
                st.success(f"已保存 {ok} 篇文献" + (f"，{fail} 篇失败" if fail else ""))
                st.rerun()

    with st.expander("或直接粘贴文献文本"):
        paste_name = st.text_input("文献名称（用于保存，缺省为粘贴.txt）", value="")
        paste_text = st.text_area("文献内容（可直接粘贴含代码的 Markdown/论文文本）", height=200)
        if st.button("保存粘贴的文本", disabled=not paste_text.strip()):
            try:
                save_literature_text(paste_name.strip() or "粘贴文献.txt", paste_text)
                st.success("文本已保存到文献库")
                st.rerun()
            except Exception as e:
                st.error(f"保存失败：{e}")


# ---------------------------------------------------------------------------
def _render_library():
    st.subheader("2️⃣ 文献库")
    records = list_literature()
    if not records:
        st.info("暂无文献，请先导入。")
        return
    st.dataframe(
        [
            {
                "标题": r["title"],
                "文件": r["file"],
                "格式": r["ext"],
                "大小(KB)": round(r["size"] / 1024, 1),
                "代码块数": len(r["code_blocks"]),
                "导入时间": r["imported_at"],
            }
            for r in records
        ],
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
def _render_extract():
    st.subheader("3️⃣ 提取代码预览")
    records = list_literature()
    if not records:
        return
    options = {r["file"]: f"{r['title']}（{r['file']}）" for r in records}
    selected = st.selectbox("选择文献查看提取的代码", list(options.values()))
    file_name = next(k for k, v in options.items() if v == selected)
    text = extract_text(get_literature_path(file_name))
    blocks = extract_code_blocks(text)
    if not blocks:
        st.info("未在文本中提取到 ```python 代码块。可检查原始内容：")
        with st.expander("查看全文"):
            st.code(text[:6000], language=None)
        return
    st.write(f"提取到 **{len(blocks)}** 个代码块：")
    for i, block in enumerate(blocks, 1):
        with st.expander(f"代码块 {i}（{len(block.splitlines())} 行）"):
            st.code(block, language="python")


# ---------------------------------------------------------------------------
def _render_external_model():
    st.subheader("4️⃣ 添加外部模型到框架")
    st.caption("上传模型定义文件（.py）即可持久注册为框架可用算法，重启后仍然有效。")
    with st.expander("从外部导入模型（.py）并注册", expanded=False):
        col1, col2 = st.columns(2)
        py_up = col1.file_uploader(
            "模型定义文件 (.py)", type=["py"], key="ext_py", label_visibility="collapsed"
        )
        w_up = col2.file_uploader(
            "权重文件 (.pth，可选)", type=["pth", "pt"], key="ext_weights",
            label_visibility="collapsed",
        )
        col3, col4 = st.columns(2)
        class_name = col3.text_input(
            "模型类名（可选，自动检测继承 BaseImageModel 的类）", key="ext_class"
        )
        reg_name = col4.text_input("注册名（可选，默认类名小写）", key="ext_reg")

        if st.button("导入并注册", disabled=py_up is None):
            tmp_py = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
                    tf.write(py_up.getbuffer())
                    tmp_py = tf.name
                name = add_external_model(
                    tmp_py,
                    registry_name=reg_name.strip(),
                    class_name=class_name.strip(),
                )
                # 可选权重：保存到 outputs/models/external/<注册名>.pth
                weights_saved = ""
                if w_up is not None:
                    wdir = os.path.join("outputs", "models", "external")
                    os.makedirs(wdir, exist_ok=True)
                    weights_saved = os.path.join(wdir, f"{name}.pth")
                    with open(weights_saved, "wb") as f:
                        f.write(w_up.getbuffer())
                st.success(f"外部模型已注册：`{name}`，可在训练/推理的模型列表中选择。")
                if weights_saved:
                    st.info(f"权重已保存：`{weights_saved}`")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败：{e}")
            finally:
                if tmp_py and os.path.exists(tmp_py):
                    os.unlink(tmp_py)

    # 已注册的外部模型一览
    from ..models.registry import list_external_models
    exts = list_external_models()
    if exts:
        st.write(f"已注册外部模型（{len(exts)} 个）：")
        st.dataframe(
            [{"注册名": r["registry_name"], "文件": r["file"], "类名": r["class_name"]} for r in exts],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
def _render_request_form():
    st.subheader("5️⃣ 创建算法迭代请求")
    discover()
    models = list_models()
    records = list_literature()

    req_type = st.radio(
        "请求类型",
        ["改进现有模型", "新建模型"],
        horizontal=True,
        help="改进：在现有模型基础上迭代；新建：基于引用文献从零设计全新模型",
    )
    col1, col2 = st.columns(2)
    with col1:
        if req_type == "改进现有模型":
            base_model = st.selectbox(
                "要改进的现有模型",
                models,
                help="迭代后的新算法将以该模型为基类/参考，注册为独立的新算法",
            )
        else:
            base_model = ""
            st.selectbox(
                "要改进的现有模型",
                ["（新建模型，无需选择基类）"],
                disabled=True,
            )
    with col2:
        ref_files = st.multiselect(
            "引用文献（支持多选）",
            [r["file"] for r in records],
            help="选择本次算法依据的文献；也可不选，仅在下方描述目标",
        )
    goal = st.text_area(
        "算法目标描述",
        placeholder=(
            "新建模型示例：根据文献中的 Transformer 架构，设计一个端到端的平板X射线去混叠网络…\n"
            "改进示例：在 UNet 编码器中引入文献的通道注意力模块…"
        ),
        height=120,
    )
    notes = st.text_input("备注（可选）", value="")
    if st.button("生成迭代请求", disabled=not goal.strip(), type="primary"):
        # 收集所选文献的全部代码块供 AI 参考
        snippets = []
        for r in records:
            if r["file"] in ref_files:
                snippets.extend(r["code_blocks"])
        request = create_iteration_request(
            base_model=base_model,
            goal=goal,
            ref_files=ref_files,
            code_snippets=snippets,
            notes=notes,
            request_type="new" if req_type == "新建模型" else "improve",
        )
        st.success(f"迭代请求已生成：`{request['id']}`，状态「待实现」。")
        st.info(
            "下一步：在对话中告诉 CodeBuddy “处理迭代请求 `<id>`”，"
            "即可根据文献实现新算法并注册到框架。"
        )
        st.rerun()


# ---------------------------------------------------------------------------
def _render_request_list():
    st.subheader("6️⃣ 迭代请求列表")
    requests = list_iteration_requests()
    if not requests:
        st.info("暂无迭代请求。")
        return
    for req in requests:
        status_zh = REQ_STATUS.get(req.get("status", "pending"), req.get("status"))
        type_zh = REQ_TYPE.get(req.get("request_type", "improve"), req.get("request_type"))
        base = req.get("base_model") or "（全新）"
        with st.expander(
            f"[{status_zh}] {req['id']} · {type_zh} · {base} · {req.get('created_at')}"
        ):
            st.markdown(f"**目标**：{req.get('goal', '')}")
            if req.get("notes"):
                st.markdown(f"**备注**：{req['notes']}")
            refs = req.get("references", [])
            if refs:
                st.markdown("**引用文献**：")
                for ref in refs:
                    st.markdown(f"- {ref.get('title')}（{ref.get('file')}）")
            if req.get("code_snippets"):
                with st.expander(f"引用代码片段（{len(req['code_snippets'])} 个）"):
                    for i, block in enumerate(req["code_snippets"], 1):
                        st.code(block[:3000], language="python")
            if req.get("status") == "pending":
                if st.button("标记为已完成", key=f"done_{req['id']}"):
                    mark_request_status(req["id"], "done")
                    st.success("已标记")
                    st.rerun()
