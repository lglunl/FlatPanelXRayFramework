"""模型注册表与自动发现机制

这是「算法可迭代」的核心：
  - 用 @register_model 注册模型类
  - 新算法文件放入 algorithms/ 目录会被自动 import 并注册
  - 外部模型定义放入 external/ 目录 + external_registry.json 记录后持久注册
  - UI 中模型下拉框自动读取注册表，新增算法无需改任何 UI 代码
"""
import importlib
import json
import os
import pkgutil
import shutil
import traceback

MODEL_REGISTRY = {}

# 内置模型列表（会被自动导入）
_BUILTIN_MODULES = ["unet", "unet_plus", "baselines"]

_PKG = "xray_framework.models"
_EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "external")
_REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "external_registry.json")


def register_model(name=None):
    """模型注册装饰器"""
    def deco(cls):
        key = (name or cls.__name__).lower()
        MODEL_REGISTRY[key] = cls
        return cls
    return deco


def _import_module(module_name: str):
    try:
        importlib.import_module(module_name)
    except Exception as e:
        traceback.print_exc()
        print(f"[registry] 加载模块 {module_name} 失败: {e}")


def discover():
    """扫描内置、algorithms 与 external 目录中的所有模型模块并注册"""
    pkg_dir = os.path.dirname(__file__)
    pkg_name = __name__.rsplit(".", 1)[0]  # xray_framework.models

    for mod in _BUILTIN_MODULES:
        _import_module(f"{pkg_name}.{mod}")

    # 自动发现 algorithms 子目录下的所有 .py
    algo_dir = os.path.join(pkg_dir, "algorithms")
    if os.path.isdir(algo_dir):
        for f in sorted(os.listdir(algo_dir)):
            if f.startswith("_") or not f.endswith(".py"):
                continue
            mod_name = f[:-3]
            _import_module(f"{pkg_name}.algorithms.{mod_name}")

    # 加载外部模型（依据 external_registry.json 持久注册）
    _discover_external()


# ---------------------------------------------------------------------------
# 外部模型：通过界面/接口持久注册到框架
# ---------------------------------------------------------------------------

def _load_external_records() -> list:
    if not os.path.isfile(_REGISTRY_FILE):
        return []
    try:
        with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_external_records(records: list):
    os.makedirs(os.path.dirname(_REGISTRY_FILE), exist_ok=True)
    with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _discover_external():
    """按 external_registry.json 中的记录加载外部模型并注册"""
    for rec in _load_external_records():
        file_name = rec.get("file", "")
        if not file_name.endswith(".py"):
            continue
        try:
            mod = importlib.import_module(f"{_PKG}.external.{file_name[:-3]}")
            cls = getattr(mod, rec["class_name"])
            MODEL_REGISTRY.setdefault(rec["registry_name"].lower(), cls)
        except Exception as e:
            traceback.print_exc()
            print(f"[registry] 外部模型加载失败: {rec.get('registry_name')}: {e}")


def add_external_model(
    py_path: str,
    registry_name: str = "",
    class_name: str = "",
) -> str:
    """持久化添加一个外部模型定义文件并注册到框架。

    步骤：复制 .py 到 external/ 目录 → 导入模块 → 定位模型类
          → 注册到 MODEL_REGISTRY → 写入 external_registry.json
    返回注册名；失败抛异常。
    """
    from .base import BaseImageModel  # 局部导入避免循环依赖

    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"模型定义文件不存在: {py_path}")

    # 1. 复制到 external 目录（重名则覆盖）
    os.makedirs(_EXTERNAL_DIR, exist_ok=True)
    dst_name = os.path.basename(py_path)
    dst_path = os.path.join(_EXTERNAL_DIR, dst_name)
    if os.path.abspath(py_path) != os.path.abspath(dst_path):
        shutil.copy2(py_path, dst_path)

    # 2. 导入模块
    mod_name = f"{_PKG}.external.{dst_name[:-3]}"
    mod = importlib.import_module(mod_name)

    # 3. 定位模型类：优先用户指定，否则自动查找继承 BaseImageModel 的类
    cls = None
    if class_name:
        cls = getattr(mod, class_name)
    if cls is None:
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseImageModel)
                and obj is not BaseImageModel
            ):
                cls = obj
                break
    if cls is None:
        raise ValueError(
            "未找到模型类：请提供 class_name，或确保文件中定义了继承 BaseImageModel 的类"
        )
    if not (isinstance(cls, type) and issubclass(cls, BaseImageModel)):
        raise TypeError(f"{getattr(cls, '__name__', class_name)} 必须继承 BaseImageModel")

    # 4. 注册
    key = (registry_name or cls.__name__).lower()
    MODEL_REGISTRY[key] = cls

    # 5. 持久化注册信息
    records = _load_external_records()
    records = [
        r for r in records
        if not (r.get("file") == dst_name and r.get("registry_name") == key)
    ]
    records.append(
        {"file": dst_name, "class_name": cls.__name__, "registry_name": key}
    )
    _save_external_records(records)
    return key


def list_external_models() -> list:
    """返回已持久注册的外部模型记录"""
    return _load_external_records()


def list_models() -> list:
    """返回所有可用模型名"""
    if not MODEL_REGISTRY:
        discover()
    return sorted(MODEL_REGISTRY.keys())


def get_model(name: str, **kwargs) -> "nn.Module":
    """按名称创建模型实例"""
    if not MODEL_REGISTRY:
        discover()
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"未知模型 '{name}'，可用: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def model_info(name: str) -> str:
    """返回模型类的 docstring（用于 UI 展示）"""
    if not MODEL_REGISTRY:
        discover()
    cls = MODEL_REGISTRY.get(name.lower())
    return (cls.__doc__ or "暂无说明").strip() if cls else "未知模型"
