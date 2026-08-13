"""模型注册表与自动发现机制

这是「算法可迭代」的核心：
  - 用 @register_model 注册模型类
  - 新算法文件放入 algorithms/ 目录会被自动 import 并注册
  - UI 中模型下拉框自动读取注册表，新增算法无需改任何 UI 代码
"""
import importlib
import os
import pkgutil
import traceback

MODEL_REGISTRY = {}

# 内置模型列表（会被自动导入）
_BUILTIN_MODULES = ["unet", "unet_plus", "baselines"]


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
    """扫描内置与 algorithms 目录中的所有模型模块并注册"""
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
