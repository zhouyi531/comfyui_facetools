from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 已移除自动预加载功能，模型将在首次使用时按需加载
# 如果需要在后台预加载以减少首次调用延迟，可通过配置启用
try:
    from . import config
    if getattr(config, 'ENABLE_PRELOAD', False) and getattr(config, 'PRELOAD_ON_STARTUP', False):
        from .utils import Models
        # 异步后台预加载，不阻塞导入
        Models.start_preload()
except Exception as _e:
    # 预加载失败不应影响节点注册
    print(f"[comfyui_facetools] Background preload skipped: {_e}")
