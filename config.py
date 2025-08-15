# ComfyUI FaceTools Configuration

# 模型预加载设置
ENABLE_PRELOAD = False  # 是否启用模型预加载
PRELOAD_ON_STARTUP = False  # 是否在ComfyUI启动时预加载
PRELOAD_TIMEOUT = 60  # 预加载超时时间（秒）

# YOLO模型设置
YOLO_WARMUP = False  # 是否预热YOLO模型
YOLO_DEVICE_AUTO = True  # 是否自动选择设备

# Landmarks模型设置
LANDMARKS_WARMUP = False  # 是否预热landmarks模型

# 性能设置
USE_HALF_PRECISION = False  # 是否使用半精度（可能会更快但精度稍低）
CACHE_MODELS = True  # 是否缓存模型

# 调试设置
VERBOSE_LOADING = True  # 是否显示详细的加载信息 