# ONNX Inference Project

## ⚠️ 注意事项

1. **环境配置**  
   需要提前配置以下环境：
   - OpenCV 4.5+（用于图像处理）
   - ONNX Runtime 1.8+（用于模型推理）

2. **运行时依赖**  
   必须将以下文件复制到生成的 `exe` 同级目录：
   - `onnxruntime.dll`（来自 ONNX Runtime 安装目录）
   - `opencv_world455.dll`（来自 OpenCV 安装目录）

## 环境配置步骤

### Windows 系统
```powershell
# 设置 OpenCV 环境变量
setx OPENCV_DIR "C:\opencv\build\x64\vc15"
setx PATH "%PATH%;%OPENCV_DIR%\bin"

# 设置 ONNX Runtime 环境变量
setx ONNXRUNTIME_DIR "C:\onnxruntime"
setx PATH "%PATH%;%ONNXRUNTIME_DIR%\lib"