#include "ModelInspector.h"

#ifdef ENGINEX_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

ModelReport ModelInspector::inspect(const QString &modelPath)
{
    ModelReport report;
#ifdef ENGINEX_HAS_ONNXRUNTIME
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EngineX");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef _WIN32
        const std::wstring path = modelPath.toStdWString();
        Ort::Session session(env, path.c_str(), options);
#else
        const std::string path = modelPath.toStdString();
        Ort::Session session(env, path.c_str(), options);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t inputCount = session.GetInputCount();
        const size_t outputCount = session.GetOutputCount();

        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            report.inputs.append(QString::fromUtf8(name.get()));
        }
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            report.outputs.append(QString::fromUtf8(name.get()));
        }

        report.loaded = true;
        report.summary = QString("Model loaded: %1 input(s), %2 output(s). Tensor layout still needs validation.")
            .arg(inputCount).arg(outputCount);
    } catch (const std::exception &e) {
        report.summary = QString("ONNX Runtime could not load this model: %1").arg(e.what());
    }
#else
    Q_UNUSED(modelPath);
    report.summary = "ONNX Runtime is not installed or was not found by CMake.";
#endif
    return report;
}
