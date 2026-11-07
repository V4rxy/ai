#include "MainWindow.h"
#include "CapturePanel.h"
#include "ModelInspector.h"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QPushButton>
#include <QStackedWidget>
#include <QVBoxLayout>

namespace {
QLabel *makeMetric(const QString &title, const QString &value)
{
    auto *label = new QLabel(QString("<b>%1</b><br><span style='font-size:22px'>%2</span>")
                             .arg(title, value));
    label->setStyleSheet("background:#111c2b;border:1px solid #26364d;border-radius:8px;padding:16px;color:#dce8f8;");
    return label;
}
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    setWindowTitle("Engine X | Titan Two + Computer Vision");
    resize(1280, 800);
    setStyleSheet(
        "QMainWindow,QWidget{background:#0b111b;color:#dce8f8;font-family:'Segoe UI';}"
        "QPushButton{background:#17263a;border:1px solid #304765;border-radius:6px;padding:9px 14px;}"
        "QPushButton:hover{background:#203957;}"
        "QListWidget{background:#101a28;border:1px solid #26364d;padding:6px;}"
    );

    auto *root = new QWidget(this);
    auto *layout = new QHBoxLayout(root);
    auto *nav = new QListWidget(root);
    nav->setFixedWidth(190);
    nav->addItems({"Overview", "Computer Vision", "ONNX Models", "Titan Two"});
    pages_ = new QStackedWidget(root);
    pages_->addWidget(makeDashboard());
    pages_->addWidget(new CapturePanel(root));
    pages_->addWidget(makeModelPage());
    pages_->addWidget(makeDevicePage());
    connect(nav, &QListWidget::currentRowChanged, pages_, &QStackedWidget::setCurrentIndex);
    nav->setCurrentRow(0);
    layout->addWidget(nav);
    layout->addWidget(pages_, 1);
    setCentralWidget(root);
}

QWidget *MainWindow::makeDashboard()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    auto *title = new QLabel("<span style='font-size:28px;font-weight:700'>ENGINE X</span><br>Device and Computer Vision workspace");
    layout->addWidget(title);

    auto *metrics = new QHBoxLayout;
    metrics->addWidget(makeMetric("TITAN TWO", "Not connected"));
    metrics->addWidget(makeMetric("CV INPUT", "Not started"));
    metrics->addWidget(makeMetric("ONNX MODEL", "Not loaded"));
    layout->addLayout(metrics);
    layout->addWidget(new QLabel("Choose a workspace from the left navigation."));
    layout->addStretch();
    return page;
}

QWidget *MainWindow::makeDevicePage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    layout->addWidget(new QLabel("<span style='font-size:24px'>Titan Two</span>"));
    deviceStatus_ = new QLabel("Device communication is not configured yet.");
    layout->addWidget(deviceStatus_);
    layout->addWidget(new QLabel("This panel will use documented Gtuner/Titan Two interfaces. No unsupported USB protocol is assumed."));
    layout->addStretch();
    return page;
}

QWidget *MainWindow::makeModelPage()
{
    auto *page = new QWidget;
    auto *layout = new QVBoxLayout(page);
    auto *choose = new QPushButton("Load ONNX model");
    auto *report = new QLabel("No model loaded.");
    report->setWordWrap(true);
    layout->addWidget(new QLabel("<span style='font-size:24px'>ONNX Model Inspector</span>"));
    layout->addWidget(choose);
    layout->addWidget(report);
    layout->addStretch();

    connect(choose, &QPushButton::clicked, this, [this, report]() {
        const QString path = QFileDialog::getOpenFileName(this, "Select ONNX model", {}, "ONNX models (*.onnx)");
        if (path.isEmpty())
            return;
        const ModelReport result = ModelInspector::inspect(path);
        QString details = result.summary;
        if (!result.inputs.isEmpty())
            details += "\nInputs: " + result.inputs.join(", ");
        if (!result.outputs.isEmpty())
            details += "\nOutputs: " + result.outputs.join(", ");
        report->setText(details);
    });
    return page;
}

void MainWindow::chooseModel()
{
    // Reserved for the model workflow.
}
