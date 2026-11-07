#include "CapturePanel.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <opencv2/imgproc.hpp>

CapturePanel::CapturePanel(QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    auto *toolbar = new QHBoxLayout;

    devices_ = new QComboBox(this);
    auto *refresh = new QPushButton("Refresh inputs", this);
    startButton_ = new QPushButton("Start preview", this);
    stopButton_ = new QPushButton("Stop", this);
    stopButton_->setEnabled(false);

    toolbar->addWidget(devices_, 1);
    toolbar->addWidget(refresh);
    toolbar->addWidget(startButton_);
    toolbar->addWidget(stopButton_);
    layout->addLayout(toolbar);

    preview_ = new QLabel("Select a capture input and start preview", this);
    preview_->setAlignment(Qt::AlignCenter);
    preview_->setMinimumSize(640, 360);
    preview_->setStyleSheet("background:#080d16;color:#8291a8;border:1px solid #26364d;");
    layout->addWidget(preview_, 1);

    status_ = new QLabel("Capture idle", this);
    layout->addWidget(status_);

    timer_ = new QTimer(this);
    timer_->setInterval(16);
    connect(refresh, &QPushButton::clicked, this, &CapturePanel::refreshDevices);
    connect(startButton_, &QPushButton::clicked, this, &CapturePanel::startCapture);
    connect(stopButton_, &QPushButton::clicked, this, &CapturePanel::stopCapture);
    connect(timer_, &QTimer::timeout, this, &CapturePanel::updateFrame);
    refreshDevices();
}

CapturePanel::~CapturePanel()
{
    stopCapture();
}

void CapturePanel::refreshDevices()
{
    devices_->clear();
    for (int index = 0; index < 10; ++index) {
        cv::VideoCapture probe(index, cv::CAP_DSHOW);
        if (probe.isOpened()) {
            devices_->addItem(QString("Video input %1").arg(index), index);
            probe.release();
        }
    }
    status_->setText(devices_->count()
        ? QString("%1 video input(s) found").arg(devices_->count())
        : "No video inputs found. Check capture-card connection and drivers.");
}

void CapturePanel::startCapture()
{
    if (devices_->currentIndex() < 0) {
        status_->setText("No input selected.");
        return;
    }

    const int deviceIndex = devices_->currentData().toInt();
    if (!capture_.open(deviceIndex, cv::CAP_DSHOW)) {
        status_->setText("Could not open this input. Try another device or check its driver.");
        return;
    }

    timer_->start();
    startButton_->setEnabled(false);
    stopButton_->setEnabled(true);
    status_->setText(QString("Previewing input %1").arg(deviceIndex));
}

void CapturePanel::stopCapture()
{
    timer_->stop();
    if (capture_.isOpened())
        capture_.release();
    startButton_->setEnabled(true);
    stopButton_->setEnabled(false);
    status_->setText("Capture idle");
}

void CapturePanel::updateFrame()
{
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        status_->setText("Capture stream stopped or returned an empty frame.");
        stopCapture();
        return;
    }
    preview_->setPixmap(QPixmap::fromImage(toQImage(frame)).scaled(
        preview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage CapturePanel::toQImage(const cv::Mat &frame)
{
    cv::Mat rgb;
    if (frame.channels() == 1)
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888).copy();
}
