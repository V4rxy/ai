#pragma once

#include <QImage>
#include <QWidget>
#include <opencv2/videoio.hpp>

class QLabel;
class QComboBox;
class QPushButton;
class QTimer;

class CapturePanel final : public QWidget
{
    Q_OBJECT

public:
    explicit CapturePanel(QWidget *parent = nullptr);
    ~CapturePanel() override;

private slots:
    void refreshDevices();
    void startCapture();
    void stopCapture();
    void updateFrame();

private:
    QLabel *preview_ = nullptr;
    QLabel *status_ = nullptr;
    QComboBox *devices_ = nullptr;
    QPushButton *startButton_ = nullptr;
    QPushButton *stopButton_ = nullptr;
    QTimer *timer_ = nullptr;
    cv::VideoCapture capture_;

    static QImage toQImage(const cv::Mat &frame);
};
