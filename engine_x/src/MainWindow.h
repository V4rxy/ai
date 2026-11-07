#pragma once

#include <QMainWindow>

class QLabel;
class QStackedWidget;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private:
    QStackedWidget *pages_ = nullptr;
    QLabel *deviceStatus_ = nullptr;

    QWidget *makeDashboard();
    QWidget *makeDevicePage();
    QWidget *makeModelPage();
    void chooseModel();
};
