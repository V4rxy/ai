#include "MainWindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("Engine X");
    QApplication::setOrganizationName("Apex Vision");

    MainWindow window;
    window.show();
    return app.exec();
}
