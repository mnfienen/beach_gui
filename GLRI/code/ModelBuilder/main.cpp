#include <QtGui/QApplication>
#include "modelbuilder.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ModelBuilder w;
    w.show();

    return a.exec();
}
