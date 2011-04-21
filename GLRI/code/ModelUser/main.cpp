#include <QtGui/QApplication>
#include "modeluser.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ModelUser w;
    w.show();

    return a.exec();
}
