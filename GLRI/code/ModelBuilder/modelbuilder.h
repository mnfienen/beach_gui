#ifndef MODELBUILDER_H
#define MODELBUILDER_H

#include <QDialog>

namespace Ui {
    class ModelBuilder;
}

class ModelGUI : public QDialog
{
    Q_OBJECT

public:
    explicit ModelBuilder(QWidget *parent = 0);
    ~ModelBuilder();

private:
    Ui::ModelBuilder *ui;
};

#endif // MODELGUI_H
