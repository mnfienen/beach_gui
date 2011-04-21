#ifndef MODELUSER_H
#define MODELUSER_H

#include <QDialog>

namespace Ui {
    class ModelUser;
}

class ModelUser : public QDialog
{
    Q_OBJECT

public:
    explicit ModelUser(QWidget *parent = 0);
    ~ModelUser();

private:
    Ui::ModelUser *ui;
};

#endif // MODELUSER_H
