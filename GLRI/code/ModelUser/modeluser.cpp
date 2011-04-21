#include "modeluser.h"
#include "ui_modeluser.h"

ModelUser::ModelUser(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ModelUser)
{
    ui->setupUi(this);
}

ModelUser::~ModelUser()
{
    delete ui;
}
