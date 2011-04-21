#include "modelbuilder.h"
#include "ui_modelbuilder.h"

ModelBuilder::ModelBuilder(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ModelBuilder)
{
    ui->setupUi(this);
}

ModelBuilder::~ModelBuilder()
{
    delete ui;
}
