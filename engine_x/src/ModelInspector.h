#pragma once

#include <QString>
#include <QStringList>

struct ModelReport
{
    bool loaded = false;
    QString summary;
    QStringList inputs;
    QStringList outputs;
};

class ModelInspector
{
public:
    static ModelReport inspect(const QString &modelPath);
};
