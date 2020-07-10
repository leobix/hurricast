import torch 
import torch.nn as nn 
import sklearn.metrics 

#=============
# Metrics, loss fn

def create_loss_fn(task='classification'):
    """
    Wrappers that uses same signature for all loss_functions.
    Can easily add new losses function here.
    #TODO: Théo--> See if we can make an entropy based loss.
    """
    assert task in ['classification',
                    'regression'], "The\
    prediction function needs either the classification or\
    regression flag."

    base_loss_fn = nn.MSELoss()
    base_classification_loss_fn = nn.CrossEntropyLoss()
    if task == 'classification':
        return base_classification_loss_fn
    else:
        return base_loss_fn


def create_eval_loss_fn(task='classification'):
    assert task in ['classification',
                    'regression']
    if task == 'classification':
        return lambda x, y: 1 - sklearn.metrics.accuracy_score(x, y)
    else:
        return nn.MSELoss()


def get_metrics(model_outputs, target, task):
    """
    Compute all relevant metrics.

    Parameters:
    ----------
    model_outputs: pre-softmax prediction/regression outputs
    target: correct class

    Out:
    ---------
    out: dict 

    """
    def get_pred(x_out):
        if len(x_out.size()) > 1:
            return x_out.argmax(-1)
        else:
            return x_out
    #Init with NA
    metrics = {}
    metrics['mse'] = "NA"
    metrics['n_tokens'] = "NA"
    metrics['confusion_matrix'] = "NA"
    metrics['accuracy'] = "NA"
    metrics['precision'] = "NA"
    metrics['recall'] = "NA"
    metrics['f1'] = 'NA'
    metrics['f1_micro'] = "NA"
    metrics['f1_macro'] = "NA"
    metrics['classification_report'] = "NA"
    
    if len(model_outputs) == 0:
        return metrics
    
        
    if task == 'classification':
        class_pred = get_pred(model_outputs)

        print('Sanity check')
        print(type(model_outputs))
        print(type(class_pred))

        try:
            print(model_outputs.size())
        
        except Exception as e:
            print('PB1', e)

        try:
            print(class_pred.size())

        except Exception as e:
            print('PB2', e)

        metrics['f1_micro'] = sklearn.metrics.f1_score(
            y_true=target, y_pred=class_pred, average='micro')
        metrics['f1_macro'] = sklearn.metrics.f1_score(
            y_true=target, y_pred=class_pred, average='macro')
        metrics['accuracy'] = sklearn.metrics.accuracy_score(
            y_true=target, y_pred=class_pred)
        metrics['precision'] = sklearn.metrics.precision_score(
            y_true=target, y_pred=class_pred, average="weighted")
        metrics['recall'] = sklearn.metrics.recall_score(
            y_true=target, y_pred=class_pred, average="weighted")
        metrics['f1'] = sklearn.metrics.f1_score(
            y_true=target, y_pred=class_pred, average="weighted")
        metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(
            y_true=target, y_pred=class_pred)
        metrics['n_tokens'] = len(target)
        metrics['classification_report'] = sklearn.metrics.classification_report(
            y_true=target, y_pred=class_pred)
        #metrics['mse'] = 'NA'

    else:
        metrics['mse'] = sklearn.metrics.mean_squared_error(
            y_true=target, y_pred=model_outputs)
        metrics['n_tokens'] = len(target)
        #metrics['confusion_matrix'] = "NA"
        #metrics['accuracy'] = "NA"
        #metrics['precision'] = "NA"
        #metrics['recall'] = "NA"
        #metrics['f1'] = 'NA'
        #metrics['f1_micro'] = "NA"
        #metrics['f1_macro'] = "NA"
    
    return metrics

#===========
#Wrapper
def create_metrics_fn(task):
    train_loss_fn = create_loss_fn(task=task)
    eval_loss_fn = create_eval_loss_fn(task=task)
    metrics_fn = lambda preds, target: get_metrics(
        preds, target, task)
    return train_loss_fn, eval_loss_fn, metrics_fn




