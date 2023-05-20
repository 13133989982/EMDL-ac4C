import os

from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存，按需分配
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn import metrics

from keras.models import Model, load_model, Sequential
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.utils.np_utils import to_categorical

def perform_eval_1(predictions, Y_test, verbose=0):
    # class_label = np.uint8([round(x) for x in predictions[:, 0]]) # round()函数进行四舍五入
    # R_ = np.uint8(Y_test)
    # R = np.asarray(R_)
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))

    CM = metrics.confusion_matrix(R, class_label, labels=None)
    CM = np.double(CM)  # CM[0][0]：TN，CM[0][1]：FP，CM[1][0]：FN，CM[1][1]：TP

    # 计算各项指标
    sn = (CM[1][1]) / (CM[1][1] + CM[1][0])  # TP/(TP+FN)
    sp = (CM[0][0]) / (CM[0][0] + CM[0][1])  # TN/(TN+FP)
    acc = (CM[1][1] + CM[0][0]) / (CM[1][1] + CM[0][0] + CM[0][1] + CM[1][0])  # (TP+TN)/(TP+TN+FP+FN)
    pre = (CM[1][1]) / (CM[1][1] + CM[0][1])  # TP/(TP+FP)
    f1 = (2 * CM[1][1]) / (2 * CM[1][1] + CM[0][1] + CM[1][0])  # 2*TP/(2*TP+FP+FN)
    mcc = (CM[1][1] * CM[0][0] - CM[0][1] * CM[1][0]) / np.sqrt(
        (CM[1][1] + CM[0][1]) * (CM[1][1] + CM[1][0]) * (CM[0][0] + CM[0][1]) * (
                    CM[0][0] + CM[1][0]))  # (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^1/2
    gmean = np.sqrt(sn * sp)
    auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    # return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]
    return sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr


def performance_mean(performance):
    Sn = np.mean(performance[:, 0])
    Sp = np.mean(performance[:, 1])
    Acc = np.mean(performance[:, 2])
    Mcc = np.mean(performance[:, 3])
    Auroc = np.mean(performance[:, 4])
    Aupr = np.mean(performance[:, 5])
    print('Sn = %.4f ± %.2f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.2f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f± %.2f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    # print('Pre = %.2f%% ± %.2f%%' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    # print('F1 = %.2f%% ± %.2f%%' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    # print('Gmean = %.4f ± %.4f' % (np.mean(performance[:, 6]), np.std(performance[:, 6])))
    print('Auroc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('Aupr = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))
    return [Sn, Sp, Acc, Mcc, Auroc, Aupr]


def performance_mean1(performance):
    # >>> print("%.2f" % a)
    print('Sn = %.6f' % np.mean(performance[:, 0]))
    print('Sp = %.6f' % np.mean(performance[:, 1]))
    print('Acc = %.6f' % np.mean(performance[:, 2]))
    # print('Pre = %.2f%% ± %.2f%%' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    # print('F1 = %.2f%% ± %.2f%%' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('Mcc = %.6f' % np.mean(performance[:, 3]))
    # print('Gmean = %.4f ± %.4f' % (np.mean(performance[:, 6]), np.std(performance[:, 6])))
    print('Auroc = %.6f' % np.mean(performance[:, 4]))
    print('Aupr = %.6f' % np.mean(performance[:, 5]))


def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC


def train_model(i, train, label, test, test_y, random_num=0, flag=0):

    k_fold = StratifiedKFold(n_splits=10, random_state=random_num, shuffle=True)

    performance = np.zeros((10, 6))

    train_y_pred = np.zeros(len(train))

    test_y_all = np.zeros((len(test), 10))

    test_y_pred = np.zeros(len(test))

    for fold_, (train_index, validation_index) in enumerate(k_fold.split(train, label)):
        print("fold {} times".format(fold_ + 1))

        X_train, y_train = train[train_index], label[train_index]
        X_validation, y_validation = train[validation_index], label[validation_index]

        y_train = to_categorical(y_train, num_classes=2)
        y_validation = to_categorical(y_validation, num_classes=2)

        # BATCH_SIZE = 64
        # EPOCHS = 3
        # weights = {0: 1, 1: 1}
        # model = load_model('model1/model' + str(i + 1) + '/ac4C_DenseBlock_model_'+str(fold_+1)+'_fold.h5')
        #
        # history = model.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), epochs=EPOCHS,
        #                     batch_size=BATCH_SIZE, shuffle=True, class_weight=weights,
        #                     callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='auto')],
        #                     verbose=0)
        #
        # with open('model/model1/log_history_' + str(fold_ + 1) + '_fold.txt', 'w') as f:
        #     f.write(str(history.history))
        #
        # if flag == 0:
        #     model.save('model/model1/ac4C_train_model_{}_fold.h5'.format(fold_ + 1))  # HDF5文件，pip install h5py
        #     del model
        #     model = load_model('model/model1/ac4C_train_model_{}_fold.h5'.format(fold_ + 1))
        model = load_model('model/model1/ac4C_train_model_{}_fold.h5'.format(fold_ + 1))

        y_pred = model.predict(X_validation)
        test_pred = model.predict(test)

        test_y_all[:, fold_] = test_pred[:, 1]

        train_y_pred[validation_index] = y_pred[:, 1]

        Sn, Sp, Acc, MCC = show_performance(y_validation[:, 1], y_pred[:, 1])
        AUC = roc_auc_score(y_validation, y_pred)
        Aupr = average_precision_score(y_validation, y_pred)

        performance[fold_, :] = np.array((Sn, Sp, Acc, MCC, AUC, Aupr))
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, Aupr = %f' % (Sn, Sp, Acc, MCC, AUC, Aupr))

    test_y_pred[:] = test_y_all.mean(axis=1)

    Sn1, Sp1, Acc1, MCC1 = show_performance(test_y[:, 1], test_y_pred)
    AUC1 = roc_auc_score(test_y[:, 1], test_y_pred)
    Aupr1 = average_precision_score(test_y[:, 1], test_y_pred)

    # performance[fold_, :] = np.array((Sn1, Sp1, Acc, MCC, AUC, Aupr))
    print('test_ ：Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, Aupr = %f' % (Sn1, Sp1, Acc1, MCC1, AUC1, Aupr1))

    a = performance_mean(performance)

    return train_y_pred, test_y_pred, a


if __name__ == '__main__':

    WINDOWS = 415

    avg_train = np.zeros((10, 6))
    avg_test = np.zeros((10, 6))


    train_pred_pin = np.zeros((2296, 10))
    test_pred_pin = np.zeros((934, 10))


    for i in range(10):

        print("##################### dataset_"+str((i+1))+" #############################")

        f_r_train = open("ac4C_data_process/train/train"+str(i+1)+".txt", "r", encoding='utf-8')
        f_r_test = open("ac4C_data_process/test/test1.txt", "r", encoding='utf-8')

        train_data = f_r_train.readlines()
        test_data = f_r_test.readlines()

        f_r_train.close()
        f_r_test.close()

        from information_coding1 import one_hot1

        train, train_Y = one_hot1(train_data, windows=WINDOWS)
        train_Y = to_categorical(train_Y, num_classes=2)
        test, test_Y = one_hot1(test_data, windows=WINDOWS)
        test_Y = to_categorical(test_Y, num_classes=2)

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%% densenet %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        dense_y_pred, dense_test_y_pred1, pin = train_model(i, train, train_Y[:, 1], test, test_Y, 0, 0)

        # BATCH_SIZE = 64
        # EPOCHS = 3
        # weights = {0: 1, 1: 1}

        # dense_model = load_model('model1/model{}/ac4C_DenseBlock_test_model.h5'.format(i + 1))
        #
        # history = dense_model.fit(x=train, y=train_Y, validation_data=(test, test_Y), epochs=EPOCHS,
        #                           batch_size=BATCH_SIZE, shuffle=True, class_weight=weights,
        #                           callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
        #                           verbose=1)
        #
        # with open('model/model1/log_history_test_' + str(i + 1) + '_fold.txt', 'w') as f:
        #     f.write(str(history.history))
        # #
        # dense_model.save("model/model1/du_dense_model"+str(i+1)+".h5")
        # del dense_model
        dense_model = load_model("model/model1/du_dense_model"+str(i+1)+".h5")
        dense_test_y_pred = dense_model.predict(test, verbose=0)

        Sn1, Sp1, Acc1, MCC1 = show_performance(test_Y[:, 1], dense_test_y_pred[:, 1])
        AUC1 = roc_auc_score(test_Y[:, 1], dense_test_y_pred[:, 1])
        Aupr1 = average_precision_score(test_Y[:, 1], dense_test_y_pred[:, 1])
        print('test ： Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, Aupr = %f' % (
            Sn1, Sp1, Acc1, MCC1, AUC1, Aupr1))

        train_pred_pin[:, i] = dense_y_pred
        test_pred_pin[:, i] = dense_test_y_pred[:, 1]

        avg_train[i, :] = np.array(pin)

        avg_test[i, :] = np.array((Sn1, Sp1, Acc1, MCC1, AUC1, Aupr1))

    pd.DataFrame(train_pred_pin).to_csv('data/data1/train_pred_pin.csv', index=False)
    train_pred_p = np.mean(train_pred_pin, axis=1)

    print("train_pred_pin", train_pred_pin.shape)
    print("train_pred_p", train_pred_p.shape)

    print("train_Y", train_Y.shape)

    Sn, Sp, Acc, MCC = show_performance(train_Y[:, 1], train_pred_p)
    AUC = roc_auc_score(train_Y[:, 1], train_pred_p)
    Aupr = average_precision_score(train_Y[:, 1], train_pred_p)
    print('********* training sets soft polling ensemble results ：Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, Aupr = %f' % (Sn, Sp, Acc, MCC, AUC, Aupr))

    pd.DataFrame(test_pred_pin).to_csv('data/data1/test_pred_pin.csv', index=False)
    test_pred_p = np.mean(test_pred_pin, axis=1)

    print("test_pred_pin", test_pred_pin.shape)
    print("test_Y", test_Y.shape)
    Sn, Sp, Acc, MCC = show_performance(test_Y[:, 1], test_pred_p)
    AUC = roc_auc_score(test_Y[:, 1], test_pred_p)
    Aupr = average_precision_score(test_Y[:, 1], test_pred_p)
    print('********* Independent testing sets soft polling ensemble results ：Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, Aupr = %f' % (Sn, Sp, Acc, MCC, AUC, Aupr))

    """ ROC curve """
    from sklearn.metrics import roc_auc_score, roc_curve

    AUC1 = roc_auc_score(test_Y[:, 1], test_pred_p)

    fpr1, tpr1, thresholds1 = roc_curve(test_Y[:, 1], test_pred_p, pos_label=1)
    print('AUC=', AUC1)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, lw=2, alpha=.8, label='ROC curve (AUC = %.4f)' % AUC1)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('model/test1_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.show()





