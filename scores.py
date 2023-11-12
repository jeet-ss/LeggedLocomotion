import numpy as np
import argparse
import scipy.io

from scipy.stats import pearsonr


def scores_gen(args):
    

    # file
    file_name = args.fp
    #
    ff = np.load(file_name, allow_pickle=True).item()
    print(ff.keys())
    if 'val_loss' in ff.keys():
        print(np.argmin(ff['val_loss']),np.min(ff['val_loss']))
    # Analyse Test Data
    preds = ff['test_pred'].squeeze()
    labels = ff['test_label']
    # positive peaks
    pp = 20
    #pred_peaks = scipy.signal.find_peaks(preds, prominence=pp, distance=50)
    label_peaks = scipy.signal.find_peaks(labels, prominence=pp, distance=50)
    # negative peaks
    neg_pp = 1 #2
    neg_label_peaks = scipy.signal.find_peaks(-labels, height=neg_pp, distance=10 )
    #neg_pred_peaks = scipy.signal.find_peaks(-preds, height=neg_pp, )
    #
    print( label_peaks[0].shape, neg_label_peaks[0].shape)

    pos_score = pearsonr(preds[label_peaks[0]],labels[label_peaks[0]])
    neg_Score = pearsonr(preds[neg_label_peaks[0]],labels[neg_label_peaks[0]])

    print("Positive Score: ", pos_score[0], "\nNegative Soce: ", neg_Score[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of legged Locomotion')
    parser.add_argument('--fp', '-filepath', type=str, default="results_3i/lstm2_0_3000_data.npy", help='file path of data')
    args = parser.parse_args()

    scores_gen(args)
