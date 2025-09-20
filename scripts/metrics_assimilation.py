import numpy as np

from stardist.src.utils.hydranet import create_assimilated_dict

def main():
    # For 5 trials, can be externded to incorporate more
    l1 = []
    l2 = []
    l3 = []
    # l4 = []
    # l5 = []
    metric = ['mean_true_score', 'accuracy']
    #'accuracy' is the AP
    #'mean_true_score' is the IoU_R.

    for m in metric:
        asd = create_assimilated_dict(l1,l2,l3,m)
        max_key = max(asd, key=lambda k: asd[k][0])
        max_val1, max_val2 = asd[max_key][0], asd[max_key][1]
        print(m, max_key, max_val1, max_val2)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter