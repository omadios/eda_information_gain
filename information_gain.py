import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import matplotlib.gridspec as gridspec


def gini_calc(data,y):
    """Given input variable (data, pd series/np array) and target(y, pd series/np array), calculate the
    maximum information gain for a binary split. 
    Code is not written to be computationally efficient, but to show clearly how to calculate the Gini Inpurity.
    Brute force computation of all possible thresholds for continous data. Supports multiple targets 
    (i.e. three classes 0,1,2) but only binary splits.
    """
    assert len(y)==len(data),'Target vector and feature vector dimension mismatch'
    
    y=np.array(y)
    data=np.array(data)

    label_unique=np.sort(np.unique(y))
    data_unique=np.sort(np.unique(data))

    len_data=len(data)
    data_interval=(data_unique[:-1]+data_unique[1:])/2

    igs=np.zeros([len(data_interval),2])

    for num,interval in enumerate(data_interval):

        #GINI INDEX LEFT NODE
        ln_data=y[data<interval]
        ln_tot = len(ln_data) #†otal in left node

        ln = 1
        for label in label_unique:
            ln_x=np.count_nonzero(ln_data == label) 
            ln -= (ln_x/ln_tot)**2 

        #GINI INDEX RIGHT NODE
        rn_data=y[data>interval]
        rn_tot = len(rn_data) #†otal in left node

        rn = 1
        for label in label_unique:
            rn_x=np.count_nonzero(rn_data == label) 
            rn -= (rn_x/rn_tot)**2 


        #GINI INDEX BEFORE SPLIT
        gn = 1
        for label in label_unique:
            gn_x=np.count_nonzero(y == label) 
            gn -= (gn_x/len_data)**2

        tot=ln_tot+rn_tot
        wgn = ln * (ln_tot/tot) + rn * (rn_tot/tot) #weight right and left node by #observations
        
        #INFORMATION GAIN: substract from gini before split the weighted gini for split
        ig = gn - wgn  
        igs[num,0]=interval
        igs[num,1]=ig

    max_gain=igs[igs[:,1]==igs[:,1].max()]
    threshold=max_gain[0,0]
    ig=max_gain[0,1]
    #print('treshold >= %.3f,  information gain = %.10f' % (max_gain[0,0],max_gain[0,1]))
    return threshold,ig


def plot_gini_hist(data,y,threshold,ig,target_name='Target',feature_name='Feature',target_label={'0':'No', '1':'Yes'}):
    """Given input variable (data) and target(y), split threshold and information gain 
    plot histograms with data before and after  the split. Only supports binary targets (i.e. 0/1)
    """
    
    def lab_gen(a,b,pre_data):
        """Labelling helper function
        """
        count_lab0=('='+str(len(a))+'/'+str(len(pre_data)))
        count_lab1=('='+str(len(b))+'/'+str(len(pre_data)))
        label=[target_label.get('0')+count_lab0,target_label.get('1')+count_lab1]
        return label
    
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    
    label=[target_label.get('0'),target_label.get('1')]

    fig = plt.figure(figsize=(8,8))
    fig.suptitle('Treshold split <= %.3f,  information gain = %.3f' % (threshold,ig),fontsize=16)
    
    #-----
    ax1 = plt.subplot(gs[0, 1:3])
    hist, bin_edges = np.histogram(data, bins='rice')

    a = data[y==0]
    b = data[y==1]
    
    label=lab_gen(a,b,data)
    
    plt.hist([a, b ], bin_edges, label=label)
    plt.legend(loc='best',title=target_name)
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    
    scale_arrow_w=0.25*np.diff(bin_edges)[0]
    scale_arrow_l=0.10*np.max(hist)

    plt.arrow(threshold, scale_arrow_l, 0, -scale_arrow_l, length_includes_head=True, head_length=0.5*scale_arrow_l,
              width=scale_arrow_w, facecolor='black')
    



    limx=ax1.get_xlim()
    limy=ax1.get_ylim()
    
    #-----
    ax2 = plt.subplot(gs[1, :2])
    ln_data=data[data<threshold]
    ln_y=y[data<threshold]

    a = ln_data[ln_y==0]
    b = ln_data[ln_y==1]
    
    
    label=lab_gen(a,b,ln_data)
    
    plt.hist([a, b ], bin_edges, label=label)
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.legend(loc='best',title=target_name)
  
    
    ax2.set_xlim(limx),ax2.set_ylim(limy)
    
    #-----
    ax3 = plt.subplot(gs[1, 2:])

    rn_data=data[data>threshold]
    rn_y=y[data>threshold]

    a = rn_data[rn_y==0]
    b = rn_data[rn_y==1]
    
    label=lab_gen(a,b,rn_data)
    plt.hist([a, b ], bin_edges, label=label)
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.legend(loc='best',title=target_name)
    
    ax3.set_xlim(limx),ax3.set_ylim(limy)
    plt.show()


