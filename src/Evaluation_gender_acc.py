import matplotlib.pyplot as plt
import numpy as np


def test_gender_acc(auto_res, novel_name, manual_res):
    '''
    auto_res: Dictionary of auto gender identification results {'Gender': {'Novel name': [names] }}
    novel_name: String of novel name
    manual_res: Dictionnary of manual gender identification results {'Novel name': {'Gender': [names]}}
    '''

    pred_label = ["Female","Male","Neither","Both"]
    true_label = ["Female","Male","Neither","Both"]
    cm = np.zeros((len(pred_label), len(true_label)),dtype = int)
    for name in auto_res['Female'][novel_name]:
        if name in manual_res[novel_name]['Female']:
            cm[0,0] += 1
        elif name in manual_res[novel_name]['Male']:
            cm[0,1] += 1 
        elif name in manual_res[novel_name]['Non_human']:
            cm[0,2] += 1
        elif name in manual_res[novel_name]['Mixed_Surnames']:
            cm[0,3] += 1
    for name in auto_res['Male'][novel_name]:
        if name in manual_res[novel_name]['Female']:
            cm[1,0] += 1
        elif name in manual_res[novel_name]['Male']:
            cm[1,1] += 1 
        elif name in manual_res[novel_name]['Non_human']:
            cm[1,2] += 1
        elif name in manual_res[novel_name]['Mixed_Surnames']:
            cm[1,3] += 1
    for name in auto_res['Non_human'][novel_name]:
        if name in manual_res[novel_name]['Female']:
            cm[2,0] += 1
        elif name in manual_res[novel_name]['Male']:
            cm[2,1] += 1 
        elif name in manual_res[novel_name]['Non_human']:
            cm[2,2] += 1
        elif name in manual_res[novel_name]['Mixed_Surnames']:
            cm[2,3] += 1
    for name in auto_res['Surname'][novel_name]:
        if name in manual_res[novel_name]['Female']:
            cm[3,0] += 1
        elif name in manual_res[novel_name]['Male']:
            cm[3,1] += 1 
        elif name in manual_res[novel_name]['Non_human']:
            cm[3,2] += 1
        elif name in manual_res[novel_name]['Mixed_Surnames']:
            cm[3,3] += 1
    acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/np.sum(cm)
    print(f'''{novel_name}:\n
    Female_precision: {cm[0,0]/np.sum(cm[0])}\n
    Male_precision: {cm[1,1]/np.sum(cm[1])}\n
    Overall acc: {acc}
          ''')
    confusion_plot(cm)
    print('-'*80)
    print('-'*80)
    print('\n')
    return cm, acc

def confusion_plot(cm):
  label1 = ["Female","Male","Neither","Both"]
  label2 = ["Female","Male","Neither","Both"]
  # Create confusion matrix plot
  fig, ax = plt.subplots()
  im = ax.imshow(cm, cmap='viridis')

  # Set tick labels and axis labels
  ax.set_xticks(np.arange(len(label2)))
  ax.set_yticks(np.arange(len(label1)))
  ax.set_xticklabels(label2)
  ax.set_yticklabels(label1)

  # Loop over data dimensions and create text annotations
  for i in range(len(label1)):
      for j in range(len(label2)):
          text = ax.text(j, i, cm[i, j],
                        ha="center", va="center", color="w")

  # Create colorbar
  cbar = ax.figure.colorbar(im, ax=ax)

  # Set plot title and display the plot
  plt.title("Confusion Matrix")
  plt.show()


# F char manual: {len_manual_f}, # F char auto: {len_auto_f}, # mis-labeled: {f_wrong}, precision: {(len_auto_f-f_wrong)/len_auto_f} \n
# M char manual: {len_manual_m}, # M char auto: {len_auto_m}, # mis-labeled: {m_wrong}, precision: {(len_auto_m-m_wrong)/len_auto_m} \n