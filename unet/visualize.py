#%%
import matplotlib.pyplot as plt
import os
from PIL import Image

root_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(root_dir,'result')

os.chdir(result_dir)
file_list = os.listdir(result_dir)
input_list = sorted([f for f in file_list if f.startswith('input')])
label_list = sorted([f for f in file_list if f.startswith('label')])
output_list = sorted([f for f in file_list if f.startswith('output')])

for i in range(len(input_list)):
    fig, ax = plt.subplots(1,3,figsize=(9,6))
    input_ = Image.open(input_list[i])
    label_ = Image.open(label_list[i])
    output_ = Image.open(output_list[i])

    ax[0].imshow(input_,cmap="gray")
    ax[1].imshow(label_,cmap="gray")
    ax[2].imshow(output_,cmap="gray")
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].set_title('Input')
    ax[1].set_title('Label')
    ax[2].set_title('Output')
    fig.suptitle("Result")
    fig.tight_layout()
    plt.show()

    input_.close()
    label_.close()
    output_.close()
