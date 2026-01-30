import os
import matplotlib.pyplot as plt
from PIL import Image

# paths provided
edited_root_encoder = "/gpfs/home6/scur0103/edits-celeba-hq/encoder/subsampled_0.02step0.003steps50png/seed42/"
edited_root_photoguard = "/gpfs/home6/scur0103/edits-celeba-hq/photoguard/subsampled_0.02step0.003steps50png/seed42/"
edited_root_facelock = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/subsampled_0.02step0.003steps50png/seed42/"
edited_root_clean = "/gpfs/home6/scur0103/clean_edit2/subsampled_set/seed42/"
#edited_root_09 = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/9images_0.09step0.02steps100/seed42/"
"""
edited_root0 = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/9images0.06step0.01steps100/seed0/"
edited_root1 = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/9images0.06step0.01steps100/seed1/"
edited_root1024 = "/gpfs/home6/scur0103/edits-celeba-hq/facelock/9images0.06step0.01steps100/seed1024/"
"""

src_root = "/gpfs/home6/scur0103/celeba-hq/subsampled_set/"
#defended_root = "/gpfs/home6/scur0103/defended-celeba-hq/encoder/subsampled_0.10step0.02/budget_0.1/"


# mapping: (image_number, prompt_number)
pairs = [
    ("062799", 2),
    ("006750", 3),
    ("007536", 0),
    ("081680", 10),
    ("009644", 13),
    ("001036", 12),
    ("011620", 18),
    ("002420", 23),
    ("017054", 19),
]

num_rows = 5
num_cols = 9

fig, axs = plt.subplots(5, len(pairs), figsize=(2 * len(pairs), 2 * num_rows))

for i, (img_num, prompt_num) in enumerate(pairs):
    #defended_path = os.path.join(defended_root, f"{img_num}.jpg")
    src_path = os.path.join(src_root, f"{img_num}.jpg")
    edited_path_clean = os.path.join(edited_root_clean, f"prompt{prompt_num}", f"{img_num}.jpg")
    

    edited_path_encoder = os.path.join(edited_root_encoder, f"prompt{prompt_num}", f"{img_num}.png")
    edited_path_facelock = os.path.join(edited_root_facelock, f"prompt{prompt_num}", f"{img_num}.png")
    edited_path_photoguard = os.path.join(edited_root_photoguard, f"prompt{prompt_num}", f"{img_num}.png")
    #edited_path_09 = os.path.join(edited_root_09, f"prompt{prompt_num}", f"{img_num}.jpg")


    # load defended (row 1)
    """
    defended_img = Image.open(defended_path)
    axs[0, i].imshow(defended_img)
    axs[0, i].set_title(f"Defended\n{img_num}")
    axs[0, i].axis('off')
    """
    
    src_img = Image.open(src_path)
    axs[0, i].imshow(src_img)
    if i == 0:
      axs[0, i].set_ylabel('Source\nImage',
                         rotation=90,
                         fontsize=20,
                         labelpad=40,
                         va='center',
                         ha='center')
      axs[0, i].set_xticks([])
      axs[0, i].set_yticks([])
      axs[0, i].spines['top'].set_visible(False)
      axs[0, i].spines['right'].set_visible(False)
      axs[0, i].spines['bottom'].set_visible(False)
      axs[0, i].spines['left'].set_visible(False)
    else:
      axs[0, i].axis('off')
      
    
    

    # load edited (row 2)
    edited_img_clean = Image.open(edited_path_clean)
    axs[1, i].imshow(edited_img_clean)
    if i == 0:
      axs[1, i].set_ylabel('No\nDefense',
                         rotation=90,
                         fontsize=20,
                         labelpad=40,
                         va='center',
                         ha='center')
      axs[1, i].set_xticks([])
      axs[1, i].set_yticks([])
      axs[1, i].spines['top'].set_visible(False)
      axs[1, i].spines['right'].set_visible(False)
      axs[1, i].spines['bottom'].set_visible(False)
      axs[1, i].spines['left'].set_visible(False)
    else:
      axs[1, i].axis('off')
    
    # load edited (row 2)
    edited_img_photoguard = Image.open(edited_path_photoguard)
    axs[2, i].imshow(edited_img_photoguard)
    if i == 0:
      axs[2, i].set_ylabel('PhotoGuard',
                         rotation=90,
                         fontsize=20,
                         labelpad=40,
                         va='center',
                         ha='center')
      axs[2, i].set_xticks([])
      axs[2, i].set_yticks([])
      axs[2, i].spines['top'].set_visible(False)
      axs[2, i].spines['right'].set_visible(False)
      axs[2, i].spines['bottom'].set_visible(False)
      axs[2, i].spines['left'].set_visible(False)
    else:
      axs[2, i].axis('off')
      
      
    
    # load edited (row 2)
    edited_img_encoder = Image.open(edited_path_encoder)
    axs[3, i].imshow(edited_img_encoder)
    if i == 0:
      axs[3, i].set_ylabel('Encoder',
                         rotation=90,
                         fontsize=20,
                         labelpad=40,
                         va='center',
                         ha='center')
      axs[3, i].set_xticks([])
      axs[3, i].set_yticks([])
      axs[3, i].spines['top'].set_visible(False)
      axs[3, i].spines['right'].set_visible(False)
      axs[3, i].spines['bottom'].set_visible(False)
      axs[3, i].spines['left'].set_visible(False)
    else:
      axs[3, i].axis('off')
      
      
    
    # load edited (row 2)
    edited_img_facelock = Image.open(edited_path_facelock)
    axs[4, i].imshow(edited_img_facelock)
    if i == 0:
      axs[4, i].set_ylabel('FaceLock',
                         rotation=90,
                         fontsize=20,
                         labelpad=40,
                         va='center',
                         ha='center')
      axs[4, i].set_xticks([])
      axs[4, i].set_yticks([])
      axs[4, i].spines['top'].set_visible(False)
      axs[4, i].spines['right'].set_visible(False)
      axs[4, i].spines['bottom'].set_visible(False)
      axs[4, i].spines['left'].set_visible(False)
    else:
      axs[4, i].axis('off')
      
    
    # Column label (a), (b), (c), ...
    for i in range(num_cols):
      label = f"({chr(97+i)})"  # 97 = 'a'
      axs[4, i].set_xlabel(label, fontsize=20, labelpad=10)
      axs[4, i].axis('on')
      axs[4, i].set_xticks([])
      axs[4, i].set_yticks([])
      axs[4, i].spines['top'].set_visible(False)
      axs[4, i].spines['right'].set_visible(False)
      axs[4, i].spines['bottom'].set_visible(False)
      axs[4, i].spines['left'].set_visible(False)


      


plt.tight_layout()
plt.savefig("budget0.02png_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
