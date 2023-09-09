import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np

plt.rcParams['font.family'] = 'Calibri'

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

prefix="results/frequencies/"

dataset_list=["FB15k-237", "wn18rr", "YAGO3-10"]
# dataset_list=["FB15k-237"]
# dataset_list=["wn18rr"]
# dataset_list=["YAGO3-10"]

model_list=["RotatE"]

subsampling_list=["none"]

# num_filter = {"FB15k-237": 2000, "wn18rr": 2000, "YAGO3-10": 4000}
num_filter = {"FB15k-237": 2000, "wn18rr": 1778, "YAGO3-10": 4444}

fig, axes = plt.subplots(3, 1, figsize=(48,48))
handles, labels = None, None
for col_id, dataset in enumerate(dataset_list):
    row_id = 0
    for model in model_list:
        for subsampling in subsampling_list:
            dir_path = prefix + "_".join([dataset,model,subsampling,"dump/"])
            with open(dir_path + "file_cbs_dump.tsv") as f_in:
                F_cbs = {}
                F_e = {}
                F_r = {}
                for line in f_in:
                    e, r, freq = [float(col) for col in line.strip().split("\t")]
                    freq -= 3 # Remove the additive term 
                    F_cbs[(e, r)] = freq
                    if e not in F_e:
                        F_e[e] = 0
                    if r not in F_r:
                        F_r[r] = 0
                    F_e[e] += freq
                    F_r[r] += freq
            filtered_entities = []
            filtered_relations = []
            for q in F_cbs.keys():
                if F_cbs[q] == 1:
                    e = q[0]
                    r = q[1]
                    filtered_entities.append(F_e[e])
                    filtered_relations.append(F_r[r])

            filtered_entities = sorted(filtered_entities, reverse=True)
            filtered_relations = sorted(filtered_relations, reverse=True)

            e_x = []
            e_y = []
            for i,v in enumerate(filtered_entities):
                if i % num_filter[dataset] == 0:
                    e_x.append(i*0.001)
                    e_y.append(v)
            r_x = []
            r_y = []
            for i,v in enumerate(filtered_relations):
                if i % num_filter[dataset] == 0:
                    if dataset=="YAGO3-10":
                        r_x.append(i*0.001+2.5)
                    else:
                        r_x.append(i*0.001+1.0)
                    r_y.append(v)

            # Plot
            if isinstance(axes, np.ndarray):
                ax = axes[col_id]
            else:
                ax = axes
            # ax = axes[col_id]
            col_id += 1
            #start_id = -len(r_x)
            start_id = 0
            print(len(e_x))
            print(len(r_x))

            # width = 1.0 * num_filter[dataset] / 4444
            if dataset=="YAGO3-10":
                width = 2.5
            else:
                width = 1.0
            ax.bar(e_x[start_id:], e_y[start_id:], width=width, color='red', label="Entity")
            ax.bar(r_x, r_y, width=width, color='blue', label="Relation")

            if dataset == "wn18rr":
                #ax.set_ylim(0.0, 4.0 * 1e-5)
                # ax.set_title(dataset.upper(), fontsize=100, fontweight='bold') # chart title on the above
                ax.set_ylabel(dataset.upper(), fontsize=100, fontweight='bold') # chart title on the left
            else:
                #ax.set_ylim(0.0, 4.5 * 1e-5)
                # ax.set_title(dataset, fontsize=100, fontweight='bold')
                ax.set_ylabel(dataset, fontsize=100, fontweight='bold')
            ax.yaxis.set_tick_params(labelsize=50)
            ax.set_yscale("log")
            #tick_spacing=1e-5
            #ax.yaxis.set_major_locator(mtick.MultipleLocator(tick_spacing))
            ax.set_xticks([])
            # ax.set_ylabel("%      ", rotation=0, fontsize=28)
            # ax.legend(fontsize=50)
            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()
                # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, fontsize=50)
                # fig.legend(['Entity', 'Relation'], fontsize=50, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0))
                fig.legend(['Entity', 'Relation'], fontsize=100, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout()
# plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(bottom=0.06)
fig.savefig("./graphs/freq_of_entities_and_relations.pdf")
