import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

plt.rcParams['font.family'] = 'Calibri'

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

prefix="results/frequencies/"
# dataset_list=["FB15k-237", "wn18rr"]
# dataset_list=["FB15k-237"]
# dataset_list=["wn18rr"]
dataset_list=["YAGO3-10"]

# model_list=["RotatE", "TransE", "ComplEx", "DistMult"]
# model_list=["RotatE", "TransE"]
# model_list=["ComplEx", "DistMult"]
model_list=["RotatE", "HAKE"]
# model_list=["HAKE"]

subsampling_list=["none", "default"]

fig, axes = plt.subplots(1, 4, figsize=(48,7))
col_id = 0
for row_id, dataset in enumerate(dataset_list):
    for model in model_list:
        for subsampling in subsampling_list:
            dir_path = prefix + "_".join([dataset,model,subsampling,"dump/"])
            with open(dir_path + "file_cbs_dump.tsv") as f_in:
                Z_cbs = 0
                F_cbs = {}
                for line in f_in:
                    e, r, freq = [float(col) for col in line.strip().split("\t")]
                    F_cbs[(e, r)] = freq
                    Z_cbs += freq
            with open(dir_path + "file_mbs_dump.tsv") as f_in:
                Z_mbs = 0
                F_mbs = {}
                for line in f_in:
                    e, r, freq = [float(col) for col in line.strip().split("\t")]
                    F_mbs[(e, r)] = freq
                    Z_mbs += freq
            X_cbs = []
            X_mbs = []
            L_cbs=[]
            L_mbs=[]
            for i,(k,v) in enumerate(sorted(F_cbs.items(), key=lambda item: item[1], reverse=True)):
                X_cbs.append(F_cbs[k] / Z_cbs)
                X_mbs.append(F_mbs[k] / Z_mbs)
                L_cbs.append(0.25*i)
                L_mbs.append(0.25*i)
            # ax = axes[row_id][col_id]
            ax = axes[col_id]
            if subsampling == "default":
                sub_name = "Base"
            else:
                sub_name = "None"
            start_id = -100
            ax.bar(L_cbs[start_id:], X_cbs[start_id:], width=0.25, color='red', label="Count")
            ax.bar(L_mbs[start_id:], X_mbs[start_id:], width=0.25, color='blue', label="Model")
            if dataset == "wn18rr":
                ax.set_ylim(0.0, 4.0 * 1e-5)
                ax.set_title(", ".join([dataset.upper(), model, sub_name]), fontsize=50, fontweight='bold')
            else:
                ax.set_ylim(0.0, 4.5 * 1e-5)
                ax.set_title(", ".join([dataset, model, sub_name]), fontsize=50, fontweight='bold')
            ax.yaxis.set_tick_params(labelsize=24)
            tick_spacing=1e-5
            ax.yaxis.set_major_locator(mtick.MultipleLocator(tick_spacing))
            ax.set_xticks([])
            ax.set_ylabel("%      ", rotation=0, fontsize=28)
            # ax.legend(fontsize=28)
            # fig.legend(['Count', 'Model'], fontsize=40, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))
            col_id += 1
            print(col_id)
fig.tight_layout()
# plt.subplots_adjust(bottom=0.17)
fig.savefig("./graphs/normed_freq_of_queries_5.pdf")
