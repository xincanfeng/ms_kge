import os
import openpyxl

# path of a file
# old_name = 'models/' + 'ComplEx_' + 'FB15k_' + 'old_id'
# new_name = 'models/' + 'ComplEx_' + 'FB15k_' + 'new_id'


# Rename a file after checking whether it exists
counts_old_file_not_exist = 0
counts_new_file_already_exist = 0
counts_success = 0
def rename_a_file(old_name, new_name):
    global counts_old_file_not_exist, counts_new_file_already_exist, counts_success
    if not os.path.exists(old_name):
        print("{} does not exist.".format(old_name))
        counts_old_file_not_exist = counts_old_file_not_exist + 1
    elif os.path.exists(new_name):
        print("{} already exists.".format(new_name))
        counts_new_file_already_exist = counts_new_file_already_exist + 1
    else:
        # Rename the file
        os.rename(old_name, new_name)
        print("{} has been renamed to {}".format(old_name, new_name))
        counts_success = counts_success + 1


def rename_files_in_list(old_list, new_list):
    for i in range(len(old_list)):
        rename_a_file(old_list[i], new_list[i])


datasets = ['FB15k-237', 'wn18rr', 'FB15k', 'wn18', 'YAGO3-10']
models = ['HAKE', 'ComplEx', 'RotatE', 'pRotatE', 'TransE', 'DistMult']
# copy the row of models manually
model_rows = [3, 52, 119, 186, 193, 200, 207]


# prepare for the information where you want to change the filenames
wb = openpyxl.load_workbook(filename='results/temp_rename.xlsx')
ws = wb.active
for e in range(len(models)):
    for x in range(len(datasets)):
        old_list = []
        middle_list = []
        new_list = []
        for c in range(model_rows[e+1]-model_rows[e]):
            # create the old_list by reading the excel in each model
            # ids_old = [
            #     0,19,20,21,22,23,24,1,2,3,4,5,6,13,14,15,16,17,18,7,8,9,10,11,
            #     12,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45
            #     ]
            id_old = ws.cell(row=model_rows[e]+c, column=2).value
            # print(id_old)
            old_list.append('models/' + models[e] + '_' + datasets[x] + '_' + str(id_old))
            # create the ids_new by counting the lenth of the experiments in each model
            middle_list.append('models/' + models[e] + '_' + datasets[x] + '_M' + str(c))
            new_list.append('models/' + models[e] + '_' + datasets[x] + '_' + str(c))
        rename_files_in_list(old_list, middle_list)
        rename_files_in_list(middle_list, new_list)
print("\n")
print("{} files do not exist.".format(counts_old_file_not_exist))
print("{} files already exist.".format(counts_new_file_already_exist))
print("{} files have been renamed.".format(counts_success))