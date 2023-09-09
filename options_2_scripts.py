# # script examples
# nohup bash run.sh train ComplEx FB15k-237 0 19 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 --mbs_freq --subsampling_model_temperature 2 --subsampling_model ./models/ComplEx_FB15k-237_0 &
# nohup bash run.sh train ComplEx wn18rr 0 20 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 --mbs_freq --subsampling_model_temperature 1 --subsampling_model ./models/ComplEx_wn18rr_0 &


# # base_config
# bash run.sh train HAKE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 "--modulus_weight 3.5" "--phase_weight 1.0" ${SUB_TYPE}
# # init_config
# bash run.sh train HAKE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 "--modulus_weight 3.5" "--phase_weight 1.0"
# # config
# bash run.sh train HAKE FB15k-237 ${GPUID} ${SUFFIX} 1024 256 1000 9.0 1.0 0.00005 100000 16 "--modulus_weight 3.5" "--phase_weight 1.0" ${SUB_MODEL} ${SUB_TEMP} ${SUB_TYPE}


model_dataset_with_core_config = {
    # Best Configuration for HAKE
    'HAKE FB15k-237 0 00': '1024 256 1000 9.0 1.0 0.00005 100000 16 "--modulus_weight 3.5" "--phase_weight 1.0"',
    'HAKE wn18rr 0 00': '512 1024 500 6.0 0.5 0.00005 80000 8 "--modulus_weight 0.5" "--phase_weight 0.5"',
    'HAKE YAGO3-10 0 00': '1024 256 500 24.0 1.0 0.0002 180000 4 "--modulus_weight 1.0" "--phase_weight 0.5"',
    # Best Configuration for RotatE
    'RotatE FB15k-237 0 00': '1024 256 1000 9.0 1.0 0.00005 100000 16 -de',
    'RotatE wn18rr 0 00': '512 1024 500 6.0 0.5 0.00005 80000 8 -de',
    'RotatE FB15k 0 00': '1024 256 1000 24.0 1.0 0.0001 150000 16 -de',
    'RotatE wn18 0 00': '512 1024 500 12.0 0.5 0.0001 80000 8 -de',
    'RotatE YAGO3-10 0 00': '1024 400 500 24.0 1.0 0.0002 100000 4 -de',
    # # Best Configuration for pRotatE
    # 'pRotatE FB15k-237 0 00': '1024 256 1000 9.0 1.0 0.00005 100000 16',
    # 'pRotatE wn18rr 0 00': '512 1024 500 6.0 0.5 0.00005 80000 8',
    # 'pRotatE FB15k 0 00': '1024 256 1000 24.0 1.0 0.0001 150000 16',
    # 'pRotatE wn18 0 00': '512 1024 500 12.0 0.5 0.0001 80000 8',
    # # Best Configuration for TransE
    # 'TransE FB15k-237 0 00': '1024 256 1000 9.0 1.0 0.00005 100000 16',
    # 'TransE wn18rr 0 00': '512 1024 500 6.0 0.5 0.00005 80000 8',
    # 'TransE FB15k 0 00': '1024 256 1000 24.0 1.0 0.0001 150000 16',
    # 'TransE wn18 0 00': '512 1024 500 12.0 0.5 0.0001 80000 8',
    # Best Configuration for ComplEx
    'ComplEx FB15k-237 0 00': '1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001',
    'ComplEx wn18rr 0 00': '512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005',
    'ComplEx FB15k 0 00': '1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002',
    'ComplEx wn18 0 00': '512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001',
    # # Best Configuration for DistMult
    # 'DistMult FB15k-237 0 00': '1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001',
    # 'DistMult wn18rr 0 00': '512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005',
    # 'DistMult FB15k 0 00': '1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002',
    # 'DistMult wn18 0 00': '512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001',
}


position_with_options = {
    # compulsory
    "narrate_bash": 'nohup bash run.sh train',
    # "model": ['HAKE', 'ComplEx', 'RotatE', 'pRotatE', 'TransE', 'DistMult'],
    "model": ['HAKE', 'ComplEx', 'RotatE'],
    # "dataset": ['FB15k-237', 'wn18rr', 'FB15k', 'wn18', 'YAGO3-10'],
    "dataset": ['FB15k-237', 'wn18rr'],
    "GPUID": 0, # modify this manually
    "SUFFIX": 00, # modify this manually
    "core_config": 000, # find the answer using function
    # elective 1
    "CNT_SUB_TYPE": ['--cnt_default', '--cnt_freq', '--cnt_uniq'],
    # elective 2
    "narrate_submodel": '--subsampling_model',
    "SUB_MODEL": './models/', # ./models/ComplEx_FB15k-237_0
    "narrate_temperature": '--subsampling_model_temperature',
    "SUB_TEMP": ['2', '1', '0.5', '0.1', '0.05', '0.01'],
    "MBS_SUB_TYPE": ['--mbs_default', '--mbs_freq', '--mbs_uniq'],
    "narrate_ratio": '--mbs_ratio', 
    "MBS_RATIO": ['0.25', '0.5', '0.75'],
    "narrate_end": '&',
}


def make_script():
    scripts = []
    narrate_bash = position_with_options.get("narrate_bash")
    narrate_end = position_with_options.get("narrate_end")
    for m in range(len(position_with_options["model"])):
        model = position_with_options["model"][m]
        for d in range(len(position_with_options["dataset"])):
            dataset = position_with_options["dataset"][d]
            # check if the model and dataset, as key, has core config
            key = model + ' ' + dataset + ' ' + '0' + ' ' + '00'
            if key in model_dataset_with_core_config:
                core_config = model_dataset_with_core_config[key]
                script_compulsory = narrate_bash + ' ' + model + ' ' + dataset + ' ' + '0' + ' ' + '00' + ' ' + core_config
                # # elective 1: base_config
                # for cnt in position_with_options["CNT_SUB_TYPE"]:
                #     script_elective = ' ' + cnt
                #     script = script_compulsory + script_elective + ' ' + narrate_end
                #     scripts.append(script)
                # # elective 2: init_config
                # script = script_compulsory + ' ' + narrate_end
                # scripts.append(script)
                # eletive 3: config
                for submodel in position_with_options["model"]:
                    narrate_submodel = position_with_options["narrate_submodel"]
                    submodel = position_with_options["SUB_MODEL"] + submodel + '_' + dataset + '_0'
                    for mbs in position_with_options["MBS_SUB_TYPE"]:
                        for temperature in position_with_options["SUB_TEMP"]:
                            narrate_temperature = position_with_options["narrate_temperature"]
                            for ratio in position_with_options["MBS_RATIO"]:
                                narrate_ratio = position_with_options["narrate_ratio"]
                                script_elective = ' ' + narrate_submodel + ' ' + submodel + ' ' + mbs + ' ' + narrate_temperature + ' ' + temperature + ' ' + narrate_ratio + ' ' + ratio
                                script = script_compulsory + script_elective + ' ' + narrate_end
                                scripts.append(script)
    return scripts


import sys

if len(sys.argv) > 1:
    var_name = sys.argv[1]
else:
    var_name = 'now'

import openpyxl

wb = openpyxl.load_workbook(filename='results/temp_script.xlsx')
# default sheet is the first sheet
ws = wb.active

scripts = make_script()
for i in range(len(scripts)):
    ws.cell(row=i+1, column=1).value = scripts[i]

wb.save('results/{}_scripts.xlsx'.format(var_name))

