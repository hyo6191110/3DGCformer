import os

def command_generate(data_use,m,p,f):
    datasets={
        'GZZone': [500, 0, 12, 12, [12, 24, 36]],
        'WX': [500, 2, 12, 12, [12, 24, 36]],
        'EZ': [500, 2, 12, 12, [12, 24, 36]],
        'JHT': [500, 0, 7, 7, [14, 28, 56]],
        'JHTfull': [500, 0, 7, 7, [14, 28, 56]],
        'NYC15m': [500, 2, 12, 12, [24, 36, 48]],
        'NYC60m': [500, 2, 12, 12, [12]],
    }
    head_str = 'python train.py --batch 16 --train_split 0.6 --model ' + m + ' --patch_method ' + p + ' --fusion_method ' + f
    output = ''
    for data in data_use:
        info = datasets[data]
        epoch = info[0]
        normalize = info[1]
        I = info[2]
        L = info[3]
        O_list = info[4]
        for O in O_list:
            cmd_str = head_str + ' --data ' + data + ' --epoch ' + str(epoch) + ' --normalize ' + str(normalize) + \
                      ' --T_in ' + str(I) + ' --T_out ' + str(O) + ' --T_label ' + str(L) + ' ;\n'
            output = output + cmd_str
    return output


def label_task():
    data_use=['GZZone', 'WX', 'EZ', 'JHT', 'NYC60m']
    m = 'GCformer'
    p = 'embed'
    f = 'concat'
    datasets={
        'GZZone': [500, 0, 12, [0, 6], [12, 24, 36]],
        'WX': [500, 2, 12, [0, 6], [12, 24, 36]],
        'EZ': [500, 2, 12, [0, 6], [12, 24, 36]],
        'JHT': [500, 0, 7, [0, 3], [14, 28, 56]],
        'NYC60m': [500, 2, 12, [0, 6], [12, 24, 36]],
    }
    head_str = 'python train.py --batch 16 --train_split 0.6 --model ' + m + ' --patch_method ' + p + ' --fusion_method ' + f
    output = ''
    for data in data_use:
        info = datasets[data]
        epoch = info[0]
        normalize = info[1]
        I = info[2]
        L_list = info[3]
        O_list = info[4]
        for L in L_list:
            for O in O_list:
                cmd_str = head_str + ' --data ' + data + ' --epoch ' + str(epoch) + ' --normalize ' + str(normalize) + \
                        ' --T_in ' + str(I) + ' --T_out ' + str(O) + ' --T_label ' + str(L) + ' ;\n'
                output = output + cmd_str
    save_path = os.path.join('cmd', 'cmd_label.sh')
    with open(save_path, 'w', encoding='utf-8') as f:
        s = output
        f.write(s)


def write_one_task(m = 0,p = 0,f = 0):
    data_use = ['GZZone', 'WX', 'EZ', 'JHT', 'NYC60m']
    model = ['GCformer', 'transformer', 'GCN-OD', 'GCN-n', 'GCformer-nopatch']
    patch = ['embed', 'divide']
    fusion = ['concat', 'add', 'mul']
    command = command_generate(data_use, model[m], patch[p], fusion[f])
    if p != 0:
        save_path = 'cmd_' + model[m] + '_P' + patch[p] + '.sh'
    elif f != 0:
        save_path = 'cmd_' + model[m] + '_F' + fusion[f] + '.sh'
    else:
        save_path = 'cmd_' + model[m] + '.sh'
    save_path = os.path.join('cmd', save_path)
    with open(save_path, 'w', encoding='utf-8') as f:
        s = command
        f.write(s)


def write_multi_tasks():
    data_use = ['GZZone', 'WX', 'EZ', 'JHT', 'NYC60m']
    data_use = ['NYC60m']
    #model = ['GCformer']
    #model = ['Autoformer', 'Informer', 'ConvLSTM', 'MPGCN', 'ODCRN', 'STGCN']
    model = ['transformer', 'GCN-OD', 'GCN-n', 'GCformer-nopatch']
    #patch = ['divide']
    patch = ['embed']
    #fusion = ['add', 'mul']
    fusion = ['concat']
    multi_command = ''
    for m in range(len(model)):
        for p in range(len(patch)):
            for f in range(len(fusion)):
                command = command_generate(data_use, model[m], patch[p], fusion[f])
                multi_command += command
    save_path =  os.path.join('cmd', 'cmd_multi_task.sh')
    with open(save_path, 'w', encoding='utf-8') as f:
        s = multi_command
        f.write(s)

if __name__ == '__main__':
    #write_one_task()
    write_multi_tasks()
    #cmd=label_task()
