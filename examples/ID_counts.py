from collections import Counter
import json

def fea(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)  # 直接返回 Python 字典或列表

    #print(f'result:{data}')
    s=[0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


    dic={}
    templist=[]
    caselist=[]
    for i,j in enumerate(data):
        case_id=j['case_id']
        acc=j['post']['rewrite_acc']
        d={'case_id': case_id, 'rewrite_acc': acc, 'selected_id': s[i]}
        dic[str(case_id)]=d
        if acc[0]<1:
            templist.append(s[i])
        if acc[0]<1 and s[i]==1 :
            caselist.append(case_id)
    print(dic)
    with open(file.replace('results','results_zhengliu'), "w") as f:
        json.dump(dic, f, indent=4)
    print('error:',Counter(templist))
    print('----------')
    print(caselist)

def dealJson(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 直接返回 Python 字典或列表
    caseID={'0_error':[],'1_error':[],'2_error':[]}
    for k,v in data.items():
        if v['rewrite_acc'][0]<1 and v['selected_id']==0:
            caseID['0_error'].append(k)
        elif v['rewrite_acc'][0]<1 and v['selected_id']==1:
            caseID['1_error'].append(k)
        elif v['rewrite_acc'][0]<1 and v['selected_id']==2:
            caseID['2_error'].append(k)
    print(caseID)
    print(len(caseID['0_error']))
    print(len(caseID['1_error']))
    print(len(caseID['2_error']))
    print('----------')

# def mk_newdata(path):
#     caseID=['0', '1', '3', '4', '6', '8','2', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '26', '27', '28']
#     k=30
#     add_num=k-len(caseID)
#
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)  # 直接返回 Python 字典或列表
#
#     new_edit_ID = [int(i) for i in caseID]
#     new_data=data[new_edit_ID]
#     adddata=data[k:k+add_num]
#     for d in adddata:
#         new_data.append(d)
#     with open(path.replace('.json','_new.json'), "w") as f:
#         json.dump(new_data, f, indent=4)
def mk_nextgroup_editdata(path):
    caseID= ['0', '1', '3', '4', '6', '8','2', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '26', '27', '28']
    k=30
    new_data_num=k-len(caseID)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 直接返回 Python 字典或列表
    nextgroup_caseID=[int(i) for i in caseID]

    nextgroup_editdata=[data[int(i)] for i in caseID]
    new_data=data[k:k+new_data_num]
    for d in new_data:
        nextgroup_editdata.append(d)
    print(len(nextgroup_editdata))
    with open(path.replace('.json','_new.json'), "w") as f:
        json.dump(nextgroup_editdata, f, indent=4)

if __name__ == '__main__':
  #   ID=[0, 0, -2, 2, 2, -2, 2, 2, -2, 0, 2, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 2, 2, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 2, 2, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2, 2, 2, -2]
  # #ID=[0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 1, 1, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2]
  #   cnt_0=0
  #   cnt_1=0
  #   cnt_2=0
  #   cnt_ori=0
  #   selected_ID=[]
  #   for i,j in enumerate(ID):
  #       if i%3==0:
  #           selected_ID.append(j)
  #   counts=Counter(selected_ID)
  #   print(counts)
  #   print(selected_ID)

##需要提取 "case_id":，"post": {"rewrite_acc":}
    file='D:/EasyEdit-main/examples/logs/results_zhengliu.json'
    # fea(file)
    dealJson(file)
    #path='D:\EasyEdit-main\data\wise\ZsRE\zsre_mend_edit.json'
    # mk_nextgroup_editdata(path)