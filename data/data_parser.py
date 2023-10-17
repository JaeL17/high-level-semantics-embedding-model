import json
import random

print("*****  Loading Dataset  *****")  
with open('./FewRel_train.json', 'r', encoding='utf8') as fp:
    FewRel = json.load(fp)
with open('./pid2name.json', 'r', encoding='utf8') as fp:
    pid = json.load(fp)

map_dict = {}
for i in FewRel.keys():
    target = FewRel[i]
    re_lab = pid[i]
    map_dict[i] = {"re_lab": re_lab[0], "lab_description": re_lab[1], 'tokens_head_tail':[]}
    for j in target:
        map_dict[i]["tokens_head_tail"].append(j)
print(f"*****  Number of relation labels: {len(map_dict.keys())}  *****")

outlst = []
for i in map_dict.keys():
    target = map_dict[i]
    map_dict[i]['dataset']= []
    for j in target['tokens_head_tail']:
        rand_state = True 
        og_sent = ' '.join(j['tokens'])
        og_h = j['h']
        og_t = j['t']
        og_h_item = " ".join(j['tokens'][og_h[-1][0][0]:og_h[-1][0][-1]+1])
        og_t_item = " ".join(j['tokens'][og_t[-1][0][0]:og_t[-1][0][-1]+1])
        
        while rand_state:
            rand = random.sample(target['tokens_head_tail'], 1)[0]
            rand_sent = " ".join(rand['tokens'])
            rand_h = rand['h']
            rand_t = rand['t']
            rand_h_item = " ".join(rand['tokens'][rand_h[-1][0][0]:rand_h[-1][0][-1]+1])
            rand_t_item = " ".join(rand['tokens'][rand_t[-1][0][0]:rand_t[-1][0][-1]+1])     
            if ' '.join(rand['tokens']) != og_sent:
                rand_state= False
        pos_sent = og_sent.replace(og_h_item, rand_h_item).replace(og_t_item, rand_t_item)
        neg_sent = rand_sent.replace(rand_h_item, og_h_item).replace(rand_t_item, og_t_item)
        sent_dict = {"og_sent":og_sent, "pos_sent": pos_sent, "neg_sent":neg_sent, 're_lab':map_dict[i]['re_lab'], 'og':j, 'rand':rand}
        map_dict[i]['dataset'].append(sent_dict)
        outlst.append(sent_dict)
        if len(og_h) + len(og_t) + len(rand_h) + len(rand_t) != 12:
            print('NA')
        
print("*****  Data parsing complete  *****")   
print(f"*****  Total number of training triplet pairs: {len(outlst)}  *****")
print("*****  Saved to fewrel_hl_train.json  *****")
with open('./fewrel_map_dict.json', 'w', encoding='utf8') as fp:
    fp.write(json.dumps(map_dict, ensure_ascii=False, indent = 4))
with open('./fewrel_hl_train.json', 'w', encoding='utf8') as fp:
    fp.write(json.dumps(outlst, ensure_ascii=False, indent = 4))
