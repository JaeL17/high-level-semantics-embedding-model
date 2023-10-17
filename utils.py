from annoy import AnnoyIndex
import os

def create_annoy(vec, annoy_name, n_tree, save_path):
    t = AnnoyIndex(len(vec[0]), 'angular')
    for count, i in enumerate(vec):
        t.add_item(count, i)
    t.build(n_tree)
    
    if not os.path.exists(save_path): 
        os.makedirs(save_path) 
    t.save(os.path.join(save_path, annoy_name))
    
def annoy_search(annoy_name, test_vec, topn):
    u= AnnoyIndex(len(test_vec[0]), 'angular')
    u.load(annoy_name)
    
    pred, dist = [],[]
    for vec in test_vec:
        res = u.get_nns_by_vector(vec, topn, include_distances=True)
        pred.append(res[0])
        dist.append(res[1])
    #print("(2-dist[i][j]**2)/2")
    return pred, dist 