import tqdm
import numpy

def cosine_similarity(v1, v2):
    return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))

def calc_topk_pretrain(funcs1, funcs2, k):
    hit_count_top = []
    similars = []
    differs = []
    sim_sorted = []
    for i in tqdm.tqdm(range(len(funcs1)), 'calc top-{}'.format(k)):  
        sub_similars = []   
        for f2 in funcs2:
            similar = cosine_similarity(funcs1[i].embedding, f2.embedding)
            sub_similars.append(similar)
        sub_sim_sorted = numpy.argsort(sub_similars)

        similars.append(sub_similars)
        sim_sorted.append(sub_sim_sorted)

        sub_hit_count_top = [0] * (k + 1)
        for j in range(1, min(len(sub_sim_sorted) + 1, k + 1)):
            id = sub_sim_sorted[-j]

            if funcs2[id].meta['name'] == funcs1[i].meta['name']:
                sub_hit_count_top[j] += 1
                break
        hit_count_top.append(sub_hit_count_top)

    return hit_count_top, sim_sorted, similars, differs

def evaluate_MRR_k(hit_count_top, pool_size, k = 1):
    hittotal = 0
    ret = []
    if len(hit_count_top) == 0 or pool_size == 0:
        return ret

    for i in range(1, k + 1):
        for hit in hit_count_top:
            hittotal += hit[i]
        ret.append(1/hittotal)
    return ret / pool_size

def evaluate_Recall_k(hit_count_top, pool_size, k = 1):
    hittotal = 0
    ret = []
    if len(hit_count_top) == 0 or pool_size == 0:
        return ret

    for i in range(1, k + 1):
        for hit in hit_count_top:
            hittotal += hit[i]
        ret.append(hittotal/pool_size)
    return ret

def filter_similar_pairs(funcs1, funcs2):
    funcs1_sim = []
    funcs2_sim = []
    for i in range(len(funcs1)):
        for j in range(len(funcs2)): 
            try:
                funcs1[i].meta['name'] = funcs1[i].meta['name'].strip('dbg.')
                funcs2[j].meta['name'] = funcs2[j].meta['name'].strip('dbg.')
                funcs1[i].meta['name'] = funcs1[i].meta['name'].strip('sym.')
                funcs2[j].meta['name'] = funcs2[j].meta['name'].strip('sym.')
                if funcs1[i].meta['name'] == funcs2[j].meta['name']:
                    funcs1_sim.append(funcs1[i])  
                    funcs2_sim.append(funcs2[j])
                    break
            except:
                pass
    return funcs1_sim, funcs2_sim

def evaluate_performance(funcs_src, funcs_dst, topk=1):
    #print_count = 0
    hit_count_top, sim_sorted, similars, differs = calc_topk_pretrain(funcs_src, funcs_dst, topk)

    #for i in range(len(funcs1_sim)):
        #sub_similars = similars[i]
        #sub_sim_sorted = sim_sorted[i]

        #if len(funcs1_sim) > 20:
        #    if hit_count_top[i][2] != 1:
        #        continue
        #if print_count >= 1:
        #    continue
        #print_count += 1

        #print("INPUT_{}: {}-{} {}".format(i, funcs1_sim[i].target, funcs1_sim[i].compiler, funcs1_sim[i].meta['name']))
        #for j in range(1, min(len(sub_sim_sorted) + 1, topk + 1)):
        #    id = sub_sim_sorted[-j]
        #    print(" -FIND_{}: [{:.4f}] {}-{} {} ".format(j, sub_similars[id], funcs2_sim[id].target, funcs2_sim[id].compiler, funcs2_sim[id].meta['name']))
        
       
    recall_k = evaluate_Recall_k(hit_count_top, pool_size=len(funcs_src), k = topk)

    print('Recall@k, k = 1-10:'.format(topk)) # 
    for v in recall_k:
        print('{:.2f} '.format(v), end="")
