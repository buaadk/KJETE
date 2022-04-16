import pickle
import argparse
import os
from collections import OrderedDict


def exec_func(args):
    for split in ["train", "dev", "test"]:
    #for split in ["dev"]:
        with open('%s/bert_all_joint_cosmos_self/%s.pickle' % (args.data_dir, split), "rb") as handle:
            data = pickle.load(handle)
        handle.close()

        context_map = OrderedDict([])
        count = 0
        """
        0:Doc_id
        1:Sample_id
        2:Event id1 event Text1
        3:Event id2 event Text2
        4:关系
        5:00 BERT编码
          01事件标号
          02词性
          03距离
          05-06 left
          07-08 right
        """
        for ex in data:
            start = ex[5][1][1][0]
            end = ex[5][1][-2][0]
            # use doc_id +start token and end token spans as unique context id
            context_id = (ex[0], start, end)
            # sample id,(left id,right id),label_idx,distance,reverse_ind,
            rel = (ex[1], ex[2], ex[3], ex[4], ex[5][3], ex[5][5:9])
            if context_id in context_map:
                context_map[context_id]['rels'].append(rel)
            else:
                context_map[context_id] = {
                    'context_id': count,
                    'doc_id': ex[0],
                    'context': ex[5][0:3],
                    'rels': [rel]
                }
                count += 1
        #save_dir = args.data_dir + "/all_context_0824/"
        save_dir = args.data_dir + "/all_context_tbd/"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open("%s/%s.pickle"%(save_dir,split),"wb") as handle:
            pickle.dump(context_map,handle,protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', type=str, default="../data")
    #p.add_argument('-data_type', type=str, default="comsense")
    p.add_argument('-data_type', type=str, default="tbd")
    args = p.parse_args()
    args.data_dir = args.data_dir + '/' + args.data_type
    exec_func(args)
    print("#finish#")
