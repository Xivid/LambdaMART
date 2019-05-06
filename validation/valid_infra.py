import os

print("Loading scores of baseline model...")
baselines = {}
with open('mslr.baseline.log') as fin:
    for line in fin:
        if "Loading training dataset" in line:
            train_set = os.path.basename(line.split()[5])
        elif "Loading validation dataset" in line:
            valid_set = os.path.basename(line.split()[5])
        elif "ndcg" in line:
            tokens = line.strip().split()
            itr = tokens[2]
            scores_iter = {}
            for token in tokens[3:]:
                metric, score = token.split(':')
                scores_iter[metric] = score
            baselines[itr] = scores_iter

print("Running optimized model...")
os.system("./lambdamart mslr.vali.conf > log")

# validation of ranking scores
flag = True
with open("log") as fin:
    for line in fin:
        if "Loading training dataset" in line:
            _train_set = os.path.basename(line.split()[5])
            if train_set != _train_set:
                print("Error: different training dataset. {} v.s. {}".format(train_set, _train_set))
                flag = False
        elif "Loading validation dataset" in line:
            _valid_set = os.path.basename(line.split()[5])
            if valid_set != _valid_set:
                print("Error: differennt validation dataset. {} v.s. {}".format(valid_set, _valid_set))
                flag = False
        elif "ndcg" in line:
            tokens = line.strip().split()
            itr = tokens[2]
            baseline_iter = baselines[itr]
            for token in tokens[3:]:
                metric, score = token.split(':')
                if metric in baseline_iter:
                    if (float(score) - float(baseline_iter[metric]))**2 > 1e-6:
                        print("Error at {}: {} v.s. {}".format(metric,baseline_iter[metric],score))
                        flag = False
    if(flag):
        print("Pass the validation!")
