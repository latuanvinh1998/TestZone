import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from torchvision import transforms
import cv2
import time

def read_pairs(pairs_filename):
	pairs = []
	with open(pairs_filename, 'r') as f:
		for line in f.readlines()[1:]:
			pair = line.strip().split()
			pairs.append(pair)
	return np.array(pairs, dtype = object)

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths(lfw_dir, pairs):
	nrof_skipped_pairs = 0
	path_list = []
	issame_list = []
	for pair in pairs:
		if len(pair) == 3:
			path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
			path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
			issame = True
		elif len(pair) == 4:
			path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
			path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
			issame = False
		if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
			path_list += (path0,path1)
			issame_list.append(issame)
		else:
			nrof_skipped_pairs += 1
	if nrof_skipped_pairs>0:
		print('Skipped %d image pairs' % nrof_skipped_pairs)
	
	return path_list, issame_list

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, 

def load_lfw(pair_path, lfw_dir, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    pairs = read_pairs(os.path.expanduser(pair_path))
    lfw_paths, y_true = get_paths(os.path.expanduser(lfw_dir), pairs)

    nrof_images = len(lfw_paths)
    labels_array = np.arange(nrof_images)           
    image_paths_array = np.array(lfw_paths)
    images=[]


    for i in range(nrof_images):
        img = cv2.imread(lfw_paths[i])
        img=cv2.resize(img,(112,112))
        img=transform(img)
        img = img.type(torch.FloatTensor)
        images.append(img)

    img_batch = torch.utils.data.DataLoader(images, batch_size=batch_size)
    return img_batch, y_true, nrof_images

def model_evaluate(model, img_batch, y_true, nrof_images, nrof_fold=10 ,embedding_size=512):

    print("\n======Evaluating Model ...======")
    evaluate_start = time.time()

    ####### START EVALUATE ######

    emb = np.zeros((nrof_images, embedding_size))

    idx_start = 0
    model.eval()

    with torch.no_grad():

        for batch in iter(img_batch):

            batch = batch.to(torch.device("cuda:0"))
            embedding = model(batch).cpu()
            emb[idx_start:idx_start+32,:] = embedding
            idx_start += 32

    tpr, fpr, acc = evaluate(emb, y_true, nrof_folds=nrof_fold)
    print('Evaluating time: %.3fs' % (time.time() - evaluate_start))
    print('Accuracy: %1.3f+-%1.3f \n' % (np.mean(acc), np.std(acc)))
    return np.mean(acc)




