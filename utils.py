import time
import faiss
import numpy as np
import torch

# Below codes are from Deep Clustering for Unsupervised Learning of Visual Features github code        
def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

def cluster_assign(images_lists, dataset):
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

def run_kmeans(x, nmb_clusters):
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    D, I = index.search(x, 1)
    
    print(clus.centroids)
    
    losses = faiss.vector_to_array(clus.obj)
    print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1], [float(d[0]) for d in D]

def compute_features_scores(dataloader, model, N, batch_size):
    model.eval()
    # discard the label information in the dataloader
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.cuda()
            aux = model(inputs)[0].data.cpu().numpy()
            aux = aux.reshape(-1, 512)
            score = model(inputs)[1].data.cpu().numpy()
            
            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                scores = np.zeros((N, score.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            score = score.astype('float32')
            
            if i < len(dataloader) - 1:
                features[i * batch_size: (i + 1) * batch_size] = aux
                scores[i * batch_size: (i + 1) * batch_size] = score
            else:
                features[i * batch_size:] = aux
                scores[i * batch_size:] = score
                
    return features, scores

class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data):
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss, _ = run_kmeans(xb, self.k)
        self.images_lists = [[] for i in range(self.k)]
        label = []
        for i in range(len(data)):
            label.append(I[i])
            self.images_lists[I[i]].append(i)
            
        label = np.array(label).reshape(-1,1)
        print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss, label