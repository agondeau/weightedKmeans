/** @file cluster.c
 *  @brief Function declarations for the k-means clustering.
 *
 *  This contains the function declarations for the k-means
 *  clustering and eventually any macros, constants,or 
 *  global variables.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

/* -- Includes -- */

/* libc includes. */
#include <stdlib.h>

/* math header file. */
#include <math.h>

/* cluster header file. */
#include "cluster.h"

/* log include. */
#include "log.h"

/* -- Defines -- */

/* Clustering constant defines. */
#define NB_ITER 100

/* Macro defines. */
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

/* -- Enumerations -- */

/** @brief Contains the different distance calculation types.
 *
 */
typedef enum _eDistanceType 
{
    DISTANCE_EUCLIDEAN = 0,
    DISTANCE_OTHER
} eDistanceType;

/** @brief Proceeds to a fake assigenment of data to  
 *         the different clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, cluster *c, uint32_t k);

/** @brief Allocates memory for the clusters dimensions.  
 *
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k);

/** @brief Frees allocated memory for the clusters dimensions.  
 *
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_freeClusters(cluster *c, uint32_t k);

/** @brief Chooses random data as centroids.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Assigns data to the nearest centroid.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged. 
 *  @return Void.
 */
static double CLUSTER_assignDataToCentroids7(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns and transfer data to the nearest cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged.
 *  @return Void.
 */
static void CLUSTER_assignDataToCentroids11(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns weighted data to the nearest centroid
 *         for features weighted algorithm.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged.
 *  @return Void.
 */
static void CLUSTER_assignWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns weighted data to the nearest centroid
 *         for objects (and features) weighted algorithm.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param internalFeatureWeights The boolean specifying if 
 *              the features weights are computed internally.
 *  @param  feaWeiMet The method used to computed the features
 *                    weights internally.
 *  @param internalObjectsWeights The boolean specifying if 
 *              the objects weights are computed internally.
 *  @param  objWeiMet The method used to computed the objects
 *                    weights internally.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @param wss The array of wss per cluster.
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged.
 *  @return Void.
 */
static void CLUSTER_assignWeightedDataToCentroids27(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, eMethodType feaWeiMet, bool internalObjectWeights, eMethodType objWeiMet, double **dist, double wss[k], bool *conv);

/** @brief Computes the squared distance between a point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d);

/** @brief Computes the squared and features weighted
 *         distance between a point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredFWDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d);

/** @brief Computes the squared distance between two clusters.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param ci The pointer to the cluster i.
 *  @param cj The pointer to the cluster j.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredDistanceClusterToCluster(cluster *ci, cluster *cj, uint64_t p, eDistanceType d);

/** @brief Computes the squared distance between a weighted 
 *         point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @param fw The pointer to the features weights.
 *  @param ow The point weight.
 *  @return The computed distance between the weigted point 
 *          and the cluster.
 */
static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow);

/** @brief Updates the centroids positions.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Transfer a point from a cluster to another.
 *
 *  @param dat The pointer to data.
 *  @param n The id of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the cluster.
 *  @param k The id of the cluster. 
 *  @return Void.
 */
static void CLUSTER_transferPointToCluster2(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK);

/** @brief Add a point to a cluster.
 *
 *  @param dat The pointer to data.
 *  @param c The pointer to the cluster.
 *  @return Void.
 */
static void CLUSTER_addPointToCluster(data *dat, cluster *c);

/** @brief Remove a point from a cluster.
 *
 *  @param dat The pointer to data.
 *  @param c The pointer to the cluster.
 *  @return Void.
 */
static void CLUSTER_removePointFromCluster(data *dat, cluster *c);

/** @brief Computes the classical version of k-means  
 *         algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return The sum of squared errors for the clustering.
 */
static double CLUSTER_kmeans4(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);

/** @brief Computes the features 
 * version of k-means algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param internalFeatureWeights The boolean that 
 *         specified if the features weights come 
 *         from internal computation or from a file.
 *  @param featureWeightsMethod The feature weights calculation
 *         method.         
 *  @return The sum of squared errors for the clustering.
 */
static double CLUSTER_featuresWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod);

/** @brief Computes the objects (and features) 
 * version of k-means algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param internalFeatureWeights The boolean specifying if 
 *              the features weights are computed internally.
 *  @param  feaWeiMet The method used to computed the features
 *                    weights internally.
 *  @param internalObjectsWeights The boolean specifying if 
 *              the objects weights are computed internally.
 *  @param  objWeiMet The method used to computed the objects
 *                    weights internally.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return The sum of squared errors for the clustering.
 */
static double CLUSTER_weightedKmeans7(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod, bool internalObjectWeights, eMethodType objectWeightsMethod, double **dist);

/** @brief Computes the silhouette score for a 
 *         clustering.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return The computed silhouette score for the clustering.
 */
static double CLUSTER_computeSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes the distance between a point
 *         and an other point .
 *
 *  @param iDat The pointer to the first point.
 *  @param jDat The pointer to the second point.
 *  @param p The number of data dimensions.
 *  @param d The type of distance calculation. 
 *  @return The computed distance between the first point 
 *          and the second point.
 */
static double CLUSTER_computeDistancePointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d);

/** @brief Computes the total sum of squares for 
 *         the data.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @return The computed total sum of squares for the data.
 */
static double CLUSTER_computeTSS(data *dat, uint64_t n, uint64_t p); 

/** @brief Computes the variance ratio criterion for 
 *         a clustering.
 *
 *  @param TSS The computed total sum of squares for the data.
 *  @param SSE The sum of squared errors for the clustering.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return The computed variance ratio criterion for the 
 *          clustering.
 */
static double CLUSTER_computeVRC2(data *dat, cluster *c, double SSE, uint64_t n, uint64_t p, uint32_t k);

/** @brief Computes the Calinski-Harabasz criterion for 
 *         a clustering.
 *
 *  @param TSS The computed total sum of squares for the data.
 *  @param SSE The sum of squared errors for the clustering.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return The computed Calinski-Harabasz criterion for the 
 *          clustering.
 */
static double CLUSTER_computeCH(double TSS, double SSE, uint64_t n, uint32_t k); 

/** @brief Initializes objects weights to 1. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of objects. 
 *  @return Void.
 */
static void CLUSTER_initObjectWeights(data *dat, uint64_t n);

/** @brief Computes features weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param m The method for features weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m);

/** @brief Computes features weights in a specific 
 *         cluster via different methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The id of the cluster.
 *  @param fw The features weights.
 *  @param m The method for objects weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m);

/** @brief Computes features weights via dispersion
 *         score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param fw The features weights.
 *  @param norm The norm from Lp-spaces.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeightsViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint8_t norm);

/** @brief Computes features weights in a specific 
 *         cluster via dispersion score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param fw The features weights.
 *  @param norm The norm from Lp-spaces.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeightsInClusterViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, uint8_t norm);

/** @brief Computes feature dispersion.
 *
 *  @param dat The pointer to a datum.
 *  @param p The specific dimension.
 *  @param c The pointer to the cluster.
 *  @param norm The norm from Lp-spaces.
 *  @return The computed dispersion.
 */
static double CLUSTER_computeFeatureDispersion(data *dat, uint64_t p, cluster *c, uint8_t norm);

/** @brief Computes objects weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param m The method for objects weights calculation.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeights3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m, double **dist);

/** @brief Computes objects weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The id of the cluster.
 *  @param m The method for objects weights calculation.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m, double **dist);

/** @brief Computes objects weights via silhouette
 *         score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes objects weights via silhouette
 *         score. The sum of objects weights in a cluster
 *         is equal to the number of points in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes objects weights via silhouette
 *         score in a cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist);

/** @brief Computes objects weights via silhouette
 *         score in a cluster. The sum of objects weights in 
 *         the cluster is equal to the number of points in the 
 *         cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own).
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own). 
 *         The sum of objects weights in a cluster is equal to 
 *         the number of points in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own).
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own) 
 *         in a cluster. The sum of objects weights in the 
 *         cluster is equal to the number of points in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes objects weights via the sum of distances 
 *         with the other centroids (different from its own).
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the sum of distances 
 *         with the other centroids (different from its own)
 *         in a cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes objects weights via the median.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMedian2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the median. 
 *         The sum of objects weights in the cluster 
 *         is equal to the number of objects in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the median in a cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK);

/** @brief Computes objects weights via the median in a cluster.
 *         The sum of objects weights in the cluster 
 *         is equal to the number of objects in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK);

/** @brief Computes the sum of squared errors for a 
 *         clustering. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @return The computed sum of squared errors for a 
 *          clustering.
 */
static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes the within sum of squares. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param wss The array of wss per cluster.
 *  @return Void.
 */
static void CLUSTER_computeNkWeightedWSS(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double wss[k]);

/** @brief Computes the within sum of squares. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param indK The cluster index.
 *  @return The computed wss for the cluster indK.
 */
static double CLUSTER_computeNkWeightedWSSInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes the matrix of distances 
 *         points to points. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param dist The triple pointer to the matrix of distances.
 *  @return Void.
 */
static void CLUSTER_ComputeMatDistPointToPoint2(data *dat, uint64_t n, uint64_t p, double ***dist);

/** @brief Frees the matrix of distances 
 *         points to points. 
 *
 *  @param n The number of the data.
 *  @param dist The triple pointer to the matrix of distances.
 *  @return Void.
 */
static void CLUSTER_FreeMatDistPointToPoint(uint64_t n, double ***dist);

/* -- Function definitions -- */

void CLUSTER_computeKmeans4(data *dat, uint64_t n, uint64_t p, uint32_t kmin, uint32_t kmax,uint32_t nbRep)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax+1], statVRC2[kmax+1], statCH[kmax+1];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k

    // Initialize statistics
    for(k=kmax;k>=kmin;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = 0.0;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);
    //double TSS2 = CLUSTER_computeTSS2(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);


    //WRN("Iteration %d", i);
    for(k=kmax;k>=kmin;k--) // From kMax to kMin
    {
        //k = kmax;    

        /*cluster test[k], test2[k];
        double WSS[k];
        double WSS2[k];
        uint32_t assign[n];
        CLUSTER_initClusters(p, test, k);
        CLUSTER_initClusters(p, test2, k);*/
        for(i=0;i<nbRep;i++) // Number of replicates
        {
            //INF("Compute for k = %d", k);

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            /*CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
            CLUSTER_randomCentroids(dat, n, p, c, k);
            double SSE = CLUSTER_kmeans_Lloyd(dat, n, p, k, c);*/
            //double SSE = CLUSTER_kmeans2(dat, n, p, k, c);
            //double SSE = CLUSTER_kmeans3(dat, n, p, k, c);
            double SSE = CLUSTER_kmeans4(dat, n, p, k, c);

            //SAY("SSE = %lf, TSS = %lf, n = %ld, k = %d", SSE, TSS2, n, k);

            /*double wss[k];
            uint32_t o;
            CLUSTER_computeWSS(dat, n, p, c, k, wss);*/

            // Compute silhouette statistic
            //double sil = CLUSTER_computeSilhouette(dat, n, p, c, k);
            //double sil = CLUSTER_computeSilhouette2(dat, n, p, c, k, dist);
            //double sil = CLUSTER_computeSilhouette3(dat, n, p, c, k, dist);
            double sil = CLUSTER_computeSilhouette4(dat, n, p, c, k, dist);
            //SAY("Silhouette = %lf", sil);

            // Compute VRC statistic
            double vrc2 = CLUSTER_computeVRC2(dat, c, SSE, n, p, k);
            //SAY("VRC2 = %lf", vrc2);

            // Compute CH statistic
            double ch = CLUSTER_computeCH(TSS, SSE, n, k);
            //SAY("CH = %lf", ch);

            // Compute CH statistic
            //double ch2 = CLUSTER_computeCH(TSS2, SSE, n, k);
            //SAY("CH = %lf", ch);

            // Save best silhouette statistic for each k
            /*if(sil > statSil[k] || i == 0)
            {
                statSil[k] = sil;
                //INF("sil[%d] = %lf, SSE = %lf, it = %ld", k, statSil[k], SSE, i);
            }*/

            bool clustNull = false;
            for(o=0;o<k;o++)
            {
                //SAY("c[%d].nbData = %ld", o, c[o].nbData);
                if(c[o].nbData == 0)
                {
                    /*if(wss[o] == 0)
                       ERR("c.nbData = %ld", c[o].nbData); */
                    clustNull = true;
                }    
            }

            // Save best VRC statistic for each k
            if((vrc2 > statVRC2[k] || i == 0) && clustNull == false)
            {
                statVRC2[k] = vrc2;
                // Save data membership for each k (VRC2)
                for(j=0;j<n;j++)
                {
                    vrc2Grp[k][j] = dat[j].clusterID;
                }
            }

            // Save best CH statistic for each k
            if((ch > statCH[k] || i == 0) && clustNull == false)
            {
                statCH[k] = ch;
                // Save data membership for each k (CH)
                for(j=0;j<n;j++)
                {
                    chGrp[k][j] = dat[j].clusterID;
                }
            }

            // Save best CH statistic for each k
            /*if((ch2 > statCH2[k] || i == 0) && clustNull == false)
            {
                statCH2[k] = ch2;
            }*/

            if((sil > statSil[k] || i == 0) && clustNull == false)
            {
                statSil[k] = sil;
                // Save data membership for each k (CH)
                for(j=0;j<n;j++)
                {
                    silGrp[k][j] = dat[j].clusterID;
                }
                //INF("sil[%d] = %lf, SSE = %lf, it = %ld", k, statSil[k], SSE, i);
                /*uint32_t l;
                for(l=0;l<k;l++)
                {
                    WSS2[l] = wss[l];
                    for(j=0;j<p;j++)
                        test2[l].centroid[j] = c[l].centroid[j];

                    test2[l].nbData = c[l].nbData;
                }*/
            }
            
            // Save best SSE statistic for each k
            /*if((SSE < statSSE[k] || i == 0) && clustNull == false)
            {
                statSSE[k] = SSE;
                uint32_t l;
                for(l=0;l<k;l++)
                {
                    WSS[l] = wss[l];
                    for(j=0;j<p;j++)
                        test[l].centroid[j] = c[l].centroid[j];

                    test[l].nbData = c[l].nbData;
                }
                uint64_t o;
                for(o=0;o<n;o++)
                {
                    assign[o] = dat[o].clusterID;
                }
            }*/

            // Free clusters dimension memory
            CLUSTER_freeClusters(c, k);
        }

        /*INF("Best SSE : %lf for k = %d (TSS = %lf, Betweenness = %lf, Bet / Tot = %lf%)", statSSE[k], k, TSS, (TSS - statSSE[k]), ((TSS - statSSE[k]) / TSS));
        uint64_t o;
        for(o=0;o<n;o++)
        {
             dat[o].clusterID = assign[o];
        }
        //statSil[k] = CLUSTER_computeSilhouette4(dat, n, p, test, k, dist);
        INF("Best sil : %lf for k = %d", statSil[k], k);*/
        
        //uint32_t l;
        /*for(l=0;l<k;l++)
        {
            SAY("WSS[%d] = %lf (nbData = %ld)", l, WSS[l], test[l].nbData);
        }*/
        /*for(l=0;l<k;l++)
        {
            SAY("WSS[%d] = %lf (nbData = %ld)", l, WSS2[l], test2[l].nbData);
        }
        CLUSTER_freeClusters(test, k);
        CLUSTER_freeClusters(test2, k);*/
    }

    // Retrieve the overall best statistics
    double silMax = -1.0, vrc2Max = 0.0, chMax = 0.0;
    uint32_t kSilMax, kVrc2Max, kChMax;
    for(k=kmax;k>=kmin;k--)
    {
        if(statSil[k] > silMax)
        {
            silMax = statSil[k];
            kSilMax = k;
        }

        if(statVRC2[k] > vrc2Max)
        {
            vrc2Max = statVRC2[k];
            kVrc2Max = k;
        }

        if(statCH[k] > chMax)
        {
            chMax = statCH[k];
            kChMax = k;
        }
    }

    WRN("");
    WRN("Final statistics -----------------------");
    INF("Best silhouette : %lf for k = %d", silMax, kSilMax);
    INF("Best VRC2 : %lf for k = %d", vrc2Max, kVrc2Max);
    INF("Best CH : %lf for k = %d", chMax, kChMax);
    //INF("Best CH2 : %lf for k = %d", ch2Max, kCh2Max);
    WRN("");
    WRN("Data membership for best indice scores -");
    /*printf("dataId\t");
    for(k=kmax;k>=K_MIN;k--)
        printf("%d-Gr\t", k);
    INF("");
    for(j=0;j<n;j++)
    {
        printf("%ld\t", j);
        for(k=kmax;k>=K_MIN;k--)
            printf("%d\t", chGrp[k][j]);
        INF("");
    }*/
    printf("dataId\t");
    printf("Sil\t");
    printf("VRC2\t");
    printf("CH\t\n");
    printf("\t%d-Gr\t", kSilMax);
    printf("%d-Gr\t", kVrc2Max);
    printf("%d-Gr\t", kChMax);
    INF("");
    for(j=0;j<n;j++)
    {
        printf("%ld\t", (j + 1));
        printf("%d\t", silGrp[kSilMax][j]);
        printf("%d\t", vrc2Grp[kVrc2Max][j]);
        printf("%d\t", chGrp[kChMax][j]);
        INF("");
    }

    WRN("----------------------------------------");

    // Free allocated distances matrix
    CLUSTER_FreeMatDistPointToPoint(n, &dist); 
}

void CLUSTER_computeWeightedKmeans3(data *dat, uint64_t n, uint64_t p, uint32_t kmin, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, eMethodType featureWeightsMethod, bool internalObjectWeights, const char *objectWeightsFile, eMethodType objectWeightsMethod)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax+1], statVRC2[kmax+1], statCH[kmax+1];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k

    // Initialize statistics
    for(k=kmax;k>=kmin;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = -1e20;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);

    // Initialize weights to 1.0
    CLUSTER_initObjectWeights(dat, n);
    if(internalFeatureWeights == false)
    {
        // Read object weights from file
    }

    if(internalObjectWeights == false)
    {
        // Read object weights from file
    }

    //WRN("Iteration %d", i);
    for(k=kmax;k>=kmin;k--) // From kMax to kMin
    {
        for(i=0;i<nbRep;i++) // Number of replicates
        {
            // Initialize weights
            if(internalObjectWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initObjectWeights(dat, n);
            }

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            double SSE = CLUSTER_weightedKmeans7(dat, n, p, k, c, internalFeatureWeights, featureWeightsMethod, internalObjectWeights, objectWeightsMethod, dist);
            
            // Compute silhouette statistic
            double sil = CLUSTER_computeSilhouette4(dat, n, p, c, k, dist);

            // Compute VRC statistic
            double vrc2 = CLUSTER_computeVRC2(dat, c, SSE, n, p, k);

            // Compute CH statistic
            double ch = CLUSTER_computeCH(TSS, SSE, n, k);

            bool clustNull = false;
            for(o=0;o<k;o++)
            {
                if(c[o].nbData == 0)
                {
                    clustNull = true;
                }    
            }

            // Save best VRC statistic for each k
            if((vrc2 > statVRC2[k] || i == 0) && clustNull == false)
            {
                statVRC2[k] = vrc2;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    vrc2Grp[k][j] = dat[j].clusterID;
                }
            }

            // Save best CH statistic for each k
            if((ch > statCH[k] || i == 0) && clustNull == false)
            {
                statCH[k] = ch;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    chGrp[k][j] = dat[j].clusterID;
                }
            }

            if((sil > statSil[k] || i == 0) && clustNull == false)
            {
                statSil[k] = sil;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    silGrp[k][j] = dat[j].clusterID;
                }
            }

            // Free clusters dimension memory
            CLUSTER_freeClusters(c, k);
        }
    }

    // Retrieve the overall best statistics
    double silMax = -1.0, vrc2Max = 0.0, chMax = 0.0;
    uint32_t kSilMax, kVrc2Max, kChMax;
    for(k=kmax;k>=kmin;k--)
    {
        if(statSil[k] > silMax)
        {
            silMax = statSil[k];
            kSilMax = k;
        }

        if(statVRC2[k] > vrc2Max)
        {
            vrc2Max = statVRC2[k];
            kVrc2Max = k;
        }

        if(statCH[k] > chMax)
        {
            chMax = statCH[k];
            kChMax = k;
        }
    }

    WRN("");
    WRN("Final statistics -----------------------");
    INF("Best silhouette : %lf for k = %d", silMax, kSilMax);
    INF("Best VRC2 : %lf for k = %d", vrc2Max, kVrc2Max);
    INF("Best CH : %lf for k = %d", chMax, kChMax);
    WRN("");
    WRN("Data membership for best indice scores -");
    printf("dataId\t");
    printf("Sil\t");
    printf("VRC2\t");
    printf("CH\t\n");
    printf("\t%d-Gr\t", kSilMax);
    printf("%d-Gr\t", kVrc2Max);
    printf("%d-Gr\t", kChMax);
    INF("");
    for(j=0;j<n;j++)
    {
        printf("%ld\t", (j + 1));
        printf("%d\t", silGrp[kSilMax][j]);
        printf("%d\t", vrc2Grp[kVrc2Max][j]);
        printf("%d\t", chGrp[kChMax][j]);
        INF("");
    }
    WRN("----------------------------------------");

    // Free allocated distances matrix
    CLUSTER_FreeMatDistPointToPoint(n, &dist); 
}

void CLUSTER_computeFeaturesWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmin, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, eMethodType featureWeightsMethod)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax+1], statVRC2[kmax+1], statCH[kmax+1];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k
    //double ow[n];

    // Initialize statistics
    for(k=kmax;k>=kmin;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = 0.0;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);

    // Initialize weights to 1.0
    if(internalFeatureWeights == false)
    {
        // Read object weights from file
    }

    //WRN("Iteration %d", i);
    for(k=kmax;k>=kmin;k--) // From kMax to kMin
    {
        for(i=0;i<nbRep;i++) // Number of replicates
        {
            cluster c[k];

            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            double SSE = CLUSTER_featuresWeightedKmeans(dat, n, p, k, c, internalFeatureWeights, featureWeightsMethod);

            // Compute silhouette statistic
            double sil = CLUSTER_computeSilhouette4(dat, n, p, c, k, dist);

            // Compute VRC statistic
            double vrc2 = CLUSTER_computeVRC2(dat, c, SSE, n, p, k);

            // Compute CH statistic
            double ch = CLUSTER_computeCH(TSS, SSE, n, k);

            // Check for null clusters
            bool clustNull = false;
            for(o=0;o<k;o++)
            {
                if(c[o].nbData == 0)
                {
                    clustNull = true;
                }    
            }

            // Save best VRC statistic for each k
            if((vrc2 > statVRC2[k] || i == 0) && clustNull == false)
            {
                statVRC2[k] = vrc2;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    vrc2Grp[k][j] = dat[j].clusterID;
                }
            }

            // Save best CH statistic for each k
            if((ch > statCH[k] || i == 0) && clustNull == false)
            {
                statCH[k] = ch;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    chGrp[k][j] = dat[j].clusterID;
                }
            }

            if((sil > statSil[k] || i == 0) && clustNull == false)
            {
                statSil[k] = sil;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    silGrp[k][j] = dat[j].clusterID;
                }
            }

            // Free clusters dimension memory
            CLUSTER_freeClusters(c, k);
        }
    }

    // Retrieve the overall best statistics
    double silMax = -1.0, vrc2Max = 0.0, chMax = 0.0;
    uint32_t kSilMax, kVrc2Max, kChMax;
    for(k=kmax;k>=kmin;k--)
    {
        if(statSil[k] > silMax)
        {
            silMax = statSil[k];
            kSilMax = k;
        }

        if(statVRC2[k] > vrc2Max)
        {
            vrc2Max = statVRC2[k];
            kVrc2Max = k;
        }

        if(statCH[k] > chMax)
        {
            chMax = statCH[k];
            kChMax = k;
        }
    }

    WRN("");
    WRN("Final statistics -----------------------");
    INF("Best silhouette : %lf for k = %d", silMax, kSilMax);
    INF("Best VRC2 : %lf for k = %d", vrc2Max, kVrc2Max);
    INF("Best CH : %lf for k = %d", chMax, kChMax);
    WRN("");
    WRN("Data membership for best indice scores -");
    printf("dataId\t");
    printf("Sil\t");
    printf("VRC2\t");
    printf("CH\t\n");
    printf("\t%d-Gr\t", kSilMax);
    printf("%d-Gr\t", kVrc2Max);
    printf("%d-Gr\t", kChMax);
    INF("");
    for(j=0;j<n;j++)
    {
        printf("%ld\t", (j + 1));
        printf("%d\t", silGrp[kSilMax][j]);
        printf("%d\t", vrc2Grp[kVrc2Max][j]);
        printf("%d\t", chGrp[kChMax][j]);
        INF("");
    }
    WRN("----------------------------------------");

    // Free allocated distances matrix
    CLUSTER_FreeMatDistPointToPoint(n, &dist); 
}

static double CLUSTER_kmeans4(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        bool conv = false; // Has converged

        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, c, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k, &conv);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        uint8_t iter = 0;
        conv = false;
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            CLUSTER_assignDataToCentroids11(dat, n, p, c, k, &conv); // MacQueen
            iter++;
        }

        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

static double CLUSTER_featuresWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("Bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        bool conv = false; // Has converged

        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, c, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k, &conv);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
        }

        uint8_t iter = 0;
        conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            CLUSTER_assignWeightedDataToCentroids(dat, n, p, c, k, &conv); 
            
            // Update centroids
            CLUSTER_computeCentroids(dat, n, p, c, k);

            // Update feature weights
            if(internalFeatureWeights == true)
            {
                CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
            }

            iter++;
        }

        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

static double CLUSTER_weightedKmeans7(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod, bool internalObjectWeights, eMethodType objectWeightsMethod, double **dist)
{
    if(!(dat == NULL || n < 2 || p < 1 || k < 2))
    {
        double wss[k];
        bool conv = false; // Has converged

        // Initialization
        //eMethodType objWeiMed = METHOD_MEDIAN;
        //eMethodType objWeiMed = METHOD_SILHOUETTE;
        //eMethodType objWeiMed = METHOD_MIN_DIST_CENTROID;
        //eMethodType objWeiMed = METHOD_SUM_DIST_CENTROID;
        //eMethodType feaWeiMed = METHOD_DISPERSION;
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, c, k);

        /*uint32_t l;
        uint64_t j;
        for(l=0;l<k;l++)
        {
            SAY("c%d nbData = %ld", l, c[l].nbData);
            for(j=0;j<p;j++)
            {
                SAY("fw = %lf", c[l].fw[j]);
            }
        }*/
        /*uint64_t i;
        for(i=0;i<n;i++)
        {
            SAY("dat%ld clusterID = %d", i, dat[i].clusterID);
        }*/

        /*uint64_t i;
        data *pt = (data *)c[0].head;
        for(i=0;i<c[0].nbData;i++)
        {
            SAY("dat clusterID = %d, 1rst dim = %lf", pt->clusterID, pt->dim[0]);
            pt = (data *) pt->succ;
        }*/

        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k, &conv);
        CLUSTER_computeCentroids(dat, n, p, c, k);
        CLUSTER_computeNkWeightedWSS(dat, n, p, c, k, wss);

        // Computes weights
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights3(dat, n, p, c, k, objectWeightsMethod, dist);
        }

        /*uint32_t l;
        double befSumWSS = 0.0;
        for(l=0;l<k;l++)
            befSumWSS += wss[l];
        //WRN("BEF = sumWSS = %lf", sumWSS);*/

        uint8_t iter = 0;
        conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            //CLUSTER_assignWeightedDataToCentroids23(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist, &conv); // Based on WSS and 1/nk
            //CLUSTER_assignWeightedDataToCentroids24(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist, &conv); // Based on SSE and 1/nk
            //CLUSTER_assignWeightedDataToCentroids25(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist, &conv); // Based on SSE
            CLUSTER_assignWeightedDataToCentroids27(dat, n, p, c, k, internalFeatureWeights, featureWeightsMethod, internalObjectWeights, objectWeightsMethod, dist, wss, &conv); // Based on sum of WSS and 1/nk

            iter++;
        }

        /*double aftSumWSS = 0.0;
        for(l=0;l<k;l++)
            aftSumWSS += wss[l];
        //WRN("AFT = sumWSS = %lf", aftSumWSS);
        WRN("ratio BEF/AFT WSS = %lf (BEF = %lf, AFT = %lf)", (befSumWSS/aftSumWSS), befSumWSS, aftSumWSS);*/

        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
    else
    
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
}

static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, cluster *c, uint32_t k)
{
    uint64_t i;
    uint32_t l = 0;
    for(i=0;i<n;i++)
    {
        // Assign fake cluster to avoid null cluster
        if(l == k)
            l = 0;

        //dat[i].clusterID = l;
        //c[dat[i].clusterID].nbData++;
        CLUSTER_addPointToCluster(&(dat[i]), &(c[l]));
        l++;
    }
}

static void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k)
{
    if(p < 1 || c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        uint32_t l;
        uint64_t j;
        for(l=0;l<k;l++)
        {
            c[l].ind = l; // Init index
            c[l].centroid = malloc(p*sizeof(double)); // Allocate cluster dimension memory
            if(c[l].centroid == NULL)
            {
                ERR("Cluster dimension memory allocation failed");
            }
            else
            {
                c[l].nbData = 0;
                c[l].head = NULL;

                c[l].fw = malloc(p*sizeof(double)); // Allocate cluster features weights memory
                if(c[l].fw == NULL)
                {
                    ERR("Cluster features weights memory allocation failed");
                }
                else
                {
                    // Init features weights to 1.0
                    for(j=0;j<p;j++)
                    {
                        c[l].fw[j] = 1.0;
                    }
                }
            }
        }
    }
}

static void CLUSTER_freeClusters(cluster *c, uint32_t k)
{
    if(c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        uint32_t l;
        for(l=0;l<k;l++)
        {
            free(c[l].centroid); // Free cluster dimension memory
            free(c[l].fw); // Free cluster features weights memory
        }
    }
}

static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint32_t l;
    uint64_t j;

    //WRN("-------------------------------");
    uint64_t rnd[k];
    for(l=0;l<k;l++)
    {
        // Avoid duplicate
        uint64_t randInd;
        bool ok = false;
        while(ok == false)
        {
            // Use random data as centroids
            randInd = rand() % n;
            ok = true;
            rnd[l] = randInd;
            uint8_t i;
            for(i=0;i<l;i++)
            {
                if(randInd == rnd[i])
                {
                    ok = false;
                }
            }
        }
        //SAY("rand = %ld", randInd);
        for(j=0;j<p;j++)
            c[l].centroid[j] = dat[randInd].dim[j];
    }
    //WRN("-------------------------------");
}

static double CLUSTER_assignDataToCentroids7(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    uint64_t i;
    uint32_t l;
    double SSE = 0.0;

    // Set convergence variable
    *conv = true;

    for(i=0;i<n;i++)
    {
        double minDist;
        uint32_t minK;
        //INF("Dist for dat%ld", i);
        for(l=0;l<k;l++)
        {
            // Calculate squared Euclidean distance
            double dist = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

            //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);
            if(l == 0)
            {
                minDist = dist;
                minK = l;
            }
            else
            {
                if(dist < minDist)
                {
                    minDist = dist;
                    minK = l; // Save the cluster for the min distance
                }
            }
        }

        /*c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
          dat[i].clusterID = minK; // Assign data to cluster
          c[dat[i].clusterID].nbData++; // Increase new cluster data number*/

        if(minK != dat[i].clusterID)
        {
            // Remove point from former cluster
            CLUSTER_removePointFromCluster(&(dat[i]), &(c[dat[i].clusterID]));
            // Add point to cluster minK
            CLUSTER_addPointToCluster(&(dat[i]), &(c[minK]));

            // Reset convergence variable
            *conv = false;
        }

        SSE += minDist;

        //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
    }

    return SSE;
}

static void CLUSTER_assignDataToCentroids11(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        // MacQueen algorithm //
        
        uint64_t i;
        uint32_t l;

        // Set convergence variable
        *conv = true;

        for(i=0;i<n;i++)
        {
            double minDist;
            uint32_t minK;

            // For each cluster
            for(l=0;l<k;l++)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

                if(l == 0)
                {
                    minDist = dist;
                    minK = l;
                }
                else
                {
                    if(dist < minDist)
                    {
                        minDist = dist;
                        minK = l; // Save the cluster for the min distance
                    }
                }
            }

            if(minK != dat[i].clusterID)
            {
                // Transfer point i to cluster minK
                CLUSTER_transferPointToCluster2(dat, i, p, c, minK);

                // Reset convergence variable
                *conv = false;
            }
        }
    }
}

static void CLUSTER_assignWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    uint64_t i;
    uint32_t l;

    // Set convergence variable
    *conv = true;

    for(i=0;i<n;i++)
    {
        double minDist;
        uint32_t minK;
        //INF("Dist for dat%ld", i);
        for(l=0;l<k;l++)
        {
            // Calculate squared Euclidean distance
            double dist = CLUSTER_computeSquaredFWDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

            //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);
            if(l == 0)
            {
                minDist = dist;
                minK = l;
            }
            else
            {
                if(dist < minDist)
                {
                    minDist = dist;
                    minK = l; // Save the cluster for the min distance
                }
            }
        }

        /*c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
          dat[i].clusterID = minK; // Assign data to cluster
          c[dat[i].clusterID].nbData++; // Increase new cluster data number*/

        if(minK != dat[i].clusterID)
        {
            // Remove point from former cluster
            CLUSTER_removePointFromCluster(&(dat[i]), &(c[dat[i].clusterID]));
            // Add point to cluster minK
            CLUSTER_addPointToCluster(&(dat[i]), &(c[minK]));

            // Reset convergence variable
            *conv = false;
        }

        //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
    }
}

static void CLUSTER_assignWeightedDataToCentroids27(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, eMethodType feaWeiMet, bool internalObjectWeights, eMethodType objWeiMet, double **dist, double wss[k], bool *conv)
{
    uint64_t i;
    uint32_t l;

    // Set convergence variable
    *conv = true;

    /*double sumWei[k];
      for(l=0;l<k;l++)
      {
      sumWei[l] = 0.0;
      }

      for(i=0;i<n;i++)
      {
      sumWei[dat[i].clusterID] += dat[i].ow;
      }
      for(l=0;l<k;l++)
      {
      SAY("WSS[%d] = %lf", l, wss[l]);
      INF("SumWei[%d] = %lf (nbData = %ld)", l, sumWei[l], c[l].nbData);
      }*/

    for(i=0;i<n;i++)
    {
        //WRN("For dat%ld from clust%d (nbData %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        // Save current cluster of point i
        uint32_t curClust = dat[i].clusterID;

        // Compute WSS of datum i cluster without datum i
        uint32_t tmpClust = (curClust + 1 >= k) ? 0 : curClust + 1; // Define a tmp cluster
        //SAY("tmpClust = %d", tmpClust);

        CLUSTER_transferPointToCluster2(dat, i, p, c, tmpClust); // Transfer datum i to tmp cluster

        // Update objects weights in former datum i cluster 
        if(internalObjectWeights == true)
        {
            // Internal computation of object weights
            CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, curClust, objWeiMet, dist);
        }

        //  Update features weights in former datum i cluster
        if(internalFeatureWeights == true)
        {
            // Internal computation of feature weights
            CLUSTER_computeFeatureWeightsInCluster(dat, n, p, c, k, curClust, feaWeiMet);
        }

        /*double sum = 0.0;
          for(j=0;j<n;j++)
          {
          if(dat[j].clusterID == curClust)
          {
          printf("ow[%ld] = %lf ", j, ow[j]);
          sum +=  ow[j];
          }
          }
          printf("ow[%ld] = %lf ", i, ow[i]);
          sum +=  ow[i];
          SAY("");
          SAY("weiSum = %lf (nbData = %ld)", sum, c[curClust].nbData + 1);

          sum = 0.0;
          for(j=0;j<n;j++)
          {
          if(dat[j].clusterID == curClust)
          {
          printf("ow[%ld] = %lf ", j, ow_fromTmp[j]);
          sum +=  ow_fromTmp[j];
          }
          }
          SAY("");
          SAY("weiSum = %lf (nbData = %ld)", sum, c[curClust].nbData);*/

        /*data *pti = (data *)c[curClust].head;
        uint64_t j; 
        double sumWei = 0.0;
        for(j=0;j<c[curClust].nbData;j++)
        {
            sumWei += pti->ow;
            pti = pti->succ;
        }
        SAY("sumWei[%d] = %lf (nbData = %ld)", curClust, sumWei, c[curClust].nbData);*/

        //double tmpFromWss = CLUSTER_computeNkWeightedWSSInCluster(dat, n, p, c, k, curClust, fw, ow_fromTmp); // Compute from WSS
        double tmpFromWss = CLUSTER_computeNkWeightedWSSInCluster(dat, n, p, c, k, curClust); // Compute from WSS
        //WRN("WSS%d = %lf (without dat%ld)", curClust, tmpFromWss, i);
        //WRN("WSS%d = %lf (with dat%ld, ow[%ld] = %lf)", curClust, wss[curClust], i, i, owTmp);

        double minSumWss = 1e20;
        uint32_t minK;
        double tmpToWss[k];
        bool improved = false;

        for(l=0;l<k;l++)
        {
            if(l != curClust)
            {
                double sumWssRef = wss[curClust] + wss[l];

                // Compute WSS of cluster l with datum i
                CLUSTER_transferPointToCluster2(dat, i, p, c, l); // Transfer datum i to tmp cluster
                // Update objects weights in former datum i cluster 
                if(internalObjectWeights == true)
                {
                    // Internal computation of object weights
                    //CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, &(ow_toTmp[l]), objWeiMet, dist);
                    CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, objWeiMet, dist);
                }

                // Update feature weights in cluster l
                if(internalFeatureWeights == true)
                {
                    // Internal computation of feature weights
                    CLUSTER_computeFeatureWeightsInCluster(dat, n, p, c, k, l, feaWeiMet);
                }

                /*data *pti = (data *)c[l].head;
                uint64_t j; 
                double sumWei = 0.0;
                for(j=0;j<c[l].nbData;j++)
                {
                    sumWei += pti->ow;
                    pti = pti->succ;
                }
                SAY("sumWei[%d] = %lf (nbData = %ld)", l, sumWei, c[l].nbData);*/

                //tmpToWss[l] = CLUSTER_computeNkWeightedWSSInCluster(dat, n, p, c, k, l, fw, &(ow_toTmp[l])); // Compute to WSS
                tmpToWss[l] = CLUSTER_computeNkWeightedWSSInCluster(dat, n, p, c, k, l); // Compute to WSS

                double newWss = tmpFromWss + tmpToWss[l];

                // Test if the deplacement from curClust to l improves the sum of WSS
                //INF("WSS%d = %lf (with dat%ld, ow[%ld] = %lf)", l, tmpToWss[l], i, i, dat[i].ow);
                //INF("WSS%d = %lf (without dat%ld)", l, wss[l], i);
                //SAY("sumWssRef = %lf, newWss = %lf", sumWssRef, newWss);
                if(newWss < sumWssRef)
                {
                    improved = true;

                    //INF("WSS improved!!!!!!!!!!!!");
                    // Test if the new sum of WSS is minimal
                    if(newWss < minSumWss)
                    {
                        //INF("WSS better");
                        //SAY("sumWssRef = %lf, newWss = %lf, minSumWss = %lf, minK = %d", sumWssRef, newWss, minSumWss, l);
                        minSumWss = newWss;
                        minK = l;
                    }

                    // Reset convergence variable
                    *conv = false;
                }
            }
        }

        // Test if WSS improved
        if(improved == true)
        {
            CLUSTER_transferPointToCluster2(dat, i, p, c, minK); // Transfer datum i to minK cluster
            wss[curClust] = tmpFromWss;
            wss[minK] = tmpToWss[minK];
        }
        else
        {
            // Transfer datum i to initial cluster
            CLUSTER_transferPointToCluster2(dat, i, p, c, curClust); 
        }

        // Update feature weights
        /*if(internalFeatureWeights == true)
          {
        // Internal computation of feature weights
        CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
        }

        // Update object weights
        if(internalObjectWeights == true)
        {
        CLUSTER_computeObjectWeights3(dat, n, p, c, k, objWeiMet, dist);
        }*/
    }
}

static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d)
{
    double dist = 0;
    switch(d)
    {
        default:
        case DISTANCE_EUCLIDEAN:
            {
                uint64_t j;
                for(j=0;j<p;j++)
                {
                    //double tmp = pow((dat->dim[j] - c->centroid[j]), 2.0);
                    /*double toto = (dat->dim[j] - c->centroid[j]);
                      double tmp = toto * toto;*/

                    double tmp = (dat->dim[j] - c->centroid[j]);
                    tmp *= tmp;

                    if(isnan(tmp))
                    {
                        dist += 0.0;
                    }
                    else
                    {
                        dist += tmp;
                    }
                }
            }
            break;
        case DISTANCE_OTHER:
            {
                WRN("Not implemented yet");
                dist = -1;
            }
            break;
    }

    return dist;
}

static double CLUSTER_computeSquaredFWDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d)
{
    double dist = 0;
    switch(d)
    {
        default:
        case DISTANCE_EUCLIDEAN:
            {
                uint64_t j;
                for(j=0;j<p;j++)
                {
                    double tmp = (dat->dim[j] - c->centroid[j]);
                    tmp *= tmp;
                    tmp *= c->fw[j];

                    if(isnan(tmp))
                    {
                        dist += 0.0;
                    }
                    else
                    {
                        dist += tmp;
                    }
                }
            }
            break;
        case DISTANCE_OTHER:
            {
                WRN("Not implemented yet");
                dist = -1;
            }
            break;
    }

    return dist;
}

static double CLUSTER_computeSquaredDistanceClusterToCluster(cluster *ci, cluster *cj, uint64_t p, eDistanceType d)
{
    double dist = 0;
    switch(d)
    {
        default:
        case DISTANCE_EUCLIDEAN:
            {
                uint64_t j;
                for(j=0;j<p;j++)
                {
                    double tmp = (ci->centroid[j] - cj->centroid[j]);
                    tmp *= tmp;

                    if(isnan(tmp))
                    {
                        dist += 0.0;
                    }
                    else
                    {
                        dist += tmp;
                    }
                }
            }
            break;
        case DISTANCE_OTHER:
            {
                WRN("Not implemented yet");
                dist = -1;
            }
            break;
    }

    return dist;
}

static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow)
{
    if(!(dat == NULL || p < 1 || c == NULL || fw == NULL))
    {
        double dist = 0.0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                    {
                        double tmp = fw[j]*pow((dat->dim[j] - c->centroid[j]), 2.0); // Apply feature weights

                        if(isnan(tmp))
                        {
                            dist += 0.0;
                        }
                        else
                        {
                            dist += tmp;
                        }
                    }
                }
                break;
            case DISTANCE_OTHER:
                {
                    WRN("Not implemented yet");
                    dist = -1;
                }
                break;
        }

        return (ow*dist); // Apply object weight
    }
    else
    {
        ERR("Bad parameter");
        return -1.0;
    }

}

static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t i,j;
    uint32_t l;

    // Reset each centroid dimension to 0
    for(l=0;l<k;l++)
        for(j=0;j<p;j++)
            c[l].centroid[j] = 0.0;

    // Compute the new centroid dimension
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            c[dat[i].clusterID].centroid[j] += (dat[i].dim[j]/(double)c[dat[i].clusterID].nbData);   
        }
    }    
}

static void CLUSTER_transferPointToCluster2(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK)
{
    uint64_t j;

    uint32_t prevClust = dat[indN].clusterID; // Retreive datum previous cluster

    // Remove point from former cluster
    CLUSTER_removePointFromCluster(&(dat[indN]), &(c[prevClust]));
    // Add point to cluster indK
    CLUSTER_addPointToCluster(&(dat[indN]), &(c[indK]));

    // Compute the new centroid dimensions
    for(j=0;j<p;j++)
    {
        if(c[prevClust].nbData == 0)
        {
            c[prevClust].centroid[j] = 0.0;
        }
        else
        {
            c[prevClust].centroid[j] = ((c[prevClust].centroid[j] * (double)(c[prevClust].nbData + 1)) - dat[indN].dim[j])/(double)c[prevClust].nbData;   
        }
        c[indK].centroid[j] = ((c[indK].centroid[j] * (double)(c[indK].nbData - 1)) + dat[indN].dim[j])/(double)c[indK].nbData;
    }
}

static void CLUSTER_addPointToCluster(data *dat, cluster *c)
{
    // Test if cluster is empty
    if(c->head == NULL)
    {
        dat->succ = NULL;
    }
    else
    {
        dat->succ = c->head;
        ((data *)c->head)->pred = dat;
    }

    dat->pred = NULL;
    c->head = dat;

    // Update data membership
    dat->clusterID = c->ind; 

    // Update number of data in the cluster
    c->nbData++;
}

static void CLUSTER_removePointFromCluster(data *dat, cluster *c)
{
    // Test if cluster is empty
    if(c->head == NULL)
    {
        ERR("Can't remove point, cluster empty");
    }
    else
    {
        // Test if dat is the first element of the cluster chain list
        if(dat->pred == NULL)
        {
            c->head = dat->succ;

            //Test if there is only one element in the cluster chain list
            if(dat->succ != NULL)
            {
                ((data *)dat->succ)->pred = NULL;
            }
        }
        // Test if dat is the last element of the cluster chain list
        else if(dat->succ == NULL)
        {
            ((data *)dat->pred)->succ = dat->succ;
        }
        // Test if dat is somewhere in the cluster chain list
        else
        {
            ((data *)dat->pred)->succ = dat->succ;
            ((data *)dat->succ)->pred = dat->pred;
        }

        // Reset datum pred & succ 
        dat->pred = NULL;
        dat->succ = NULL; 

        // Update data membership
        dat->clusterID = -1;

        // Update number of data in the cluster
        c->nbData--;
    }
}

static double CLUSTER_computeSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -2.0;
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k];
        uint64_t i,j;
        uint32_t l;

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                //WRN("dist[%ld][%ld] = %lf", i, j, dist[i][j]);
                if((j != i) && (dat[j].clusterID == dat[i].clusterID))
                {
                    d += dist[i][j];
                }
            }

            if((c[dat[i].clusterID].nbData - 1) == 0)
            {
                a[i] = 0.0;
            }
            else
            {
                a[i] = d / (double)(c[dat[i].clusterID].nbData - 1);
            }

            //SAY("a[%ld] = %lf", i, a[i]);

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
            {
                distCluster[l] = 0.0;
            }

            for(j=0;j<n;j++)
            {
                if(dat[j].clusterID != dat[i].clusterID)
                {
                    distCluster[dat[j].clusterID] += (dist[i][j]/(double)c[dat[j].clusterID].nbData);
                }
            }

            /*for(l=0;l<k;l++)
            {
                SAY("disCluster[%d] = %lf", l, distCluster[l]);
            }*/

            b[i] = 1.0e20;
            for(l=0;l<k;l++)
            {
                if((l != dat[i].clusterID) && (distCluster[l] != 0) && (distCluster[l] < b[i]))
                {
                    b[i] = distCluster[l];
                }
            }

            //SAY("b[%ld] = %lf", i, b[i]);

            // Calculate s[i]
            if(c[dat[i].clusterID].nbData == 1)
            {
                s[i] = 0;
            }
            else
            {
                s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
            }
               
            //SAY("i = %ld, s[i] = %lf, b[i] = %lf, a[i] = %lf, dat[i].clusterID = %d, c[dat[i].clusterID].nbData = %ld", i, s[i], b[i], a[i], dat[i].clusterID, c[dat[i].clusterID].nbData);
            //SAY("s[%d] = %lf", i, s[i]);
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        for(l=0;l<k;l++)
        {
            //WRN("sk[%d] = %lf, nbdata = %ld", l, sk[l], c[l].nbData);
            if(!isnan(sk[l]))
                sil += sk[l];
        }

        return (sil/k);
    }
}

static double CLUSTER_computeDistancePointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d)
{
    if(iDat == NULL || jDat == NULL || p < 1)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double dist = 0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                        dist += pow((iDat->dim[j] - jDat->dim[j]), 2.0);
                }
                break;
            case DISTANCE_OTHER:
                {
                    WRN("Not implemented yet");
                    dist = -1;
                }
                break;
        }

        return sqrt(dist);
    }
}

static double CLUSTER_computeTSS(data *dat, uint64_t n, uint64_t p)
{
    if(dat == NULL || n < 2 || p < 1)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        uint64_t i, j;
        double mean[p];
        double TSS = 0.0;

        for(j=0;j<p;j++)
            mean[j] = 0.0;

        for(i=0;i<n;i++)
            for(j=0;j<p;j++)
                mean[j] += (dat[i].dim[j]/(double)n);

        for(i=0;i<n;i++)
            for(j=0;j<p;j++)
                TSS += pow(dat[i].dim[j] - mean[j], 2.0);

        return TSS;
    }
}

static double CLUSTER_computeVRC2(data *dat, cluster *c, double SSE, uint64_t n, uint64_t p, uint32_t k)
{
    uint64_t i, j;
    uint32_t l;
    double mean[p];
    for(j=0;j<p;j++)
    {
        mean[j] = 0.0;
        for(i=0;i<n;i++)
        {
            mean[j] += (double) dat[i].dim[j] / (double) n;
        }
    }

    double SSB = 0.0;

    for(l=0;l<k;l++)
    {
        for(j=0;j<p;j++)
        {
            SSB += (double) c[l].nbData * pow((c[l].centroid[j] - mean[j]), 2.0); 
        }
    }

    return (SSB / (k - 1)) / (SSE / (n - k)); 
}

static double CLUSTER_computeCH(double TSS, double SSE, uint64_t n, uint32_t k)
{
    double tmp = SSE / (double)(n - k);
    return ( (TSS - SSE) / (double)(k - 1)) / tmp; 
}

static void CLUSTER_initObjectWeights(data *dat, uint64_t n)
{
    uint64_t i;
    for(i=0;i<n;i++)
        dat[i].ow = 1.0;
}

static void CLUSTER_computeFeatureWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m)
{
    switch(m)
    {
        default:
        case METHOD_DISPERSION :
            {
                CLUSTER_computeFeatureWeightsViaDispersion(dat, n, p, c, k, 2); // Using L2-norm
            }
            break;
        case METHOD_OTHER:
            {
                WRN("Not implemented yet");
            }
            break;
    }
}

static void CLUSTER_computeFeatureWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m)
{
    switch(m)
    {
        default:
        case METHOD_DISPERSION :
            {
                CLUSTER_computeFeatureWeightsInClusterViaDispersion(dat, n, p, c, k, indK, 2); // Using L2-norm
            }
            break;
        case METHOD_OTHER:
            {
                WRN("Not implemented yet");
            }
            break;
    }
}

static void CLUSTER_computeFeatureWeightsInClusterViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, uint8_t norm)
{
    uint64_t i, j, m;
    double disp[p]; // Dispersion per feature
    uint64_t nbdataClust = c[indK].nbData;

    // Initialization
    for(j=0;j<p;j++)
    {
        disp[j] = 0.0;
    }

    // Compute dispersion
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        for(j=0;j<p;j++)
        {
            disp[j] += CLUSTER_computeFeatureDispersion(pti, j, &(c[indK]), norm);
        }

        // Update pti
        pti = (data *)pti->succ;
    }

    // Compute weights
    double tmp[p];
    for(j=0;j<p;j++)
    {
        tmp[j] = 0.0;
    }

    for(j=0;j<p;j++)
    {
        for(m=0;m<p;m++)
        {
            tmp[j] += pow((disp[j] / disp[m]),(1 / (norm - 1)));
            if(isnan(tmp[j]))
                tmp[j] = 0.0; 
        }

        if(c[indK].nbData == 1)
        {
            //fw[indK][j] = 1.0;
            c[indK].fw[j] = 1.0;
        }
        else
        {
            // The sum of features weights as to be equal to unity 
            //fw[indK][j] = pow((1 / tmp[j]), norm);
            c[indK].fw[j] = pow((1 / tmp[j]), norm);
            if(isinf(c[indK].fw[j]))
                c[indK].fw[j] = 1.0; 
            //SAY("Weight[%d][%ld] = %lf", l, j, fw[l][j]);
        }
    }
}

static void CLUSTER_computeFeatureWeightsViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint8_t norm)
{
    uint64_t i, j;
    uint32_t l,m;
    double disp[k][p]; // Dispersion per cluster and per feature

    // Initialization
    for(l=0;l<k;l++)
    {
        for(j=0;j<p;j++)
        {
            disp[l][j] = 0.0;
        }
    }

    // Compute dispersion
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            disp[dat[i].clusterID][j] += CLUSTER_computeFeatureDispersion(&(dat[i]), j, &(c[dat[i].clusterID]), norm);
        }
    }

    // Compute weights
    for(l=0;l<k;l++)
    {
        double tmp[p];
        for(j=0;j<p;j++)
        {
            tmp[j] = 0.0;
        }

        for(j=0;j<p;j++)
        {
            for(m=0;m<p;m++)
            {
                tmp[j] += pow((disp[l][j] / disp[l][m]),(1 / (norm - 1)));
                if(isnan(tmp[j]))
                    tmp[j] = 0.0; 
            }

            if(c[l].nbData == 1)
            {
                //fw[l][j] = 1.0;
                c[l].fw[j] = 1.0;
            }
            else
            {
                // The sum of features weights as to be equal to unity 
                //fw[l][j] = pow((1 / tmp[j]), norm);
                c[l].fw[j] = pow((1 / tmp[j]), norm);
                /*if(isinf(fw[l][j]))
                    fw[l][j] = 1.0;*/
                if(isinf(c[l].fw[j]))
                    c[l].fw[j] = 1.0;
                //SAY("Weight[%d][%ld] = %lf", l, j, fw[l][j]);
            }
        }
    }
}

static double CLUSTER_computeFeatureDispersion(data *dat, uint64_t p, cluster *c, uint8_t norm)
{
    return pow(dat->dim[p] - c->centroid[p], norm); 
}

static void CLUSTER_computeObjectWeights3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m, double **dist)
{
    switch(m)
    {
        case METHOD_SILHOUETTE :
            {
                CLUSTER_computeObjectWeightsViaSilhouette4(dat, n, p, c, k, dist);
            }
            break;
        case METHOD_SILHOUETTE_NK :
            {
                CLUSTER_computeObjectWeightsViaSilhouetteNK(dat, n, p, c, k, dist);
            }
            break;
        case METHOD_MEDIAN :
            {
                CLUSTER_computeObjectWeightsViaMedian2(dat, n, p, c, k);
            }
            break;
        case METHOD_MEDIAN_NK :
            {
                CLUSTER_computeObjectWeightsViaMedianNK(dat, n, p, c, k);
            }
            break;
        case METHOD_MIN_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsViaMinDistCentroid(dat, n, p, c, k);
            }
            break;
        case METHOD_MIN_DIST_CENTROID_NK :
            {
                CLUSTER_computeObjectWeightsViaMinDistCentroidNK(dat, n, p, c, k);
            }
            break;
        case METHOD_SUM_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsViaSumDistCentroid(dat, n, p, c, k);
            }
            break;
        default:
            {
                WRN("Not implemented yet");
            }
            break;
    }
}

static void CLUSTER_computeObjectWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m, double **dist)
{
    switch(m)
    {
        case METHOD_SILHOUETTE :
            {
                CLUSTER_computeObjectWeightsInClusterViaSilhouette(dat, n, p, c, k, indK, dist);
            }
            break;
        case METHOD_SILHOUETTE_NK :
            {
                CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(dat, n, p, c, k, indK, dist);
            }
            break;
        case METHOD_MEDIAN :
            {
                CLUSTER_computeObjectWeightsInClusterViaMedian(dat, n, p, c, indK);
            }
            break;
        case METHOD_MEDIAN_NK :
            {
                CLUSTER_computeObjectWeightsInClusterViaMedianNK(dat, n, p, c, indK);
            }
            break;
        case METHOD_MIN_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(dat, n, p, c, k, indK);
            }
            break;
        case METHOD_MIN_DIST_CENTROID_NK :
            {
                CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(dat, n, p, c, k, indK);
            }
            break;
        case METHOD_SUM_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(dat, n, p, c, k, indK);
            }
            break;
        default:
            {
                WRN("Not implemented yet");
            }
            break;
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaSilhouette(dat, n, p, c, k, l, dist);
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(dat, n, p, c, k, l, dist);
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist)
{
    uint64_t nbdataClust = c[indK].nbData;
    double a[nbdataClust], b[nbdataClust], s[nbdataClust], sk = 0.0, distCluster[k];
    uint64_t i,j;
    uint32_t l;

    data *pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
        double d = 0.0;
        data *ptj = (data *)c[indK].head;
        for(j=0;j<nbdataClust;j++)
        {
            if(ptj->ind != pti->ind)
            {
                d += dist[pti->ind][ptj->ind];
            }

            // Update ptj
            ptj = (data *)ptj->succ;
        }

        if((c[indK].nbData - 1) == 0)
        {
            a[i] = 0.0;
        }
        else
        {
            a[i] = d / (double)(c[indK].nbData - 1);
        }

        // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
        for(l=0;l<k;l++)
            distCluster[l] = 0.0;

        for(j=0;j<n;j++)
            if(dat[j].clusterID != indK)
                distCluster[dat[j].clusterID] += (dist[pti->ind][j]/c[dat[j].clusterID].nbData);
        b[i] = 1.0e20;
        for(l=0;l<k;l++)
            if(l != indK && distCluster[l] != 0 && distCluster[l] < b[i])
                b[i] = distCluster[l];

        // Calculate s[i]
        if(c[indK].nbData == 1)
        {
            s[i] = 0;
        }
        else
        {
            s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
        }

        //SAY("s[%ld] = %lf", i, s[i]);

        if(s[i] < 0 || s[i] > 0)
            s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
        else // si = 0
            s[i] = 0.5;

        // Calculate sum of s[i] per cluster 
        sk += s[i];

        //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
        pti->ow = s[i];

        if(isnan(pti->ow) || isinf(pti->ow))
        {
            //ERR("ow = %lf, s = %lf, sk = %lf, cNbDat = %ld", ow[i], s[i], sk[dat[i].clusterID], c[dat[i].clusterID].nbData);
            pti->ow = 1.0;
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist)
{
    uint64_t nbdataClust = c[indK].nbData;
    double a[nbdataClust], b[nbdataClust], s[nbdataClust], sk = 0.0, distCluster[k];
    uint64_t i,j;
    uint32_t l;

    data *pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
        double d = 0.0;
        data *ptj = (data *)c[indK].head;
        for(j=0;j<nbdataClust;j++)
        {
            if(ptj->ind != pti->ind)
            {
                d += dist[pti->ind][ptj->ind];
            }

            // Update ptj
            ptj = (data *)ptj->succ;
        }

        if((c[indK].nbData - 1) == 0)
        {
            a[i] = 0.0;
        }
        else
        {
            a[i] = d / (double)(c[indK].nbData - 1);
        }

        // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
        for(l=0;l<k;l++)
            distCluster[l] = 0.0;

        for(j=0;j<n;j++)
            if(dat[j].clusterID != indK)
                distCluster[dat[j].clusterID] += (dist[pti->ind][j]/c[dat[j].clusterID].nbData);
        b[i] = 1.0e20;
        for(l=0;l<k;l++)
            if(l != indK && distCluster[l] != 0 && distCluster[l] < b[i])
                b[i] = distCluster[l];

        // Calculate s[i]
        if(c[indK].nbData == 1)
        {
            s[i] = 0;
        }
        else
        {
            s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
        }

        //SAY("s[%ld] = %lf", i, s[i]);

        if(s[i] < 0 || s[i] > 0)
            s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
        else // si = 0
            s[i] = 0.5;

        // Calculate sum of s[i] per cluster 
        sk += s[i];

        //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
        // Update pti
        pti = (data *)pti->succ;
    }

    // Calculate object weights
    pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        pti->ow = (s[i] / sk) * (double) c[indK].nbData;

        if(isnan(pti->ow) || isinf(pti->ow))
        {
            //ERR("ow = %lf, s = %lf, sk = %lf, cNbDat = %ld", ow[i], s[i], sk[dat[i].clusterID], c[dat[i].clusterID].nbData);
            pti->ow = 1.0;
        }
        //}
        //SAY("ow[%ld] = %lf", i, ow[i]);
        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsViaMedian2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMedian(dat, n, p, c, l);
    }
}

static void CLUSTER_computeObjectWeightsViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMedianNK(dat, n, p, c, l);
    }
}

static int cmpfunc(const void * a, const void * b)
{
    return (*(double*)a > *(double*)b) ? 1 : (*(double*)a < *(double*)b) ? -1:0 ;
}

static void CLUSTER_computeObjectWeightsInClusterViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK)
{
    uint64_t i, j;
    uint64_t nbDataClust = c[indK].nbData;
    double median[p]; // Median per cluster
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    for(j=0;j<p;j++)
    {
        double dim[nbDataClust];
        data *pti = (data *)c[indK].head;
        for(i=0;i<nbDataClust;i++)
        {
            dim[i] = pti->dim[j];
            //WRN("dim[%ld][%ld] = %lf",j , i, dim[i]);

            // Update pti
            pti = (data *)pti->succ;
        }

        // Sort dimension value in ascending way
        qsort(dim, nbDataClust, sizeof(double), cmpfunc);

        /*for(i=0;i<nbDataClust;i++)
        {
            INF("dim[%ld][%ld] = %lf",j , i, dim[i]);
        }*/

        // Compute the median
        if(!(nbDataClust % 2))
        {
            median[j] = (dim[(nbDataClust / 2)] + dim[(nbDataClust / 2) - 1]) / 2;
        }
        else
        {
            median[j] = dim[((nbDataClust + 1) / 2) - 1];
        }
        //ERR("median[%ld] = %lf", j, median[j]);
    }

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        // Initialize tmp weights
        w[i] = 0.0;

        for(j=0;j<p;j++)
        {
            //SAY("pti[%ld]->dim[%ld] = %lf, median[%ld] = %lf, abs = %lf", pti->ind, j, pti->dim[j], j, median[j], fabs((pti->dim[j] - median[j])));
            w[i] += fabs(pti->dim[j] - median[j]); 
        }
        //SAY("w[%ld] = %lf", pti->ind, w[i]);

        sumWeights += w[i];

        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = w[i];
        }

        //SAY("w[%ld] = %lf", i, w[i]);

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK)
{
    uint64_t i, j;
    uint64_t nbDataClust = c[indK].nbData;
    double median[p]; // Median per cluster
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    for(j=0;j<p;j++)
    {
        double dim[nbDataClust];
        data *pti = (data *)c[indK].head;
        for(i=0;i<nbDataClust;i++)
        {
            dim[i] = pti->dim[j];
            //WRN("dim[%ld][%ld] = %lf",j , i, dim[i]);

            // Update pti
            pti = (data *)pti->succ;
        }

        // Sort dimension value in ascending way
        qsort(dim, nbDataClust, sizeof(double), cmpfunc);

        /*for(i=0;i<nbDataClust;i++)
        {
            INF("dim[%ld][%ld] = %lf",j , i, dim[i]);
        }*/

        // Compute the median
        if(!(nbDataClust % 2))
        {
            median[j] = (dim[(nbDataClust / 2)] + dim[(nbDataClust / 2) - 1]) / 2;
        }
        else
        {
            median[j] = dim[((nbDataClust + 1) / 2) - 1];
        }
        //ERR("median[%ld] = %lf", j, median[j]);
    }

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        // Initialize tmp weights
        w[i] = 0.0;

        for(j=0;j<p;j++)
        {
            //SAY("pti[%ld]->dim[%ld] = %lf, median[%ld] = %lf, abs = %lf", pti->ind, j, pti->dim[j], j, median[j], fabs((pti->dim[j] - median[j])));
            w[i] += fabs(pti->dim[j] - median[j]); 
        }
        //SAY("w[%ld] = %lf", pti->ind, w[i]);

        sumWeights += w[i];

        //SAY("w[%ld] = %lf", i, w[i]);

        // Update pti
        pti = (data *)pti->succ;
    }

    // Compute objects weights
    pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = (w[i] / sumWeights) * (double) nbDataClust;
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(dat, n, p, c, k, l);
    }
}

static void CLUSTER_computeObjectWeightsViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(dat, n, p, c, k, l);
    }
}

static void CLUSTER_computeObjectWeightsViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(dat, n, p, c, k, l);
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    // Based on the nearest centroid different of the point one

    uint64_t i;
    uint64_t l;
    uint64_t nbDataClust = c[indK].nbData;
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        //WRN("For dat%ld belonging to c%d(nbData = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);

        // Initialize tmp weights
        w[i] = 1e20;

        for(l=0;l<k;l++)
        {
            if(l != indK)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(pti, p, &(c[l]), DISTANCE_EUCLIDEAN);

                //SAY("w[%ld] = %lf, dist = %lf for c%d (k = %d, indK = %d)", pti->ind, w[i], dist, l, k, indK);
                if(dist < w[i])
                {
                    w[i] = dist;
                }
            }
        } 

        // Compute ratio distance with its centroid / distance with the nearest other centroid 
        //WRN("w[%ld] = %lf", pti->ind, w[i]);
        w[i] = 1.0 / w[i];
        //INF("w[%ld] = %lf", pti->ind, w[i]);
        sumWeights += w[i];

        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = w[i];
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    // Based on the nearest centroid different of the point one

    uint64_t i;
    uint64_t l;
    uint64_t nbDataClust = c[indK].nbData;
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        //WRN("For dat%ld belonging to c%d(nbData = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);

        // Initialize tmp weights
        w[i] = 1e20;

        for(l=0;l<k;l++)
        {
            if(l != indK)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(pti, p, &(c[l]), DISTANCE_EUCLIDEAN);

                //SAY("w[%ld] = %lf, dist = %lf for c%d (k = %d, indK = %d)", pti->ind, w[i], dist, l, k, indK);
                if(dist < w[i])
                {
                    w[i] = dist;
                }
            }
        } 

        // Compute ratio distance with its centroid / distance with the nearest other centroid 
        //WRN("w[%ld] = %lf", pti->ind, w[i]);
        w[i] = 1.0 / w[i];
        //INF("w[%ld] = %lf", pti->ind, w[i]);
        sumWeights += w[i];

        // Update pti
        pti = (data *)pti->succ;
    }

    // Compute objects weights
    pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = (w[i] / sumWeights) * (double) nbDataClust;
        }

        //WRN("ow[%ld] = %lf", i, ow[i]);
        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    // Based on the sum of distances with other centroids 

    uint64_t i;
    uint64_t l;
    uint64_t nbDataClust = c[indK].nbData;
    double sumDist = 0.0;

    //ERR("------------------------------------------");

    // Compute sum of distances with other centroids
    for(l=0;l<k;l++)
    {
        if(l != indK)
        {
            // Calculate squared Euclidean distance
            sumDist += CLUSTER_computeSquaredDistanceClusterToCluster(&(c[indK]), &(c[l]), p, DISTANCE_EUCLIDEAN);

        }
    }

    //WRN("sumDist[%d] = %lf, 1 / sumDist = %lf", indK, sumDist, (1/sumDist));

    // Compute objects weights
    data *pti = (data *)c[indK].head;
    double sumWei = 0.0;
    for(i=0;i<nbDataClust;i++)
    {
        pti->ow = (1 / sumDist);

        sumWei += pti->ow;

        //WRN("ow[%ld] = %lf", i, pti->ow);
        // Update pti
        pti = (data *)pti->succ;
    }
    //WRN("sumWei[%d] = %lf (nbData = %ld)", indK, sumWei, nbDataClust);
}

static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    double SSE = 0.0;
    uint64_t i;

    for(i=0;i<n;i++)
    {
        SSE += CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN); 
    }

    return SSE;
}

static void CLUSTER_computeNkWeightedWSS(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double wss[k])
{
    uint64_t i;
    uint32_t l;

    for(l=0;l<k;l++)
    {
        wss[l] = 0.0;
    }

    for(i=0;i<n;i++)
    {
        //wss[dat[i].clusterID] += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)&(fw[dat[i].clusterID]), dat[i].ow)/(double)c[dat[i].clusterID].nbData;
        wss[dat[i].clusterID] += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)c[dat[i].clusterID].fw, dat[i].ow)/(double)c[dat[i].clusterID].nbData;
    }
}

static double CLUSTER_computeNkWeightedWSSInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    uint64_t i;
    double wss = 0.0;

    /*for(i=0;i<n;i++)
      {
      if(dat[i].clusterID == indK)
      {
      wss += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[indK]), DISTANCE_EUCLIDEAN, (double *)&(fw[indK]), ow[i])/(double)c[indK].nbData;
      }
      }*/

    data *pti = (data *)c[indK].head;  
    for(i=0;i<c[indK].nbData;i++)
    {
        //wss += CLUSTER_computeSquaredDistanceWeightedPointToCluster(pti, p, &(c[indK]), DISTANCE_EUCLIDEAN, (double *)&(fw[indK]), pti->ow)/(double)c[indK].nbData;
        wss += CLUSTER_computeSquaredDistanceWeightedPointToCluster(pti, p, &(c[indK]), DISTANCE_EUCLIDEAN, (double *)c[indK].fw, pti->ow)/(double)c[indK].nbData;

        // Update pt
        pti = (data *)pti->succ;
    }

    return wss;
}

static void CLUSTER_ComputeMatDistPointToPoint2(data *dat, uint64_t n, uint64_t p, double ***dist)
{
    if(dat == NULL || n < 2 || p < 1)
    {
        ERR("Bad parameter");
    }
    else
    {
        *dist = malloc(n * sizeof(double *));
        if(*dist == NULL)
        {
            ERR("Fails to allocate global memory for distances matrix");
        }
        else
        {
            uint64_t i, j;
            // Create the distance matrix of i vs j
            for(i=0;i<n;i++)
            {
                (*dist)[i] = malloc(n * sizeof(double));
                if((*dist)[i] == NULL)
                {
                    ERR("Fails to allocate dimension memory for distances matrix");
                }
                else
                {
                    for (j=0;j<n;j++)
                    {
                        (*dist)[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);
                    }
                }
            }
        }
    }
}

static void CLUSTER_FreeMatDistPointToPoint(uint64_t n, double ***dist)
{
    if(n < 2)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i;
        for (i=0;i<n;i++)
        {
            free((*dist)[i]);
        }
        free(*dist);
    }
}
