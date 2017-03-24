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
#define K_MIN 2

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

/** @brief Contains the different weights calculation types.
 *
 */
typedef enum _eMethodType 
{
    METHOD_SILHOUETTE = 0,
    METHOD_SSE,
    METHOD_ABOD,
    METHOD_AVERAGE_SSE,
    METHOD_MEDIAN,
    METHOD_DISPERSION,
    METHOD_OTHER
} eMethodType; 

/** @brief Proceeds to a fake assigenment of data to  
 *         the different clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, uint32_t k);

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
 *  @return Void.
 */
static double CLUSTER_assignDataToCentroids7(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static double CLUSTER_assignDataToCentroids8(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static double CLUSTER_assignDataToCentroids9(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static double CLUSTER_assignDataToCentroids10(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static void CLUSTER_assignDataToCentroids11(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

static void CLUSTER_assignDataToCentroids12(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns weighted data to the nearest centroid.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param fw The features weights.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static double CLUSTER_assignWeightedDataToCentroids15(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow);

static double CLUSTER_assignWeightedDataToCentroids16(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow);

static double CLUSTER_assignWeightedDataToCentroids17(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_assignWeightedDataToCentroids18(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_assignWeightedDataToCentroids19(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_assignWeightedDataToCentroids20(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_assignWeightedDataToCentroids21(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_assignWeightedDataToCentroids22(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static void CLUSTER_assignWeightedDataToCentroids23(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist, bool *conv);

static void CLUSTER_assignWeightedDataToCentroids24(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist, bool *conv);

static void CLUSTER_assignWeightedDataToCentroids25(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist, bool *conv);

static void CLUSTER_assignWeightedDataToCentroids26(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double ow[n][k], bool *conv);

/** @brief Computes the squared distance between a point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d);

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

/** @brief Computes the squared distance between a weighted 
 *         point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @param fw The pointer to the features weights.
 *  @param ow The point weight.
 *  @param sd The array of dimensions standard deviation.
 *  @return The computed distance between the weigted point 
 *          and the cluster.
 */
static double CLUSTER_computeNormalizedSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow, double sd[p]);

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

/** @brief Updates a centroid positions.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The cluster number. 
 *  @return Void.
 */
static void CLUSTER_computeCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static void CLUSTER_transferPointToCluster(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK);

static void CLUSTER_transferPointToCluster2(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK);

static void kmeans_Lloyd(double *x, int *pn, int *pp, double *cen, int *pk, int *cl, int *pmaxiter, int *nc, double *wss);

static double CLUSTER_kmeans_Lloyd(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);

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
static double CLUSTER_kmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);
static double CLUSTER_kmeans2(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);
static double CLUSTER_kmeans3(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);
static double CLUSTER_kmeans4(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);

/** @brief Computes the classical version of k-means  
 *         algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param internalFeatureWeights The boolean that 
 *           specified if the features weights come 
 *           from internal computation or from a file.
 *  @param fw The features weights.
 *  @param internalObjectWeights The boolean that 
 *           specified if the objects weights come 
 *           from internal computation or from a file.
 *  @param ow The pointer to the objects weights.
 *  @return The sum of squared errors for the clustering.
 */
static double CLUSTER_weightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow);

static double CLUSTER_weightedKmeans2(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double dist[n][n]);

static double CLUSTER_weightedKmeans3(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_weightedKmeans4(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_weightedKmeans5(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_weightedKmeans6(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_weightedKmeans7(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist);

static double CLUSTER_weightedKmeans8(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double ow[n][k], double **dist);

/** @brief Computes the silhouette score for a 
 *         clustering.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return The computed silhouette score for the clustering.
 */
static double CLUSTER_computeSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static double CLUSTER_computeSilhouette2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double dist[n][n]);

static double CLUSTER_computeSilhouette3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

static double CLUSTER_computeSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes the silhouette score with weights for a 
 *         clustering.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param fw The features weights.
 *  @param ow The pointer to the objects weights.
 *  @return The computed silhouette score for the clustering.
 */
static double CLUSTER_computeWeightedSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow);

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

/** @brief Computes the distance between a weighted point
 *         and an other point .
 *
 *  @param iDat The pointer to the first point.
 *  @param jDat The pointer to the second point.
 *  @param p The number of data dimensions.
 *  @param d The type of distance calculation.
 *  @param fw The pointer to the features weights.
 *  @param ow The weight of the first object.
 *  @return The computed distance between the first point 
 *          and the second point.
 */
static double CLUSTER_computeDistanceWeightedPointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d, double *fw, double iOw);

/** @brief Computes the total sum of squares for 
 *         the data.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @return The computed total sum of squares for the data.
 */
static double CLUSTER_computeTSS(data *dat, uint64_t n, uint64_t p); 
static double CLUSTER_computeTSS2(data *dat, uint64_t n, uint64_t p);

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
static double CLUSTER_computeVRC(double TSS, double SSE, uint64_t n, uint32_t k);  

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
 *  @param ow The pointer to the objects weights.
 *  @param n The number of objects. 
 *  @return Void.
 */
static void CLUSTER_initObjectWeights(double *ow, uint64_t n);
static void CLUSTER_initObjectWeights2(uint64_t n, uint32_t k, double ow[n][k]);

/** @brief Initializes features weights to 1. 
 *
 *  @param k The number of clusters
 *  @param p The number of weights dimensions.
 *  @param fw The features weights.
 *  @return Void.
 */
static void CLUSTER_initFeatureWeights(uint32_t k, uint64_t p, double fw[k][p]);

/** @brief Computes features weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param fw The features weights.
 *  @param m The method for objects weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], eMethodType m);

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
static void CLUSTER_computeFeatureWeightsViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], uint8_t norm);

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
 *  @param ow The pointer to the objects weights.
 *  @param m The method for objects weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m);

static void CLUSTER_computeObjectWeights2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m, double dist[n][n]);

static void CLUSTER_computeObjectWeights3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m, double **dist);

static void CLUSTER_computeObjectWeights4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double ow[n][k], eMethodType m, double **dist);

/** @brief Computes objects weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The id of the cluster.
 *  @param ow The pointer to the objects weights.
 *  @param m The method for objects weights calculation.
 *  @return Void.
 */

static void CLUSTER_computeObjectWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double *ow, eMethodType m, double **dist);

/** @brief Computes objects weights via silhouette
 *         score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

static void CLUSTER_computeObjectWeightsViaSilhouette2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, double dist[n][n]);

static void CLUSTER_computeObjectWeightsViaSilhouette3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, double **dist);

static void CLUSTER_computeObjectWeightsViaSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, double **dist);

static void CLUSTER_computeObjectWeightsInClusterViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double *ow, double **dist);

static double CLUSTER_computeObjectWeightInClusterViaSilhouette(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist);

static double CLUSTER_computeObjectWeightInClusterViaSilhouette2(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist);

static double CLUSTER_computeObjectWeightInClusterViaMedian(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t indK);

static double CLUSTER_computeObjectWeightInClusterViaMedian2(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t indK);

/** @brief Computes objects weights via the sum 
 *         of squared errors.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

/** @brief Computes objects weights via the angle 
 *         based detection of outliers.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaABOD(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

/** @brief Computes objects weights via the average sum 
 *         of squared errors.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaAverageSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

/** @brief Computes objects weights via the median.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param ow The pointer to the objects weights.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

static void CLUSTER_computeObjectWeightsViaMedian2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

static void CLUSTER_computeObjectWeightsViaMedian3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

static void CLUSTER_computeObjectWeightsInClusterViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow);

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

static double CLUSTER_computeSSEInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

static void CLUSTER_computeWSS(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double wss[k]);

/** @brief Computes the sum of squared errors for a 
 *         clustering. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param fw The features weights.
 *  @param ow The pointer to the objects weights.
 *  @return The computed sum of squared errors for a 
 *          clustering.
 */
static double CLUSTER_computeWeightedSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow);

static double CLUSTER_computeWeightedSSE2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow);

static double CLUSTER_computeNkWeightedSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow);

static double CLUSTER_computeWeightedSSEInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow);

static double CLUSTER_computeNkWeightedSSEInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow);

/** @brief Computes the matrix of distances 
 *         points to points. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param dist The triple pointer to the matrix of distances.
 *  @return Void.
 */
static void CLUSTER_ComputeMatDistPointToPoint(data *dat, uint64_t n, uint64_t p, double dist[n][n]);

static void CLUSTER_ComputeMatDistPointToPoint2(data *dat, uint64_t n, uint64_t p, double ***dist);

/** @brief Frees the matrix of distances 
 *         points to points. 
 *
 *  @param n The number of the data.
 *  @param dist The triple pointer to the matrix of distances.
 *  @return Void.
 */
static void CLUSTER_FreeMatDistPointToPoint(uint64_t n, double ***dist);

/** @brief Computes the standard deviation 
 *         for a dimension in a cluster. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The cluster ID.
 *  @param sd The array of dimensions standard deviation.
 *  @return Void.
 */
static void CLUSTER_ComputeStdDevForDimInClust(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double sd[p]);


/* -- Function definitions -- */

void CLUSTER_computeKmeans4(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax], statVRC2[kmax], statCH[kmax], statCH2[kmax], statSSE[kmax];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k

    // Initialize statistics
    for(k=kmax;k>=K_MIN;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = 0.0;
        statCH2[k] = 0.0;
        statSSE[k] = 1.0e20;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);
    //double TSS2 = CLUSTER_computeTSS2(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);


    //WRN("Iteration %d", i);
    for(k=kmax;k>=K_MIN;k--) // From kMax to kMin
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
    for(k=kmax;k>=K_MIN;k--)
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

        /*if(statCH2[k] > ch2Max)
        {
            ch2Max = statCH2[k];
            kCh2Max = k;
        }*/
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

void CLUSTER_computeWeightedKmeans2(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, bool internalObjectWeights, const char *objectWeightsFile)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax], statVRC2[kmax], statCH[kmax], statSSE[kmax];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k
    double ow[n]; // Objects weights
    double fw[kmax][p]; // Features weights

    // Initialize statistics
    for(k=kmax;k>=K_MIN;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = 0.0;
        statSSE[k] = 1.0e20;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);

    // Initialize weights to 1.0
    CLUSTER_initFeatureWeights(kmax, p, fw);
    CLUSTER_initObjectWeights(ow, n);
    if(internalFeatureWeights == false)
    {
        // Read object weights from file
    }

    if(internalObjectWeights == false)
    {
        // Read object weights from file
    }

    //WRN("Iteration %d", i);
    for(k=kmax;k>=K_MIN;k--) // From kMax to kMin
    {
        /*cluster test[k], test2[k];
        double WSS[k];
        double WSS2[k];
        for(o=0;o<k;o++)
        {
            WSS[o] = 0.0;
            WSS2[o] = 0.0;
        }
        uint32_t assign[n];
        CLUSTER_initClusters(p, test, k);
        CLUSTER_initClusters(p, test2, k);*/
        for(i=0;i<nbRep;i++) // Number of replicates
        {
            // Initialize weights
            if(internalFeatureWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initFeatureWeights(k, p, fw);
            }

            if(internalObjectWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initObjectWeights(ow, n);
            }

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            //double wSSE = CLUSTER_weightedKmeans5(dat, n, p, k, c, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            double wSSE = CLUSTER_weightedKmeans6(dat, n, p, k, c, internalFeatureWeights, fw, internalObjectWeights, ow, dist);

            //SAY("SSE = %lf, TSS = %lf, n = %ld, k = %d", SSE, TSS2, n, k);

            /*double wss[k];
            CLUSTER_computeWSS(dat, n, p, c, k, wss);*/

            // Compute the non-weighted SSE for statistics computation
            double SSE = CLUSTER_computeSSE(dat, n, p, c, k);
            
            // Compute silhouette statistic
            double sil = CLUSTER_computeSilhouette4(dat, n, p, c, k, dist);
            //SAY("Silhouette = %lf", sil);

            // Compute VRC statistic
            double vrc2 = CLUSTER_computeVRC2(dat, c, SSE, n, p, k);
            //SAY("VRC2 = %lf", vrc2);

            // Compute CH statistic
            double ch = CLUSTER_computeCH(TSS, SSE, n, k);
            //SAY("CH = %lf", ch);

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

            //WRN("sil = %lf, statSil = %lf, clustNull = %d", sil, statSil[k], clustNull);
            if((sil > statSil[k] || i == 0) && clustNull == false)
            {
                statSil[k] = sil;
                // Save data membership for each k (VRC)
                for(j=0;j<n;j++)
                {
                    silGrp[k][j] = dat[j].clusterID;
                }
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
                //ERR("YES : SSE = %lf, statSSE = %lf, i = %d, clustNull = %d", SSE, statSSE[k], i, clustNull);
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
        INF("Best sil : %lf for k = %d", statSil[k], k);
        
        uint32_t l;*/
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
    for(k=kmax;k>=K_MIN;k--)
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

void CLUSTER_computeWeightedKmeans3(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, bool internalObjectWeights, const char *objectWeightsFile)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax], statVRC2[kmax], statCH[kmax], statSSE[kmax];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k
    double ow[n]; // Objects weights
    double fw[kmax][p]; // Features weights

    // Initialize statistics
    for(k=kmax;k>=K_MIN;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = 0.0;
        statSSE[k] = 1.0e20;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);

    // Initialize weights to 1.0
    CLUSTER_initFeatureWeights(kmax, p, fw);
    CLUSTER_initObjectWeights(ow, n);
    if(internalFeatureWeights == false)
    {
        // Read object weights from file
    }

    if(internalObjectWeights == false)
    {
        // Read object weights from file
    }

    //WRN("Iteration %d", i);
    for(k=kmax;k>=K_MIN;k--) // From kMax to kMin
    {
        for(i=0;i<nbRep;i++) // Number of replicates
        {
            // Initialize weights
            if(internalFeatureWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initFeatureWeights(k, p, fw);
            }

            if(internalObjectWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initObjectWeights(ow, n);
            }

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            double SSE = CLUSTER_weightedKmeans7(dat, n, p, k, c, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            
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
    for(k=kmax;k>=K_MIN;k--)
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

void CLUSTER_computeWeightedKmeans4(data *dat, uint64_t n, uint64_t p, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, bool internalObjectWeights, const char *objectWeightsFile)
{
    uint32_t i, k, o;
    uint64_t j;
    double statSil[kmax], statVRC2[kmax], statCH[kmax], statSSE[kmax];
    uint32_t silGrp[kmax+1][n], vrc2Grp[kmax+1][n], chGrp[kmax+1][n]; // Data membership for each k
    //double ow[n];
    double ow[n][kmax]; // Objects weights
    double fw[kmax][p]; // Features weights

    // Initialize statistics
    for(k=kmax;k>=K_MIN;k--)
    {
        statSil[k] = -1.0;
        statVRC2[k] = 0.0;
        statCH[k] = 0.0;
        statSSE[k] = 1.0e20;
    }

    // Compute total sum of squares 
    double TSS = CLUSTER_computeTSS(dat, n, p);

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint2(dat, n, p, &dist);

    // Initialize weights to 1.0
    CLUSTER_initFeatureWeights(kmax, p, fw);
    CLUSTER_initObjectWeights2(n, kmax, ow);
    if(internalFeatureWeights == false)
    {
        // Read object weights from file
    }

    if(internalObjectWeights == false)
    {
        // Read object weights from file
    }

    //WRN("Iteration %d", i);
    for(k=kmax;k>=K_MIN;k--) // From kMax to kMin
    {
        for(i=0;i<nbRep;i++) // Number of replicates
        {
            // Initialize weights
            if(internalFeatureWeights == true)
            {
                // Initialize object weights to 1.0
                CLUSTER_initFeatureWeights(k, p, fw);
            }

            if(internalObjectWeights == true)
            {
                // Initialize object weights to 1.0
                //CLUSTER_initObjectWeights(ow, n);
                CLUSTER_initObjectWeights2(n, kmax, ow);
            }

            cluster c[k];
            // Allocate clusters dimension memory
            CLUSTER_initClusters(p, c, k);

            double SSE = CLUSTER_weightedKmeans8(dat, n, p, k, c, internalFeatureWeights, fw, internalObjectWeights, ow, dist);

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
    for(k=kmax;k>=K_MIN;k--)
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

static double CLUSTER_kmeans2(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        uint8_t iter = 0;
        double SSEref = 1.0e20, SSE;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Assign data to the nearest centroid
            SSE = CLUSTER_assignDataToCentroids7(dat, n, p, c, k);

            // Update centroids position
            CLUSTER_computeCentroids(dat, n, p, c, k);

            // Test algorithm convergence
            if(fabs(SSEref - SSE) > (SSE / 1000.0))            
            {
                SSEref = SSE;
            }
            else			
            {
                conv = true;
            }

            iter++;
        }

        return SSE; 
    }
}

static double CLUSTER_kmeans3(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        uint8_t iter = 0;
        double SSEref = 1.0e20, SSE;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Assign data to the nearest centroid
            SSE = CLUSTER_assignDataToCentroids8(dat, n, p, c, k); // MacQueen
            //SSE = CLUSTER_assignDataToCentroids10(dat, n, p, c, k); // Hartigan and Wong

            // Test algorithm convergence
            if(fabs(SSEref - SSE) > (SSE / 1000.0))            
            {
                SSEref = SSE;
            }
            else			
            {
                conv = true;
            }

            iter++;
        }

        return SSE; 
    }
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
        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        uint8_t iter = 0;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            CLUSTER_assignDataToCentroids11(dat, n, p, c, k, &conv); // MacQueen
            //CLUSTER_assignDataToCentroids12(dat, n, p, c, k, &conv); // Hartigan and Wong

            iter++;
        }

        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

static double CLUSTER_weightedKmeans5(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        // Initialization
        eMethodType objWeiMed = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;
        eMethodType feaWeiMed = METHOD_DISPERSION;
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        //CLUSTER_assignWeightedDataToCentroids12(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMed, dist);
        }

        uint8_t iter = 0;
        double SSEref = 1.0e20, SSE;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Assign data to the nearest centroid
            //SSE = CLUSTER_assignWeightedDataToCentroids12(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);
            //SSE = CLUSTER_assignWeightedDataToCentroids13(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);
            //SSE = CLUSTER_assignWeightedDataToCentroids14(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);
            SSE = CLUSTER_assignWeightedDataToCentroids15(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);
            //SSE = CLUSTER_assignWeightedDataToCentroids16(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);

            // Update centroids position
            CLUSTER_computeCentroids(dat, n, p, c, k);

            // Update weights
            if(internalFeatureWeights == true)
            {
                // Internal computation of features weights
                CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
            }

            if(internalObjectWeights == true)
            {
                // Internal computation of objects weights
                CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMed, dist);
            }

            // Test algorithm convergence
            if(fabs(SSEref - SSE) > (SSE / 1000.0))            
            {
                SSEref = SSE;
            }
            else			
            {
                conv = true;
            }

            iter++;
        }

        return SSE; 
    }
}

static double CLUSTER_weightedKmeans6(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        // Initialization
        eMethodType objWeiMed = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;
        eMethodType feaWeiMed = METHOD_DISPERSION;
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        //CLUSTER_assignWeightedDataToCentroids12(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMed, dist);
        }

        uint8_t iter = 0;
        double SSEref = 1.0e20, SSE;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Assign data to the nearest centroid
            //SSE = CLUSTER_assignWeightedDataToCentroids17(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            //SSE = CLUSTER_assignWeightedDataToCentroids18(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            //SSE = CLUSTER_assignWeightedDataToCentroids19(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            //SSE = CLUSTER_assignWeightedDataToCentroids20(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            //SSE = CLUSTER_assignWeightedDataToCentroids21(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist);
            SSE = CLUSTER_assignWeightedDataToCentroids22(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist);

            // Update weights
            if(internalFeatureWeights == true)
            {
                // Internal computation of features weights
                CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
            }

            // Test algorithm convergence
            if(fabs(SSEref - SSE) > (SSE / 1000.0))            
            {
                SSEref = SSE;
            }
            else			
            {
                conv = true;
            }

            iter++;
        }

        return SSE; 
    }
}

static double CLUSTER_weightedKmeans7(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        // Initialization
        eMethodType objWeiMed = METHOD_MEDIAN/*METHOD_SILHOUETTE*/;
        eMethodType feaWeiMed = METHOD_DISPERSION;
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMed, dist);
        }

        uint8_t iter = 0;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            //CLUSTER_assignWeightedDataToCentroids23(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist, &conv); // Based on WSS and 1/nk
            //CLUSTER_assignWeightedDataToCentroids24(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist, &conv); // Based on SSE and 1/nk
            CLUSTER_assignWeightedDataToCentroids25(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, dist, &conv); // Based on SSE

            iter++;
        }

        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

static double CLUSTER_weightedKmeans8(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double ow[n][k], double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        ERR("k-means : bad parameter : dat = %p, n = %ld, p = %ld, k = %d", dat, n, p, k);
        return -1.0;
    }
    else
    {
        // Initialization
        eMethodType objWeiMed = METHOD_MEDIAN/*METHOD_SILHOUETTE*/;
        eMethodType feaWeiMed = METHOD_DISPERSION;
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, k);
        CLUSTER_randomCentroids(dat, n, p, c, k);
        CLUSTER_assignDataToCentroids7(dat, n, p, c, k);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights4(dat, n, p, c, k, ow, objWeiMed, dist);
        }

        uint8_t iter = 0;
        bool conv = false; // Has converged
        while(iter < NB_ITER && conv == false)
        {
            /*uint64_t y;
            uint32_t z;
            double sum[k];
            for(z=0;z<k;z++)
            {
                sum[z] = 0.0;
            }

            for(y=0;y<n;y++)
            {
                sum[dat[y].clusterID] += ow[y][dat[y].clusterID];
                for(z=0;z<k;z++)
                {
                    SAY("ow[%ld][%d] = %lf (c%d)", y, z, ow[y][z], dat[y].clusterID);
                }
            }

            for(z=0;z<k;z++)
            {
                SAY("sum[%d] = %lf (nbData = %ld)",z, sum[z], c[z].nbData);
            }*/

            // Data assignation
            CLUSTER_assignWeightedDataToCentroids26(dat, n, p, c, k, internalFeatureWeights, fw, internalObjectWeights, ow, &conv); // Based on SSE

            // Update centroids position
            CLUSTER_computeCentroids(dat, n, p, c, k);

            // Update weights
            if(internalFeatureWeights == true)
            {
                // Internal computation of features weights
                CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
            }

            if(internalObjectWeights == true)
            {
                // Internal computation of objects weights
                CLUSTER_computeObjectWeights4(dat, n, p, c, k, ow, objWeiMed, dist);
            }

            iter++;
        }

        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, uint32_t k)
{
    if(dat == NULL || n < 2 || k < 2)
        ERR("Bad parameter");
    else
    {
        uint64_t i;
        uint32_t l = 0;
        for(i=0;i<n;i++)
        {
            // Assign fake cluster to avoid null cluster
            if(l == k)
                l = 0;

            dat[i].clusterID = l;
            l++;
        }
    }
}

static void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k)
{
    if(p < 1 || c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        uint32_t l;
        for(l=0;l<k;l++)
        {
            c[l].centroid = malloc(p*sizeof(double)); // Allocate cluster dimension memory
            if(c[l].centroid == NULL)
            {
                ERR("Cluster dimension memory allocation failed");
            }
            else
            {
                c[l].nbData = 0;
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
        }
    }
}

static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
        ERR("Bad parameter");
    else
    {
        uint32_t l;
        uint64_t j;

        // Initialize cluster data number
        for(l=0;l<k;l++)
            c[l].nbData = 0;

        // Fake assignment
        uint64_t i;
        for(i=0;i<n;i++)
            c[dat[i].clusterID].nbData++;

        //WRN("-------------------------------");
        uint64_t rnd[k];
        for(l=0;l<k;l++)
        {
            if(c[l].centroid != NULL)
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
            else
            {
                ERR("Cluster dimensions not allocated");
            }
        }
        //WRN("-------------------------------");
    }
}

static double CLUSTER_assignDataToCentroids7(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minDist;
            double distClus;
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

            c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
            dat[i].clusterID = minK; // Assign data to cluster
            c[dat[i].clusterID].nbData++; // Increase new cluster data number
            SSE += minDist;

            //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_assignDataToCentroids8(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        // MacQueen algorithm //
        
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minDist;
            double distClus;
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

            // Transfer point i to cluster minK
            CLUSTER_transferPointToCluster2(dat, i, p, c, minK);

            SSE += CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[minK]), DISTANCE_EUCLIDEAN);;

            //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_assignDataToCentroids9(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        // Hartigan and Wong algorithm //
        
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minSSE = CLUSTER_computeSSEInCluster(dat, n, p, c, dat[i].clusterID);
            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;
            uint32_t minK;

            for(l=0;l<k;l++)
            {
                if(l != curClust)
                {
                    double d;

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                    // Calculate squared SSE in cluster l 
                    SSE = CLUSTER_computeSSEInCluster(dat, n, p, c, dat[i].clusterID);

                    if(SSE < minSSE)
                    {
                        minSSE = SSE;
                        minK = l; // Save the cluster for the min distance
                        curClust = l;
                    }
                    else
                    {
                        // Transfer point i to cluster l
                        CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);
                    }
                }
            }
        }

        return CLUSTER_computeSSE(dat, n, p, c, k);
    }
}

static double CLUSTER_assignDataToCentroids10(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        // Hartigan and Wong algorithm //
        
        uint64_t i;
        uint32_t l;
        double SSE[k];

        for(l=0;l<k;l++)
        {
            // Compute SSE in cluster l
            SSE[l] = CLUSTER_computeSSEInCluster(dat, n, p, c, l);

            for(i=0;i<n;i++)
            {
                // Save datum current cluster 
                uint32_t curClust = dat[i].clusterID;

                // Transfer point i to cluster l
                CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                // Recompute SS in cluster l
                double sse = CLUSTER_computeSSEInCluster(dat, n, p, c, l);

                if(sse < SSE[l])
                {
                    SSE[l] = sse;
                }
                else
                {
                    // Transfer point i to its previous cluster 
                    CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);
                }
            }
        }

        double finalSSE = 0.0;
        for(l=0;l<k;l++)
        {
            finalSSE += CLUSTER_computeSSEInCluster(dat, n, p, c, l);
        }

        return finalSSE;
    }
}

static void CLUSTER_assignDataToCentroids11(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        ERR("Bad parameter");
        return -1;
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
            double distClus;
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

static void CLUSTER_assignDataToCentroids12(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        // Hartigan and Wong algorithm //
        
        uint64_t i;
        uint32_t l;

        // Set convergence variable
        *conv = true;

        for(l=0;l<k;l++)
        {
            // Compute SSE in cluster l
            double refSSE = CLUSTER_computeSSEInCluster(dat, n, p, c, l);

            for(i=0;i<n;i++)
            {
                // Save datum current cluster 
                uint32_t curClust = dat[i].clusterID;

                // Transfer point i to cluster l
                CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                // Recompute SS in cluster l
                double sse = CLUSTER_computeSSEInCluster(dat, n, p, c, l);

                if(sse < refSSE)
                {
                    refSSE = sse;

                    // Reset convergence variable
                    *conv = false;
                }
                else
                {
                    // Transfer point i to its previous cluster 
                    CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);
                }
            }
        }
    }
}

static double CLUSTER_assignWeightedDataToCentroids15(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minDist;
            double distClus;
            uint32_t minK;
            //WRN("Dist for dat%ld (ow = %lf)", i, ow[i]);
            for(l=0;l<k;l++)
            {
                double dist;

                if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                {
                    // Calculate squared Euclidean distance
                    dist = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]);
                }
                else
                {
                    // Calculate squared Euclidean distance
                    dist = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]) / (double) c[l].nbData;
                }

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

            c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
            dat[i].clusterID = minK; // Assign data to cluster
            c[dat[i].clusterID].nbData++; // Increase new cluster data number
            SSE += minDist;

            //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids16(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double SSE = 0.0;

        for(i=0;i<n;i++)
        {
            double minDist;
            double distClus;
            uint32_t minK;
            //WRN("Dist for dat%ld (ow = %lf)", i, ow[i]);
            for(l=0;l<k;l++)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]);

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

            c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
            dat[i].clusterID = minK; // Assign data to cluster
            c[dat[i].clusterID].nbData++; // Increase new cluster data number
            SSE += minDist;

            //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids17(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double SSE = 0.0;
        eMethodType objWeiMet = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;

        for(i=0;i<n;i++)
        {
            double minDist;
            uint32_t minK;

            //WRN("Dist for dat%ld (ow = %lf)", i, ow[i]);
            for(l=0;l<k;l++)
            {
                double d;
                double w[n];
                double fromClus[p];
                double toClus[p];

                // Save current cluster of point i
                uint32_t prevClust = dat[i].clusterID;

                //WRN("Dat[%ld] from cluster %d to cluster %d",i,prevClust, l);

                // Save current positions of cluster l and prevClust 
                //printf("Current centroids : ");
                for(j=0;j<p;j++)
                {
                   fromClus[j] = c[prevClust].centroid[j];
                   toClus[j] = c[l].centroid[j];
                   //printf("toClus[%ld] = %lf, ",j, toClus[j]);
                }
                //SAY("");
                
                // Save current weights
                //printf("Current weights : ");
                for(j=0;j<n;j++)
                {
                    w[j] = ow[j];
                    //printf("ow[%ld] = %lf, ",j, ow[j]);
                }
                //SAY("");
                
                //SAY("BEF : nbData c%d = %ld, nbData c%d = %ld", prevClust, c[prevClust].nbData, l, c[l].nbData);
                // Temporary moving of point i to cluster l
                c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                dat[i].clusterID = l; // Assign data to cluster
                c[dat[i].clusterID].nbData++; // Increase new cluster data number
                //SAY("AFT : nbData c%d = %ld, nbData c%d = %ld", prevClust, c[prevClust].nbData, l, c[l].nbData);

                // Update previous and new clusters of point i
                CLUSTER_computeCentroid(dat, n, p, c, prevClust);
                CLUSTER_computeCentroid(dat, n, p, c, dat[i].clusterID);
                /*printf("Updated centroids : ");
                for(j=0;j<p;j++)
                {
                   printf("toClus[%ld] = %lf, ",j, c[dat[i].clusterID].centroid[j]);
                }
                SAY("");*/

                // Update weight of points in cluster l
                if(internalObjectWeights == true)
                {
                    // Internal computation of objects weights
                    CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                    //CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMet, dist);
                    // Save current weights
                    /*printf("Updated weights : ");
                    for(j=0;j<n;j++)
                    {
                        printf("ow[%ld] = %lf, ",j, ow[j]);
                    }
                    SAY("");*/

                }

                if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                {
                    // Calculate squared Euclidean distance
                    d = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]);
                }
                else
                {
                    // Calculate squared Euclidean distance
                    d = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]) / (double) c[l].nbData;
                }

                //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);

                if(l == 0)
                {
                    minDist = d;
                    minK = l;
                    //WRN("\tFirst cluster : minDist = %lf", minDist);
                }
                else
                {
                    //WRN("\td = %lf, minDist = %lf", d, minDist);
                    if(d < minDist)
                    {
                        //INF("\tDist improved, dat[%ld] moved to cluster %d", i, l);
                        minDist = d;
                        minK = l; // Save the cluster for the min distance
                    }
                    else
                    {
                        //ERR("\tDist not improved, dat[%ld] go back to cluster %d", i, prevClust);
                        // Moving of point i to cluster l
                        c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                        dat[i].clusterID = prevClust; // Assign data to cluster
                        c[dat[i].clusterID].nbData++; // Increase new cluster data number
                        // Reset current positions of cluster l and prevClust 
                        //SAY("\tRES : nbData c%d = %ld, nbData c%d = %ld", prevClust, c[prevClust].nbData, l, c[l].nbData);
                        //printf("\tReset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            c[prevClust].centroid[j] = fromClus[j];
                            c[l].centroid[j] = toClus[j];
                            //printf("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        //SAY("");

                        // Reset weight of point i
                        //printf("\tReset weights : ");
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = w[j];
                                //printf("ow[%ld] = %lf, ",j, ow[j]);
                            }
                            //SAY("");
                        }
                    }
                }
            }

            SSE += minDist;

            //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
        }

        return SSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids18(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double SSE = 0.0;
        eMethodType objWeiMet = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;

        for(i=0;i<n;i++)
        {
            double minSSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;
            uint32_t minK;

            //WRN("SSE for dat%ld (ow = %lf)", i, ow[i]);
            for(l=0;l<k;l++)
            {
                if(l != curClust)
                {
                    double d;
                    double w[n];
                    double fromClus[p];
                    double toClus[p];

                    //WRN("Dat[%ld] from cluster %d to cluster %d",i,curClust, l);

                    // Save current positions of cluster l and prevClust 
                    //printf("Current centroids : ");
                    for(j=0;j<p;j++)
                    {
                        fromClus[j] = c[curClust].centroid[j];
                        toClus[j] = c[l].centroid[j];
                        //printf("toClus[%ld] = %lf, ",j, toClus[j]);
                    }
                    //SAY("");

                    // Save current weights
                    //printf("Current weights : ");
                    for(j=0;j<n;j++)
                    {
                        w[j] = ow[j];
                        //printf("ow[%ld] = %lf, ",j, ow[j]);
                    }
                    //SAY("");

                    //SAY("BEF : nbData c%d = %ld, nbData c%d = %ld", prevClust, c[prevClust].nbData, l, c[l].nbData);
                    // Temporary moving of point i to cluster l
                    c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                    dat[i].clusterID = l; // Assign data to cluster
                    c[dat[i].clusterID].nbData++; // Increase new cluster data number
                    //SAY("AFT : nbData c%d = %ld, nbData c%d = %ld", prevClust, c[prevClust].nbData, l, c[l].nbData);

                    // Update previous and new clusters of point i
                    CLUSTER_computeCentroid(dat, n, p, c, curClust);
                    CLUSTER_computeCentroid(dat, n, p, c, dat[i].clusterID);
                    /*printf("Updated centroids : ");
                      for(j=0;j<p;j++)
                      {
                      printf("toClus[%ld] = %lf, ",j, c[dat[i].clusterID].centroid[j]);
                      }
                      SAY("");*/

                    // Update weight of points in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of objects weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                        //CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMet, dist);
                        // Save current weights
                        /*printf("Updated weights : ");
                          for(j=0;j<n;j++)
                          {
                          printf("ow[%ld] = %lf, ",j, ow[j]);
                          }
                          SAY("");*/

                    }

                    if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                    {
                        // Calculate squared Euclidean distance
                        d = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]);
                    }
                    else
                    {
                        // Calculate squared SSE in cluster l 
                        SSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
                    }

                    //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);


                    //WRN("\tSSE = %lf, minSSE = %lf", SSE, minSSE);
                    if(SSE < minSSE)
                    {
                        //INF("\tDist improved, dat[%ld] moved to cluster %d", i, l);
                        minSSE = SSE;
                        minK = l; // Save the cluster for the min distance
                        curClust = l;
                    }
                    else
                    {
                        //ERR("\tDist not improved, dat[%ld] go back to cluster %d", i, curClust);
                        // Moving of point i to cluster l
                        c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                        dat[i].clusterID = curClust; // Assign data to cluster
                        c[dat[i].clusterID].nbData++; // Increase new cluster data number
                        // Reset current positions of cluster l and prevClust 
                        //SAY("\tRES : nbData c%d = %ld, nbData c%d = %ld", prevClust, c[prevClust].nbData, l, c[l].nbData);
                        //printf("\tReset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            c[curClust].centroid[j] = fromClus[j];
                            c[l].centroid[j] = toClus[j];
                            //printf("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        //SAY("");

                        // Reset weight of point i
                        //printf("\tReset weights : ");
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = w[j];
                                //printf("ow[%ld] = %lf, ",j, ow[j]);
                            }
                            //SAY("");
                        }
                    }
                }

                //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
            }
        }

        double finalSSE = 0.0;
            for(l=0;l<k;l++)
            {
                finalSSE += CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, l, fw, ow);
            }

        return finalSSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids19(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double SSE = 0.0;
        eMethodType objWeiMet = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;

        for(i=0;i<n;i++)
        {
            double minSSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;
            uint32_t minK;

            //WRN("SSE for dat%ld (ow = %lf)", i, ow[i]);
            for(l=0;l<k;l++)
            {
                if(l != curClust && (c[curClust].nbData - 1) > 0)
                {
                    double d;
                    double w[n];
                    double fromClus[p];
                    double toClus[p];

                    //WRN("Dat[%ld] from cluster %d to cluster %d",i,curClust, l);

                    // Save current positions of cluster l and prevClust 
                    /*printf("Current centroids : ");
                    for(j=0;j<p;j++)
                    {
                        fromClus[j] = c[curClust].centroid[j];
                        toClus[j] = c[l].centroid[j];
                        printf("toClus[%ld] = %lf, ",j, toClus[j]);
                    }
                    SAY("");*/

                    // Save current weights
                    //printf("Current weights : ");
                    for(j=0;j<n;j++)
                    {
                        w[j] = ow[j];
                        //printf("ow[%ld] = %lf, ",j, ow[j]);
                    }
                    //SAY("");

                    /*SAY("Current centroids : ");
                    for(j=0;j<p;j++)
                    {
                        SAY("fromClus[%ld] = %lf, ",j, c[curClust].centroid[j]);
                        SAY("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                    }
                    SAY("");*/

                    //SAY("BEF : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster(dat, i, p, c, l);
                    /*printf("Updated centroids : ");
                      for(j=0;j<p;j++)
                      {
                      printf("toClus[%ld] = %lf, ",j, c[dat[i].clusterID].centroid[j]);
                      }
                      SAY("");*/
                    //SAY("AFT : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);

                    // Update weight of points in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of objects weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                        //CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMet, dist);
                        // Save current weights
                        /*printf("Updated weights : ");
                          for(j=0;j<n;j++)
                          {
                          printf("ow[%ld] = %lf, ",j, ow[j]);
                          }
                          SAY("");*/
                    }

                    if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                    {
                        // Calculate squared Euclidean distance
                        d = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]);
                    }
                    else
                    {
                        // Calculate squared SSE in cluster l 
                        SSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
                    }

                    //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);


                    //WRN("\tSSE = %lf, minSSE = %lf", SSE, minSSE);
                    if(SSE < minSSE)
                    {
                        //INF("\tDist improved, dat[%ld] moved to cluster %d", i, l);
                        minSSE = SSE;
                        minK = l; // Save the cluster for the min distance
                        curClust = l;
                    }
                    else
                    {
                        //ERR("\tDist not improved, dat[%ld] go back to cluster %d", i, curClust);
                        //SAY("BEF : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);
                        // Transfer point i to cluster l
                        CLUSTER_transferPointToCluster(dat, i, p, c, curClust);
                        //SAY("AFT : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);
                        // Reset current positions of cluster l and prevClust 
                        /*SAY("Reset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            SAY("fromClus[%ld] = %lf, ",j, c[curClust].centroid[j]);
                            SAY("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        SAY("");*/
                        /*SAY("\tRES : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);
                        printf("\t0 - Reset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            printf("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        SAY("");
                        printf("\tReset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            c[curClust].centroid[j] = fromClus[j];
                            c[l].centroid[j] = toClus[j];
                            printf("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        SAY("");*/

                        // Reset weight of point i
                        //printf("\tReset weights : ");
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = w[j];
                                //printf("ow[%ld] = %lf, ",j, ow[j]);
                            }
                            //SAY("");
                        }
                    }
                }

                //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
            }
        }

        double finalSSE = 0.0;
            for(l=0;l<k;l++)
            {
                finalSSE += CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, l, fw, ow);
            }

        return finalSSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids20(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double SSE = 0.0;
        eMethodType objWeiMet = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;

        for(i=0;i<n;i++)
        {
            double minSSE = CLUSTER_computeNkWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;
            uint32_t minK;

            //WRN("SSE for dat%ld (ow = %lf)", i, ow[i]);
            for(l=0;l<k;l++)
            {
                if(l != curClust)
                {
                    double d;
                    double w[n];

                    //WRN("Dat[%ld] from cluster %d to cluster %d",i,curClust, l);

                    // Save current positions of cluster l and prevClust 
                    /*printf("Current centroids : ");
                    for(j=0;j<p;j++)
                    {
                        fromClus[j] = c[curClust].centroid[j];
                        toClus[j] = c[l].centroid[j];
                        printf("toClus[%ld] = %lf, ",j, toClus[j]);
                    }
                    SAY("");*/

                    // Save current weights
                    //printf("Current weights : ");
                    for(j=0;j<n;j++)
                    {
                        w[j] = ow[j];
                        //printf("ow[%ld] = %lf, ",j, ow[j]);
                    }
                    //SAY("");

                    /*SAY("Current centroids : ");
                    for(j=0;j<p;j++)
                    {
                        SAY("fromClus[%ld] = %lf, ",j, c[curClust].centroid[j]);
                        SAY("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                    }
                    SAY("");*/

                    //SAY("BEF : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);
                    /*printf("Updated centroids : ");
                      for(j=0;j<p;j++)
                      {
                      printf("toClus[%ld] = %lf, ",j, c[dat[i].clusterID].centroid[j]);
                      }
                      SAY("");*/
                    //SAY("AFT : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);

                    // Update weight of points in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of objects weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                        //CLUSTER_computeObjectWeights3(dat, n, p, c, k, ow, objWeiMet, dist);
                        // Save current weights
                        /*printf("Updated weights : ");
                          for(j=0;j<n;j++)
                          {
                          printf("ow[%ld] = %lf, ",j, ow[j]);
                          }
                          SAY("");*/
                    }

                    if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                    {
                        // Calculate squared Euclidean distance
                        d = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i]);
                    }
                    else
                    {
                        // Calculate squared SSE in cluster l 
                        SSE = CLUSTER_computeNkWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
                    }

                    //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);


                    //WRN("\tSSE = %lf, minSSE = %lf", SSE, minSSE);
                    if(SSE < minSSE)
                    {
                        //INF("\tDist improved, dat[%ld] moved to cluster %d", i, l);
                        minSSE = SSE;
                        minK = l; // Save the cluster for the min distance
                        curClust = l;
                    }
                    else
                    {
                        //ERR("\tDist not improved, dat[%ld] go back to cluster %d", i, curClust);
                        //SAY("BEF : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);
                        // Transfer point i to cluster l
                        CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);
                        //SAY("AFT : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);
                        // Reset current positions of cluster l and prevClust 
                        /*SAY("Reset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            SAY("fromClus[%ld] = %lf, ",j, c[curClust].centroid[j]);
                            SAY("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        SAY("");*/
                        /*SAY("\tRES : nbData c%d = %ld, nbData c%d = %ld", curClust, c[curClust].nbData, l, c[l].nbData);
                        printf("\t0 - Reset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            printf("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        SAY("");
                        printf("\tReset centroids : ");
                        for(j=0;j<p;j++)
                        {
                            c[curClust].centroid[j] = fromClus[j];
                            c[l].centroid[j] = toClus[j];
                            printf("toClus[%ld] = %lf, ",j, c[l].centroid[j]);
                        }
                        SAY("");*/

                        // Reset weight of point i
                        //printf("\tReset weights : ");
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = w[j];
                                //printf("ow[%ld] = %lf, ",j, ow[j]);
                            }
                            //SAY("");
                        }
                    }
                }

                //SAY("dat[%ld] centroid : %d (nb data = %ld)", i, dat[i].clusterID, c[dat[i].clusterID].nbData);
            }
        }

        double finalSSE = 0.0;
        for(l=0;l<k;l++)
        {
            finalSSE += CLUSTER_computeNkWeightedSSEInCluster(dat, n, p, c, l, fw, ow);
        }

        return finalSSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids21(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        eMethodType objWeiMet = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;

        // Compute initial SSE
        double minSSE;
        if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
        {
            minSSE = CLUSTER_computeWeightedSSE(dat, n, p, c, k, fw, ow);
        }
        else
        {
            minSSE = CLUSTER_computeNkWeightedSSE(dat, n, p, c, k, fw, ow);
        }

        for(i=0;i<n;i++)
        {
            for(l=0;l<k;l++)
            {
                // Save datum current cluster 
                uint32_t curClust = dat[i].clusterID;
                double w[n];

                if(l != curClust)
                {
                    // Save current object weights
                    for(j=0;j<n;j++)
                    {
                        w[j] = ow[j];
                    }

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                    // Update object weight of points in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of objects weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                    }

                    double sse;
                    // Recompute SSE 
                    if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                    {
                        sse = CLUSTER_computeWeightedSSE(dat, n, p, c, k, fw, ow);
                    }
                    else
                    {
                        sse = CLUSTER_computeNkWeightedSSE(dat, n, p, c, k, fw, ow);
                    }

                    // Test if SSE improved
                    if(sse < minSSE)
                    {
                        minSSE = sse;
                    }
                    else
                    {
                        // Transfer point i to its previous cluster 
                        CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);

                        // Reset weight of point i
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = w[j];

                            }
                        }
                    }
                }
            }
        }

        return minSSE;
    }
}

static double CLUSTER_assignWeightedDataToCentroids22(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        eMethodType objWeiMet = /*METHOD_MEDIAN*/METHOD_SILHOUETTE;

        for(l=0;l<k;l++)
        {
            // Compute SSE in cluster l
            double refSSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, l, fw, ow);

            for(i=0;i<n;i++)
            {
                double w[n];

                if(internalObjectWeights == true)
                {
                    // Save current object weights
                    for(j=0;j<n;j++)
                    {
                        w[j] = ow[j];
                    }
                }

                // Save datum current cluster 
                uint32_t curClust = dat[i].clusterID;

                // Transfer point i to cluster l
                CLUSTER_transferPointToCluster2(dat, i, p, c, l);
                
                // Update weight of points in cluster l
                if(internalObjectWeights == true)
                {
                    // Internal computation of objects weights
                    CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                }

                // Recompute SSE in cluster l
                double sse;
                if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                {
                    sse = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, l, fw, ow);
                }
                else
                {
                    sse = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, l, fw, ow);
                }

                if(sse < refSSE)
                {
                    refSSE = sse;
                }
                else
                {
                    // Transfer point i to its previous cluster 
                    CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);

                    // Reset weight of points
                    if(internalObjectWeights == true)
                    {
                        for(j=0;j<n;j++)
                        {
                            ow[j] = w[j];
                        }
                    }
                }
            }
        }

        double finalSSE = 0.0;
        for(l=0;l<k;l++)
        {
            finalSSE += CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, l, fw, ow);
        }

        return finalSSE;
    }
}

static void CLUSTER_assignWeightedDataToCentroids23(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        eMethodType objWeiMet = METHOD_MEDIAN/*METHOD_SILHOUETTE*/;
        eMethodType feaWeiMed = METHOD_DISPERSION;

        // Set convergence variable
        *conv = true;

        for(i=0;i<n;i++)
        {
            // Calculte SSE reference of point i
            double minSSE;
            if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
            {
                minSSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
            }
            else
            {
                minSSE = CLUSTER_computeNkWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
            }

            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;

            for(l=0;l<k;l++)
            {
                if(l != curClust)
                {
                    double ow_tmp[n];
                    double fw_from_tmp[p];
                    double fw_to_tmp[p];

                    // Save current object weights
                    if(internalObjectWeights == true)
                    {
                        for(j=0;j<n;j++)
                        {
                            ow_tmp[j] = ow[j];
                        }
                    }

                    // Save current feature weights
                    if(internalFeatureWeights == true)
                    {
                        for(j=0;j<p;j++)
                        {
                            fw_from_tmp[j] = fw[curClust][j];
                            fw_to_tmp[j] = fw[l][j];
                        }
                    }

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                    // Update object weights in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of object weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                    }

                    // Update feature weights in cluster l
                    if(internalFeatureWeights == true)
                    {
                        // Internal computation of feature weights
                        CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
                    }

                    // Calculate SSE in cluster l
                    double SSE;
                    if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                    {
                        SSE = CLUSTER_computeWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
                    }
                    else
                    {
                        SSE = CLUSTER_computeNkWeightedSSEInCluster(dat, n, p, c, dat[i].clusterID, fw, ow);
                    }

                    if(SSE < minSSE)
                    {
                        minSSE = SSE;
                        curClust = l;

                        // Reset convergence variable
                        *conv = false;
                    }
                    else
                    {
                        // Transfer point i to cluster l
                        CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);

                        // Reset weight of point i
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = ow_tmp[j];
                            }
                        }

                        // Reset feature weights
                        if(internalFeatureWeights == true)
                        {
                            for(j=0;j<p;j++)
                            {
                                fw[curClust][j] = fw_from_tmp[j];
                                fw[l][j] = fw_to_tmp[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

static void CLUSTER_assignWeightedDataToCentroids24(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        eMethodType objWeiMet = METHOD_MEDIAN/*METHOD_SILHOUETTE*/;
        eMethodType feaWeiMed = METHOD_DISPERSION;

        // Set convergence variable
        *conv = true;

        for(i=0;i<n;i++)
        {
            // Calculte SSE reference of point i
            double minSSE;
            if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
            {
                minSSE = CLUSTER_computeWeightedSSE(dat, n, p, c, k, fw, ow);
            }
            else
            {
                minSSE = CLUSTER_computeNkWeightedSSE(dat, n, p, c, k, fw, ow);
            }

            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;

            for(l=0;l<k;l++)
            {
                if(l != curClust)
                {
                    double ow_tmp[n];
                    double fw_from_tmp[p];
                    double fw_to_tmp[p];

                    // Save current object weights
                    if(internalObjectWeights == true)
                    {
                        for(j=0;j<n;j++)
                        {
                            ow_tmp[j] = ow[j];
                        }
                    }

                    // Save current feature weights
                    if(internalFeatureWeights == true)
                    {
                        for(j=0;j<p;j++)
                        {
                            fw_from_tmp[j] = fw[curClust][j];
                            fw_to_tmp[j] = fw[l][j];
                        }
                    }

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                    // Update object weights in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of object weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                    }

                    // Update feature weights in cluster l
                    if(internalFeatureWeights == true)
                    {
                        // Internal computation of feature weights
                        CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
                    }

                    // Calculate SSE in cluster l
                    double SSE;
                    if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                    {
                        SSE = CLUSTER_computeWeightedSSE(dat, n, p, c, k, fw, ow);
                    }
                    else
                    {
                        SSE = CLUSTER_computeNkWeightedSSE(dat, n, p, c, k, fw, ow);
                    }

                    if(SSE < minSSE)
                    {
                        minSSE = SSE;
                        curClust = l;

                        // Reset convergence variable
                        *conv = false;
                    }
                    else
                    {
                        // Transfer point i to cluster l
                        CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);

                        // Reset weight of point i
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = ow_tmp[j];
                            }
                        }

                        // Reset feature weights
                        if(internalFeatureWeights == true)
                        {
                            for(j=0;j<p;j++)
                            {
                                fw[curClust][j] = fw_from_tmp[j];
                                fw[l][j] = fw_to_tmp[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

static void CLUSTER_assignWeightedDataToCentroids25(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double *ow, double **dist, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        eMethodType objWeiMet = METHOD_MEDIAN/*METHOD_SILHOUETTE*/;
        eMethodType feaWeiMed = METHOD_DISPERSION;

        // Set convergence variable
        *conv = true;

        for(i=0;i<n;i++)
        {
            // Calculte SSE reference of point i
            double minSSE = CLUSTER_computeWeightedSSE(dat, n, p, c, k, fw, ow);

            // Save current cluster of point i
            uint32_t curClust = dat[i].clusterID;

            for(l=0;l<k;l++)
            {
                if(l != curClust)
                {
                    double ow_tmp[n];
                    double fw_from_tmp[p];
                    double fw_to_tmp[p];

                    // Save current object weights
                    if(internalObjectWeights == true)
                    {
                        for(j=0;j<n;j++)
                        {
                            ow_tmp[j] = ow[j];
                        }
                    }

                    // Save current feature weights
                    if(internalFeatureWeights == true)
                    {
                        for(j=0;j<p;j++)
                        {
                            fw_from_tmp[j] = fw[curClust][j];
                            fw_to_tmp[j] = fw[l][j];
                        }
                    }

                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);

                    // Update object weights in cluster l
                    if(internalObjectWeights == true)
                    {
                        // Internal computation of object weights
                        CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, ow, objWeiMet, dist);
                    }

                    // Update feature weights in cluster l
                    if(internalFeatureWeights == true)
                    {
                        // Internal computation of feature weights
                        CLUSTER_computeFeatureWeights(dat, n, p, c, k, fw, feaWeiMed);
                    }

                    // Calculate SSE in cluster l
                    double SSE = CLUSTER_computeWeightedSSE(dat, n, p, c, k, fw, ow);

                    if(SSE < minSSE)
                    {
                        minSSE = SSE;
                        curClust = l;

                        // Reset convergence variable
                        *conv = false;
                    }
                    else
                    {
                        // Transfer point i to cluster l
                        CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);

                        // Reset weight of point i
                        if(internalObjectWeights == true)
                        {
                            for(j=0;j<n;j++)
                            {
                                ow[j] = ow_tmp[j];
                            }
                        }

                        // Reset feature weights
                        if(internalFeatureWeights == true)
                        {
                            for(j=0;j<p;j++)
                            {
                                fw[curClust][j] = fw_from_tmp[j];
                                fw[l][j] = fw_to_tmp[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

static void CLUSTER_assignWeightedDataToCentroids26(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, double fw[k][p], bool internalObjectWeights, double ow[n][k], bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1;
    }
    else
    {
        uint64_t i;
        uint32_t l;

        // Set convergence variable
        *conv = true;

        for(i=0;i<n;i++)
        {
            double minDist;
            double distClus;
            uint32_t minK;

            for(l=0;l<k;l++)
            {
                //WRN("Dist for dat%ld (ow[%d] = %lf) to c%d (nbData %ld)", i, l, ow[i][l], l, c[l].nbData);
                double dist;

                if(internalFeatureWeights || (internalFeatureWeights && internalObjectWeights))
                {
                    // Calculate squared Euclidean distance
                    dist = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i][l]);
                }
                else
                {
                    // Calculate squared Euclidean distance
                    dist = CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN, (double *)&(fw[l]), ow[i][l]) / (double) c[l].nbData;
                }

                //SAY("Dist with c%d = %lf (c nbData = %ld)", l, dist, c[l].nbData);

                //SAY("Mindist = %lf, dist = %lf", minDist, dist);
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
                //SAY("Moved to c%d", minK);
                c[dat[i].clusterID].nbData--; // Decrease previous cluster data number
                dat[i].clusterID = minK; // Assign data to cluster
                c[dat[i].clusterID].nbData++; // Increase new cluster data number
                // Reset convergence variable
                *conv = false;
            }
        }
    }
}

static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d)
{
    if(dat == NULL || p < 1 || c == NULL)
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
}

static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow)
{
    if(dat == NULL || p < 1 || c == NULL || fw == NULL)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
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
}

static double CLUSTER_computeNormalizedSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow, double sd[p])
{
    if(dat == NULL || p < 1 || c == NULL || fw == NULL)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
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
                        //WRN("dist = %lf", dist);
                        double tmp = fw[j]*(pow((dat->dim[j] - c->centroid[j]), 2.0) / pow(sd[j],2.0)); // Apply feature weights

                        if(isnan(tmp) || isinf(tmp))
                        {
                            dist += 0.0;
                        }
                        else
                        {
                            dist += tmp;
                        }

                        //SAY("dist = %lf, tmp = %lf, ow = %lf, fw[%ld] = %lf, sd2[%ld] = %lf, d = %lf", dist, tmp, ow, j, fw[j], j,  pow(sd[j],2.0), pow((dat->dim[j] - c->centroid[j]), 2.0));
                    }
                }
                break;
            case DISTANCE_OTHER:
                {
                    WRN("Not implemented yet");
                    dist = -1.0;
                }
                break;
        }

        return (ow*dist); // Apply object weight
    }
}

static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
    }
    else
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
}

static void CLUSTER_computeCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 0)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i,j;

        // Reset each centroid dimensions to 0
        for(j=0;j<p;j++)
            c[k].centroid[j] = 0.0;

        // Compute the new centroid dimensions
        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                for(j=0;j<p;j++)
                {
                    c[k].centroid[j] += (dat[i].dim[j]/(double)c[k].nbData);   
                }
            }
        }    
    }
}

static void CLUSTER_transferPointToCluster(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK)
{
    if(dat == NULL || indN < 0 || p < 1 || c == NULL || indK < 0)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t j;

        uint32_t prevClust = dat[indN].clusterID; // Retreive datum previous cluster
        c[prevClust].nbData--; // Decrease previous cluster data number
        dat[indN].clusterID = indK; // Assign data to cluster
        c[indK].nbData++; // Increase new cluster data number

        // Compute the new centroid dimensions
        for(j=0;j<p;j++)
        {
            c[prevClust].centroid[j] = ((c[prevClust].centroid[j] * (double)(c[prevClust].nbData + 1)) - dat[indN].dim[j])/(double)c[prevClust].nbData;   
            c[indK].centroid[j] = ((c[indK].centroid[j] * (double)(c[indK].nbData - 1)) + dat[indN].dim[j])/(double)c[indK].nbData;
            /*c[prevClust].centroid[j] += (c[prevClust].centroid[j] - dat[indN].dim[j])/(double)(c[prevClust].nbData + 1);
            c[indK].centroid[j] += (dat[indN].dim[j] - c[indK].centroid[j])/(double)c[indK].nbData;*/
        }
    }
}

static void CLUSTER_transferPointToCluster2(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK)
{
    if(dat == NULL || indN < 0 || p < 1 || c == NULL || indK < 0)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t j;

        uint32_t prevClust = dat[indN].clusterID; // Retreive datum previous cluster
        c[prevClust].nbData--; // Decrease previous cluster data number
        dat[indN].clusterID = indK; // Assign data to cluster
        c[indK].nbData++; // Increase new cluster data number

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
            /*c[prevClust].centroid[j] += (c[prevClust].centroid[j] - dat[indN].dim[j])/(double)(c[prevClust].nbData + 1);
            c[indK].centroid[j] += (dat[indN].dim[j] - c[indK].centroid[j])/(double)c[indK].nbData;*/
        }
    }
}

static double CLUSTER_computeSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -2.0;
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k], dist[n][n];
        uint64_t i,j;
        uint32_t l;

        /*double **dist = malloc(n*sizeof(double *));
          for(i=0;i<n;i++)
          dist[i] = malloc(n*sizeof(double));*/

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        // Create the distance matrix of i vs j
        for(i=0;i<n;i++)
        {
            for (j=0;j<n;j++)
            {
                dist[i][j] = CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);
                //SAY("dist[%ld][%ld] = %lf", i, j, dist[i][j]);
            }
        }

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;
            //SAY("i = %ld, s[i] = %lf, b[i] = %lf, a[i] = %lf, dat[i].clusterID = %d, c[dat[i].clusterID].nbData = %ld", i, s[i], b[i], a[i], dat[i].clusterID, c[dat[i].clusterID].nbData);
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        for(l=0;l<k;l++)
        {
            //SAY("sk[%d] = %lf", l, sk[l]);
            if (!isnan(sk[l]))
                sil += sk[l];
        }

        /*for(i=0;i<n;i++)
          free(dist[i]);
          free(dist);*/

        return (sil/k);
    }
}

static double CLUSTER_computeSilhouette2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double dist[n][n])
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

        /*double **dist = malloc(n*sizeof(double *));
          for(i=0;i<n;i++)
          dist[i] = malloc(n*sizeof(double));*/

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;
            //SAY("i = %ld, s[i] = %lf, b[i] = %lf, a[i] = %lf, dat[i].clusterID = %d, c[dat[i].clusterID].nbData = %ld", i, s[i], b[i], a[i], dat[i].clusterID, c[dat[i].clusterID].nbData);
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        for(l=0;l<k;l++)
        {
            //SAY("sk[%d] = %lf", l, sk[l]);
            if (!isnan(sk[l]))
                sil += sk[l];
        }

        /*for(i=0;i<n;i++)
          free(dist[i]);
          free(dist);*/

        return (sil/k);
    }
}

static double CLUSTER_computeSilhouette3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
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

        /*double **dist = malloc(n*sizeof(double *));
          for(i=0;i<n;i++)
          dist[i] = malloc(n*sizeof(double));*/

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
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];
            }

            if((c[dat[i].clusterID].nbData - 1) == 0)
                a[i] = 0.0;
            else
                a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;
            //SAY("i = %ld, s[i] = %lf, b[i] = %lf, a[i] = %lf, dat[i].clusterID = %d, c[dat[i].clusterID].nbData = %ld", i, s[i], b[i], a[i], dat[i].clusterID, c[dat[i].clusterID].nbData);
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        for(l=0;l<k;l++)
        {
            //SAY("sk[%d] = %lf", l, sk[l]);
            if (!isnan(sk[l]))
                sil += sk[l];
        }

        /*for(i=0;i<n;i++)
          free(dist[i]);
          free(dist);*/

        return (sil/k);
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

static double CLUSTER_computeWeightedSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -2.0;
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k], dist[n][n];
        uint64_t i,j;
        uint32_t l;

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        // Create the distance matrix of i vs j
        for(i=0;i<n;i++)
        {
            for (j=0;j<n;j++)
            {
                dist[i][j] = CLUSTER_computeDistanceWeightedPointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN, (double *)&(fw[dat[i].clusterID]), ow[i]);
                //SAY("dist2[%ld][%ld] = %lf", i, j, dist[i][j]);
            }
        }

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;
            //SAY("i = %ld, s[i] = %lf, b[i] = %lf, a[i] = %lf, dat[i].clusterID = %d, c[dat[i].clusterID].nbData = %ld", i, s[i], b[i], a[i], dat[i].clusterID, c[dat[i].clusterID].nbData);
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        for(l=0;l<k;l++)
        {
            //SAY("sk[%d] = %lf", l, sk[l]);
            if (!isnan(sk[l]))
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

static double CLUSTER_computeDistanceWeightedPointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d, double *fw, double iOw)
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
                    {
                        dist += fw[j]*pow(((iDat->dim[j]) - (jDat->dim[j])), 2.0); // Apply features weights
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

        return iOw*sqrt(dist); // Apply i object weight
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

static double CLUSTER_computeTSS2(data *dat, uint64_t n, uint64_t p)
{
    if(dat == NULL || n < 2 || p < 1)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        uint64_t i, j;
        double TSS = 0.0;
        double	sx=0, sx2 = 0, var = 0, tmp = 0;

        for(j=0;j<p;j++)
        {
            sx = 0.0;
            sx2 = 0.0;

            for(i=0;i<n;i++)
            {
                tmp = dat[i].dim[j];
                sx = sx + tmp;
                sx2 = sx2 + (tmp * tmp);
            }

            var = sx2 - ((sx * sx) / (double) n);
            TSS = TSS + var;
        }

        return TSS;
    }
}

static double CLUSTER_computeVRC(double TSS, double SSE, uint64_t n, uint32_t k)
{
    return ((TSS * (double)(n - k)) / (SSE * (double)(k - 1)));   
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

    return ((SSB / SSE) * ((n - k) / (k - 1)));   
}

static double CLUSTER_computeCH(double TSS, double SSE, uint64_t n, uint32_t k)
{
    double tmp = SSE / (double)(n - k);
    return ( (TSS - SSE) / (double)(k - 1)) / tmp; 
}

static void CLUSTER_initObjectWeights(double *ow, uint64_t n)
{
    uint64_t i;
    for(i=0;i<n;i++)
        ow[i] = 1.0;
}

static void CLUSTER_initObjectWeights2(uint64_t n, uint32_t k, double ow[n][k])
{
    uint64_t i;
    uint32_t l;
    for(i=0;i<n;i++)
        for(l=0;l<k;l++)
            ow[i][l] = 1.0;
}

static void CLUSTER_initFeatureWeights(uint32_t k, uint64_t p, double fw[k][p])
{
    uint64_t j;
    uint32_t l;
    for(l=0;l<k;l++)
    {
        for(j=0;j<p;j++)
        {
            fw[l][j] = 1.0;
        }
    }
}

static void CLUSTER_computeFeatureWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], eMethodType m)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || fw == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        switch(m)
        {
            default:
            case METHOD_DISPERSION :
                {
                    CLUSTER_computeFeatureWeightsViaDispersion(dat, n, p, c, k, fw, 2); // Using L2-norm
                }
                break;
            case METHOD_OTHER:
                {
                    WRN("Not implemented yet");
                }
                break;
        }
    }
}

static void CLUSTER_computeFeatureWeightsViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], uint8_t norm)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || fw == NULL)
    {
        ERR("Bad parameter");
    }
    else
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
                    fw[l][j] = 1.0;
                }
                else
                {
                    // The sum of features weights as to be equal to the number of features
                    fw[l][j] = (1 / tmp[j]) * p;
                    if(isinf(fw[l][j]))
                        fw[l][j] = 1.0; 
                    //SAY("Weight[%d][%ld] = %lf", l, j, fw[l][j]);
                }
            }
        }
    }
}

static double CLUSTER_computeFeatureDispersion(data *dat, uint64_t p, cluster *c, uint8_t norm)
{
    if(dat == NULL || p < 0 || c == NULL || norm < 0)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        return pow(dat->dim[p] - c->centroid[p], norm); 
    }
}

static void CLUSTER_computeObjectWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        switch(m)
        {
            default:
            case METHOD_SILHOUETTE :
                {
                    CLUSTER_computeObjectWeightsViaSilhouette(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_SSE :
                {
                    CLUSTER_computeObjectWeightsViaSSE(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_ABOD :
                {
                    CLUSTER_computeObjectWeightsViaABOD(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_AVERAGE_SSE :
                {
                    CLUSTER_computeObjectWeightsViaAverageSSE(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_MEDIAN :
                {
                    //CLUSTER_computeObjectWeightsViaMedian(dat, n, p, c, k, ow);
                    //WRN("Simple method");
                    CLUSTER_computeObjectWeightsViaMedian2(dat, n, p, c, k, ow);
                    //WRN("Complex method");
                    //CLUSTER_computeObjectWeightsViaMedian3(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_OTHER:
                {
                    WRN("Not implemented yet");
                }
                break;
        }
    }
}

static void CLUSTER_computeObjectWeights2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m, double dist[n][n])
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        switch(m)
        {
            default:
            case METHOD_SILHOUETTE :
                {
                    CLUSTER_computeObjectWeightsViaSilhouette2(dat, n, p, c, k, ow, dist);
                }
                break;
            case METHOD_SSE :
                {
                    CLUSTER_computeObjectWeightsViaSSE(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_ABOD :
                {
                    CLUSTER_computeObjectWeightsViaABOD(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_AVERAGE_SSE :
                {
                    CLUSTER_computeObjectWeightsViaAverageSSE(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_MEDIAN :
                {
                    //CLUSTER_computeObjectWeightsViaMedian(dat, n, p, c, k, ow);
                    //WRN("Simple method");
                    //CLUSTER_computeObjectWeightsViaMedian2(dat, n, p, c, k, ow);
                    //WRN("Complex method");
                    CLUSTER_computeObjectWeightsViaMedian3(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_OTHER:
                {
                    WRN("Not implemented yet");
                }
                break;
        }
    }
}

static void CLUSTER_computeObjectWeights3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, eMethodType m, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        switch(m)
        {
            default:
            case METHOD_SILHOUETTE :
                {
                    //CLUSTER_computeObjectWeightsViaSilhouette3(dat, n, p, c, k, ow, dist);
                    CLUSTER_computeObjectWeightsViaSilhouette4(dat, n, p, c, k, ow, dist);
                }
                break;
            case METHOD_SSE :
                {
                    CLUSTER_computeObjectWeightsViaSSE(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_ABOD :
                {
                    CLUSTER_computeObjectWeightsViaABOD(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_AVERAGE_SSE :
                {
                    CLUSTER_computeObjectWeightsViaAverageSSE(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_MEDIAN :
                {
                    //CLUSTER_computeObjectWeightsViaMedian(dat, n, p, c, k, ow);
                    //WRN("Simple method");
                    CLUSTER_computeObjectWeightsViaMedian2(dat, n, p, c, k, ow);
                    //WRN("Complex method");
                    //CLUSTER_computeObjectWeightsViaMedian3(dat, n, p, c, k, ow);
                }
                break;
            case METHOD_OTHER:
                {
                    WRN("Not implemented yet");
                }
                break;
        }
    }
}

static void CLUSTER_computeObjectWeights4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double ow[n][k], eMethodType met, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i;
        uint32_t l;

        for(i=0;i<n;i++)
        {
            // Save point i current cluster
            uint32_t curClust = dat[i].clusterID;
            for(l=0;l<k;l++)
            {
                //SAY("From c%d to c%d", curClust, l);
                /*if(l != curClust)
                {*/
                    //SAY("Transfered");
                    // Transfer point i to cluster l
                    CLUSTER_transferPointToCluster2(dat, i, p, c, l);
                /*}*/

                switch(met)
                {
                    default:
                    case METHOD_SILHOUETTE :
                        {
                            //ow[i][l] = CLUSTER_computeObjectWeightInClusterViaSilhouette(dat, n, i, p, c, k, l, dist);
                            ow[i][l] = CLUSTER_computeObjectWeightInClusterViaSilhouette2(dat, n, i, p, c, k, l, dist);
                            //WRN("ow[%ld][%d] = %lf(cluster %d, nbData %ld)", i, l, ow[i][l], dat[i].clusterID, c[dat[i].clusterID].nbData);
                        }
                        break;
                    case METHOD_MEDIAN :
                        {
                            //ow[i][l] = CLUSTER_computeObjectWeightInClusterViaMedian(dat, n, i, p, c, l);
                            ow[i][l] = CLUSTER_computeObjectWeightInClusterViaMedian2(dat, n, i, p, c, l);
                            //WRN("ow[%ld][%d] = %lf(cluster %d, nbData %ld)", i, l, ow[i][l], dat[i].clusterID, c[dat[i].clusterID].nbData);
                        }
                        break;
                    case METHOD_OTHER:
                        {
                            WRN("Not implemented yet");
                        }
                        break;
                }
                //WRN("ow[%ld][%d] = %lf (belongs to c%d)", i, l, ow[i][l], curClust);
            }

            /*if(dat[i].clusterID != curClust)
            {*/
                // Move point i to its original cluster 
                CLUSTER_transferPointToCluster2(dat, i, p, c, curClust);
           /* }*/
        }
    }
}

static void CLUSTER_computeObjectWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double *ow, eMethodType m, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        switch(m)
        {
            default:
            case METHOD_SILHOUETTE :
                {
                    CLUSTER_computeObjectWeightsInClusterViaSilhouette(dat, n, p, c, k, indK, ow, dist);
                }
                break;
            case METHOD_MEDIAN :
                {
                    CLUSTER_computeObjectWeightsInClusterViaMedian(dat, n, p, c, indK, ow);
                }
                break;
            case METHOD_OTHER:
                {
                    WRN("Not implemented yet");
                }
                break;
        }
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k], dist[n][n];
        uint64_t i,j;
        uint32_t l;
        //double totWeightSum = 0.0; // Total sum of weights

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        // Create the distance matrix of i vs j
        for(i=0;i<n;i++)
            for (j=0;j<n;j++)
                dist[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;

            //SAY("s[%ld] = %lf", i, s[i]);

            if(s[i] < 0 || s[i] > 0)
                s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
            else // si = 0
                s[i] = 0.5;

            // Calculate sum of s[i] per cluster 
            sk[dat[i].clusterID] += s[i];
            //totWeightSum += s[i];
            
            if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                ow[i] = s[i];
            }
            //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
        }

        // Calculate object weights
        /*for(i=0;i<n;i++)
        {
            // The sum of weights per cluster has to be equal to the number of data per cluster
            if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
            {
                //ow[i] = (s[i]/sk[dat[i].clusterID])*c[dat[i].clusterID].nbData;
                //ow[i] = (s[i]/totWeightSum) * (double) n;
                ow[i] = s[i];
            }
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }*/
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, double dist[n][n])
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k];
        uint64_t i,j;
        uint32_t l;
        //double totWeightSum = 0.0; // Total sum of weights

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;

            //SAY("s[%ld] = %lf", i, s[i]);

            if(s[i] < 0 || s[i] > 0)
                s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
            else // si = 0
                s[i] = 0.5;

            // Calculate sum of s[i] per cluster 
            sk[dat[i].clusterID] += s[i];
            //totWeightSum += s[i];
            
            if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                ow[i] = s[i];
            }
            //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
        }

        // Calculate object weights
        /*for(i=0;i<n;i++)
        {
            // The sum of weights per cluster has to be equal to the number of data per cluster
            if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
            {
                //ow[i] = (s[i]/sk[dat[i].clusterID])*c[dat[i].clusterID].nbData;
                //ow[i] = (s[i]/totWeightSum) * (double) n;
                ow[i] = s[i];
            }
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }*/
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k];
        uint64_t i,j;
        uint32_t l;
        //double totWeightSum = 0.0; // Total sum of weights

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
                    d += dist[i][j];

                if((c[dat[i].clusterID].nbData - 1) == 0)
                    a[i] = 0.0;
                else
                    a[i] = d/(double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            s[i] = (b[i] != a[i]) ?  (b[i] - a[i]) / fmax(a[i], b[i]) : 0.0;

            //SAY("s[%ld] = %lf", i, s[i]);

            if(s[i] < 0 || s[i] > 0)
                s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
            else // si = 0
                s[i] = 0.5;

            // Calculate sum of s[i] per cluster 
            sk[dat[i].clusterID] += s[i];
            //totWeightSum += s[i];
            
            if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                ow[i] = s[i];
            }
            //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
        }

        // Calculate object weights
        for(i=0;i<n;i++)
        {
            // The sum of weights per cluster has to be equal to the number of data per cluster
            if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
            {
                ow[i] = (s[i] / sk[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;
                //ow[i] = (s[i]/totWeightSum) * (double) n;
                //ow[i] = s[i];
            }
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette4(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k];
        uint64_t i,j;
        uint32_t l;
        //double totWeightSum = 0.0; // Total sum of weights

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if(j != i && dat[j].clusterID == dat[i].clusterID)
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

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
                distCluster[l] = 0.0;

            for(j=0;j<n;j++)
                if(dat[j].clusterID != dat[i].clusterID)
                    distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
            b[i] = 1.0e20;
            for(l=0;l<k;l++)
                if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                    b[i] = distCluster[l];

            // Calculate s[i]
            if(c[dat[i].clusterID].nbData == 1)
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
            sk[dat[i].clusterID] += s[i];
            //totWeightSum += s[i];
            
            /*if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                ow[i] = s[i];
            }*/
            //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
        }

        // Calculate object weights
        for(i=0;i<n;i++)
        {
            // The sum of weights per cluster has to be equal to the number of data per cluster
            /*if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
            {*/
                ow[i] = (s[i] / sk[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;

                if(isnan(ow[i]) || isinf(ow[i]))
                {
                    //ERR("ow = %lf, s = %lf, sk = %lf, cNbDat = %ld", ow[i], s[i], sk[dat[i].clusterID], c[dat[i].clusterID].nbData);
                    ow[i] = 1.0;
                }
                //ow[i] = (s[i]/totWeightSum) * (double) n;
                //ow[i] = s[i];
            //}
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double *ow, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || indK < 0 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k];
        uint64_t i,j;
        uint32_t l;
        //double totWeightSum = 0.0; // Total sum of weights

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == indK)
            {
                // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
                double d = 0.0;
                for(j=0;j<n;j++)
                {
                    if(j != i && dat[j].clusterID == dat[i].clusterID)
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

                // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
                for(l=0;l<k;l++)
                    distCluster[l] = 0.0;

                for(j=0;j<n;j++)
                    if(dat[j].clusterID != dat[i].clusterID)
                        distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
                b[i] = 1.0e20;
                for(l=0;l<k;l++)
                    if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                        b[i] = distCluster[l];

                // Calculate s[i]
                if(c[dat[i].clusterID].nbData == 1)
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
                sk[dat[i].clusterID] += s[i];
                //totWeightSum += s[i];

                /*if(c[dat[i].clusterID].nbData == 1)
                  {
                  ow[i] = 1.0;
                  }
                  else
                  {
                  ow[i] = s[i];
                  }*/
                //SAY("ow[%ld] = %lf, cluster %d", i, ow[i], dat[i].clusterID);
            }
        }

        // Calculate object weights
        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == indK)
            {
                // The sum of weights per cluster has to be equal to the number of data per cluster
                /*if(c[dat[i].clusterID].nbData == 1)
                  ow[i] = 1.0;
                  else
                  {*/
                ow[i] = (s[i] / sk[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;

                if(isnan(ow[i]) || isinf(ow[i]))
                {
                    //ERR("ow = %lf, s = %lf, sk = %lf, cNbDat = %ld", ow[i], s[i], sk[dat[i].clusterID], c[dat[i].clusterID].nbData);
                    ow[i] = 1.0;
                }
                //ow[i] = (s[i]/totWeightSum) * (double) n;
                //ow[i] = s[i];
                //}
                //SAY("ow[%ld] = %lf", i, ow[i]);
            }
        }
    }
}

static double CLUSTER_computeObjectWeightInClusterViaSilhouette(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || indK < 0 || indN < 0 || indN >= n)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double a[n], b[n], s[n], sk = 0.0, ow_tmp, distCluster[k];
        uint64_t i,j;
        uint32_t l;

        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == indK)
            {
                // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
                double d = 0.0;
                for(j=0;j<n;j++)
                {
                    if(j != i && dat[j].clusterID == dat[i].clusterID)
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

                // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
                for(l=0;l<k;l++)
                    distCluster[l] = 0.0;

                for(j=0;j<n;j++)
                    if(dat[j].clusterID != dat[i].clusterID)
                        distCluster[dat[j].clusterID] += (dist[i][j]/c[dat[j].clusterID].nbData);
                b[i] = 1.0e20;
                for(l=0;l<k;l++)
                    if(l != dat[i].clusterID && distCluster[l] != 0 && distCluster[l] < b[i])
                        b[i] = distCluster[l];

                // Calculate s[i]
                if(c[dat[i].clusterID].nbData == 1)
                {
                    s[i] = 0;
                }
                else
                {
                    s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
                }

                if(s[i] < 0 || s[i] > 0)
                    s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
                else // si = 0
                    s[i] = 0.5;

                // Calculate sum of s[i] per cluster 
                sk += s[i];
                //SAY("skTmp = %lf (s[%ld] = %lf)", sk, i, s[i]);
            }
        }

        // Calculate object weights
        // The sum of weights per cluster has to be equal to the number of data per cluster
        //SAY("s[%ld] = %lf, sk = %lf, c%d nbData = %ld", indN, s[indN], sk, indK, c[indK].nbData);
        ow_tmp = (s[indN] / sk) * (double) c[indK].nbData;

        if(isnan(ow_tmp) || isinf(ow_tmp))
        {
            ow_tmp = 1.0;
        }

        return ow_tmp;
    }
}

static double CLUSTER_computeObjectWeightInClusterViaSilhouette2(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || indK < 0 || indN < 0 || indN >= n)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double a, b, s, ow_tmp, distCluster[k];
        uint64_t j;
        uint32_t l;

        // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
        double d = 0.0;
        for(j=0;j<n;j++)
        {
            if(j != indN && dat[j].clusterID == dat[indN].clusterID)
            {
                d += dist[indN][j];
            }
        }

        if((c[dat[indN].clusterID].nbData - 1) == 0)
        {
            a = 0.0;
        }
        else
        {
            a = d / (double)(c[dat[indN].clusterID].nbData - 1);
        }

        // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
        for(l=0;l<k;l++)
            distCluster[l] = 0.0;

        for(j=0;j<n;j++)
            if(dat[j].clusterID != dat[indN].clusterID)
                distCluster[dat[j].clusterID] += (dist[indN][j]/c[dat[j].clusterID].nbData);
        b = 1.0e20;
        for(l=0;l<k;l++)
            if(l != dat[indN].clusterID && distCluster[l] != 0 && distCluster[l] < b)
                b = distCluster[l];

        // Calculate s[i]
        if(c[dat[indN].clusterID].nbData == 1)
        {
            s = 0;
        }
        else
        {
            s = (b != a) ?  ((b - a) / fmax(a, b)) : 0.0;
        }

        if(s < 0 || s > 0)
            s = 1 - ((s + 1) / 2); // Rescale silhouette to 0-1 
        else // si = 0
            s = 0.5;

        //SAY("skTmp = %lf (s[%ld] = %lf)", sk, i, s[i]);

        // Calculate object weights
        // The sum of weights per cluster has to be equal to the number of data per cluster
        //SAY("s[%ld] = %lf, sk = %lf, c%d nbData = %ld", indN, s[indN], sk, indK, c[indK].nbData);
        ow_tmp = s;

        if(isnan(ow_tmp) || isinf(ow_tmp))
        {
            ow_tmp = 1.0;
        }

        return ow_tmp;
    }
}

static void CLUSTER_computeObjectWeightsViaSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double squaredDist[n]; // Squared distance from points to clusters
        double sumSSE[k]; // SSE per cluster
        double totSSE = 0.0; // Total SSE

        for(l=0;l<k;l++)
        {
            sumSSE[l] = 0.0;
        }

        // Compute SSE per cluster
        for(i=0;i<n;i++)
        {
            squaredDist[i] = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN);
            sumSSE[dat[i].clusterID] += squaredDist[i];
            totSSE += squaredDist[i];
        }

        // Compute objects weights
        for(i=0;i<n;i++)
        {
            // The sum of weights per cluster has to be equal to the number of data per cluster
            if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                //ow[i] = (squaredDist[i] / sumSSE[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;
                //ow[i] = (squaredDist[i] / totSSE) /** (double) n*/;
                ow[i] = squaredDist[i];
            }
            //SAY("ow[%ld] = %lf, c %d", i, ow[i], dat[i].clusterID);
        }
    }
}

static void CLUSTER_computeObjectWeightsViaABOD(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i, j, m;
        uint32_t l;

        for(l=0;l<k;l++)
        {
            for(i=0;i<n;i++)
            {
                // Retrieve all data in the i datum cluster
                double datClu[c[dat[i].clusterID].nbData - 1][p];
                uint64_t o = 0;
                for(m=0;m<n;m++)
                {
                    if(dat[m].clusterID == dat[i].clusterID && m != i)
                    {
                        for(j=0;j<p;j++)
                        {
                            datClu[o][j] = dat[m].dim[j];
                        }
                        o++;
                    }
                }
                
                for(m=0;m<(c[dat[i].clusterID].nbData - 1);m++)
                {
                    for(o=0;o<(c[dat[i].clusterID].nbData - 1);o++)
                    {
                        if(o != m)
                        {
                            double len = 0.0;
                            double mi = 0.0;
                            double io = 0.0;
                            for(j=0;j<p;j++)
                            {
                                len += (dat[i].dim[j] - datClu[m][j]) * (datClu[o][j] - dat[i].dim[j]);
                                mi += pow((dat[i].dim[j] - datClu[m][j]), 2.0);
                                io += pow((datClu[o][j] - dat[i].dim[j]), 2.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void CLUSTER_computeObjectWeightsViaAverageSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double avgSSE[k]; // Average SSE per cluster
        //double squaredDist[n];
        double dist[n];
        double sumWeights[k];

        for(l=0;l<k;l++)
        {
            avgSSE[l] = 0.0;
            sumWeights[l] = 0.0;
        }

        // Compute average SSE per cluster
        for(i=0;i<n;i++)
        {
            /*squaredDist[i] = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN);
              avgSSE[dat[i].clusterID] += squaredDist[i] / (double) c[dat[i].clusterID].nbData;*/
            dist[i] = CLUSTER_computeDistancePointToPoint(&(dat[i]), &(c[dat[i].clusterID]), p, DISTANCE_EUCLIDEAN);
            avgSSE[dat[i].clusterID] += dist[i] / (double) c[dat[i].clusterID].nbData;
        }

        // Compute objects weights
        for(i=0;i<n;i++)
        {
            //SAY("ow[%ld] = %lf", i, squaredDist[i] / avgSSE[dat[i].clusterID]);
            //ow[i] = squaredDist[i] / avgSSE[dat[i].clusterID]; // Tmp
            ow[i] = dist[i] / avgSSE[dat[i].clusterID];
            sumWeights[dat[i].clusterID] += ow[i];
        }

        // The sum of weights per cluster has to be equal to the number of data per cluster
        for(i=0;i<n;i++)
        {
            if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
                ow[i] = (ow[i] / sumWeights[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }
    }
}

static void CLUSTER_computeObjectWeightsViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i;
        uint32_t l;
        double median[k]; // Median per cluster
        double dist[n]; // Distance point to cluster
        double sumWeights[k]; // Sum of weights per cluster

        // Compute the median for each cluster
        for(l=0;l<k;l++)
        {
            sumWeights[l] = 0.0;

            double distPerCluster[c[l].nbData];
            uint64_t j = 0;

            for(i=0;i<n;i++)
            {
                if(dat[i].clusterID == l)
                {
                    distPerCluster[j] = CLUSTER_computeDistancePointToPoint(&(dat[i]), &(c[dat[i].clusterID]), p, DISTANCE_EUCLIDEAN);
                    dist[i] = distPerCluster[j];
                    j++;
                }
            }

            double tmp;
            for(i=0;i<c[l].nbData;i++)
            {
                for(j=i+1;j<c[l].nbData;j++)
                {
                    if(distPerCluster[i] > distPerCluster[j])
                    {
                        tmp = distPerCluster[i];
                        distPerCluster[i] = distPerCluster[j];
                        distPerCluster[j] = tmp;
                    }
                }
            }

            // Take the {(n + 1) ÷ 2}th value as median
            median[l] = distPerCluster[((c[l].nbData+1)/2)];
        }
        
        // Compute objects weights
        for(i=0;i<n;i++)
        {
            ow[i] = dist[i] / median[dat[i].clusterID];
            sumWeights[dat[i].clusterID] += ow[i];
            //SAY("ow[%ld] = %lf", i, ow[i]); 
        }

        // The sum of weights per cluster has to be equal to the number of data per cluster
        for(i=0;i<n;i++)
        {
            if(c[dat[i].clusterID].nbData == 1)
                ow[i] = 1.0;
            else
                ow[i] = (ow[i] / sumWeights[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;
            //SAY("ow[%ld] = %lf", i, ow[i]);
        }
    }
}

static void CLUSTER_computeObjectWeightsViaMedian2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double median[k][p]; // Median per cluster
        double w[n]; // Tmp weights
        double sumWeights[k]; // Sum of weights per clusters
        double totWeightSum = 0.0; // Total sum of weights
        
        for(l=0;l<k;l++)
        {
            double datPerCluster[c[l].nbData][p];
            uint64_t m = 0;

            for(i=0;i<n;i++)
            {
                if(dat[i].clusterID == l)
                {
                    for(j=0;j<p;j++)
                    {
                        datPerCluster[m][j] = dat[i].dim[j];
                    }
                    m++;
                }
            }

            // Sort each dimension in the ascending way
            double tmp;
            for(i=0;i<c[l].nbData;i++)
            {
                for(m=i+1;m<c[l].nbData;m++)
                {
                    for(j=0;j<p;j++)
                    {
                        if(datPerCluster[i][j] > datPerCluster[m][j])
                        {
                            tmp = datPerCluster[i][j];
                            datPerCluster[i][j] = datPerCluster[m][j];
                            datPerCluster[m][j] = tmp;
                        }
                    }
                }
            }

            // Compute the median
            for(j=0;j<p;j++)
            {
                if(!(c[l].nbData % 2))
                {
                    median[l][j] = (datPerCluster[(c[l].nbData / 2)][j] + datPerCluster[(c[l].nbData / 2) - 1][j]) / 2;
                }
                else
                {
                    median[l][j] = datPerCluster[((c[l].nbData + 1) / 2) - 1][j];
                }
                //SAY("median[%d][%ld] = %lf", l, j, median[l][j]);
            }
        }

        for(l=0;l<k;l++)
        {
            sumWeights[l] = 0.0;
        }

        // Compute tmp weights
        for(i=0;i<n;i++)
        {
            // Initialize tmp weights
            w[i] = 0.0;

            for(j=0;j<p;j++)
            {
                w[i] += pow((dat[i].dim[j] - median[dat[i].clusterID][j]), 2.0); 
            }

            sumWeights[dat[i].clusterID] += w[i];
            totWeightSum += w[i];
        }

        // Compute objects weights
        for(i=0;i<n;i++)
        {
            if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                ow[i] = (w[i] / sumWeights[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;
                //ow[i] = (w[i] / totWeightSum) /** (double) n*/;

                //ow[i] = (w[i] / sumWeights[dat[i].clusterID]);
                //ow[i] = w[i];
            }
            //SAY("ow[%ld] = %lf, c = %d", i, ow[i], dat[i].clusterID);
        }
    }
}

static int cmpfunc(const void * a, const void * b)
{
    return (*(double*)a > *(double*)b) ? 1 : (*(double*)a < *(double*)b) ? -1:0 ;
}

static void CLUSTER_computeObjectWeightsInClusterViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 0 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double median[p]; // Median per cluster
        double w[n]; // Tmp weights
        double sumWeights = 0.0; // Sum of weights in cluster k

        double datPerCluster[p][c[k].nbData];
        
        for(j=0;j<p;j++)
        {
            uint64_t m = 0;
            for(i=0;i<n;i++)
            {
                if(dat[i].clusterID == k)
                {
                    datPerCluster[j][m] = dat[i].dim[j];
                    m++;
                }
            }
        }

        /*printf("BEF : ");
        for(j=0;j<c[k].nbData;j++)
        {
            printf("dim[1][%ld] = %lf, ", j, datPerCluster[0][j]);
        }
        SAY("");*/

        for(j=0;j<p;j++)
        {
            qsort((double *)&(datPerCluster[j]), c[k].nbData, sizeof(double), cmpfunc);
        }

        /*printf("AFT : ");
        for(j=0;j<c[k].nbData;j++)
        {
            printf("dim[1][%ld] = %lf, ", j, datPerCluster[0][j]);
        }
        SAY("");*/

        // Compute the median
        for(j=0;j<p;j++)
        {
            if(!(c[k].nbData % 2))
            {
                median[j] = (datPerCluster[j][(c[k].nbData / 2)] + datPerCluster[j][(c[k].nbData / 2) - 1]) / 2;
            }
            else
            {
                median[j] = datPerCluster[j][((c[k].nbData + 1) / 2) - 1];
            }
            //SAY("median[%d][%ld] = %lf", l, j, median[l][j]);
        }


        // Compute tmp weights
        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                // Initialize tmp weights
                w[i] = 0.0;

                for(j=0;j<p;j++)
                {
                    w[i] += pow((dat[i].dim[j] - median[j]), 2.0); 
                }

                sumWeights += w[i];
            }
        }

        // Compute objects weights
        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                if(c[dat[i].clusterID].nbData == 1)
                {
                    ow[i] = 1.0;
                }
                else
                {
                    ow[i] = (w[i] / sumWeights) * (double) c[dat[i].clusterID].nbData;
                }
            }
        }
    }
}

static double CLUSTER_computeObjectWeightInClusterViaMedian(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t indK)
{
    if(dat == NULL || n < 2 || indN < 0 || indN > n || p < 1 || c == NULL || indK < 0)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double median[p]; // Median
        double w[n]; // Tmp weights
        double sumWeights = 0.0; // Sum of weights in cluster k
        double ow_tmp;

        double datPerCluster[p][c[indK].nbData];
        
        for(j=0;j<p;j++)
        {
            uint64_t m = 0;
            for(i=0;i<n;i++)
            {
                if(dat[i].clusterID == indK)
                {
                    datPerCluster[j][m] = dat[i].dim[j];
                    m++;
                }
            }
        }

        /*printf("BEF : ");
        for(j=0;j<c[k].nbData;j++)
        {
            printf("dim[1][%ld] = %lf, ", j, datPerCluster[0][j]);
        }
        SAY("");*/

        for(j=0;j<p;j++)
        {
            qsort((double *)&(datPerCluster[j]), c[indK].nbData, sizeof(double), cmpfunc);
        }

        /*printf("AFT : ");
        for(j=0;j<c[k].nbData;j++)
        {
            printf("dim[1][%ld] = %lf, ", j, datPerCluster[0][j]);
        }
        SAY("");*/

        // Compute the median
        for(j=0;j<p;j++)
        {
            if(!(c[indK].nbData % 2))
            {
                median[j] = (datPerCluster[j][(c[indK].nbData / 2)] + datPerCluster[j][(c[indK].nbData / 2) - 1]) / 2;
            }
            else
            {
                median[j] = datPerCluster[j][((c[indK].nbData + 1) / 2) - 1];
            }
            //SAY("median[%d][%ld] = %lf", l, j, median[l][j]);
        }


        // Compute tmp weights
        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == indK)
            {
                // Initialize tmp weights
                w[i] = 0.0;

                for(j=0;j<p;j++)
                {
                    w[i] += pow((dat[i].dim[j] - median[j]), 2.0); 
                }

                sumWeights += w[i];
            }
        }

        // Compute objects weights
        if(c[indK].nbData == 1)
        {
            ow_tmp = 1.0;
        }
        else
        {
            ow_tmp = (w[indN] / sumWeights) * (double) c[indK].nbData;
        }

        return ow_tmp;
    }
}

static double CLUSTER_computeObjectWeightInClusterViaMedian2(data *dat, uint64_t n, uint64_t indN, uint64_t p, cluster *c, uint32_t indK)
{
    if(dat == NULL || n < 2 || indN < 0 || indN > n || p < 1 || c == NULL || indK < 0)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        uint64_t i, j;
        uint32_t l;
        double median[p]; // Median
        double w; // Tmp weights
        double ow_tmp;
        double datPerCluster[p][c[indK].nbData];
        
        for(j=0;j<p;j++)
        {
            uint64_t m = 0;
            for(i=0;i<n;i++)
            {
                if(dat[i].clusterID == indK)
                {
                    datPerCluster[j][m] = dat[i].dim[j];
                    m++;
                }
            }
        }

        /*printf("BEF : ");
        for(j=0;j<c[k].nbData;j++)
        {
            printf("dim[1][%ld] = %lf, ", j, datPerCluster[0][j]);
        }
        SAY("");*/

        for(j=0;j<p;j++)
        {
            qsort((double *)&(datPerCluster[j]), c[indK].nbData, sizeof(double), cmpfunc);
        }

        /*printf("AFT : ");
        for(j=0;j<c[k].nbData;j++)
        {
            printf("dim[1][%ld] = %lf, ", j, datPerCluster[0][j]);
        }
        SAY("");*/

        // Compute the median
        for(j=0;j<p;j++)
        {
            if(!(c[indK].nbData % 2))
            {
                median[j] = (datPerCluster[j][(c[indK].nbData / 2)] + datPerCluster[j][(c[indK].nbData / 2) - 1]) / 2;
            }
            else
            {
                median[j] = datPerCluster[j][((c[indK].nbData + 1) / 2) - 1];
            }
            //SAY("median[%d][%ld] = %lf", l, j, median[l][j]);
        }


        // Compute tmp weights
        // Initialize tmp weights
        w = 0.0;

        for(j=0;j<p;j++)
        {
            w += pow((dat[indN].dim[j] - median[j]), 2.0); 
        }

        // Compute objects weights
        if(c[indK].nbData == 1)
        {
            ow_tmp = 1.0;
        }
        else
        {
            ow_tmp = w;
        }

        return ow_tmp;
    }
}

static void CLUSTER_computeObjectWeightsViaMedian3(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || ow == NULL)
    {
        ERR("Bad parameter");
    }
    else
    {
        // Improved Weiszfeld algorithm calculation
        uint64_t i, j;
        uint32_t l;
        double y[k][p]; // Median per cluster
        double eps = 1e-5; // Convergence criterion
        double w[n]; // Tmp weights 
        double sumWeights[k]; // Sum of weights per cluster
        double totWeightSum = 0.0; // Total sum of weights
        uint8_t nbIt = 0; // Number of iteration

        for(l=0;l<k;l++)
        {
            bool conv = false; // Has converged

            // Initialization of medians to the cluster average
            for(j=0;j<p;j++)
            {
                y[l][j] = c[l].centroid[j];
            }

            while(conv == false && nbIt < 100)
            {
                double sum1[p]; // 1 / (y - xi)²
                double sum2[p]; // xi / (y - xi)²
                double r[p]; // xi - y / (xi - y)²
                double T[p];
                double y1[p]; // Tmp median
                double convDist = 0.0;

                for(j=0;j<p;j++)
                {
                    sum1[j] = 0.0;
                    sum2[j] = 0.0;
                    r[j] = 0.0;
                }

                for(i=0;i<n;i++)
                {
                    if(dat[i].clusterID == l)
                    {
                        for(j=0;j<p;j++)
                        {
                            double tmp = pow((y[l][j] - dat[i].dim[j]), 2.0);
                            if(tmp == 0.0)
                                sum1[j] += 0.0;
                            else
                                sum1[j] += 1 / tmp;

                            if(tmp == 0)
                            {
                                sum2[j] += 0.0;
                            }
                            else
                            {
                                sum2[j] += dat[i].dim[j] / tmp;
                            }
    
                            tmp = pow((dat[i].dim[j] - y[l][j]), 2.0);
                            
                            if(tmp == 0)
                            {
                                r[j] += 0.0;
                            }
                            else
                            {
                                r[j] += (dat[i].dim[j] - y[l][j]) / tmp;
                            }
                        }  
                    }    
                }

                for(j=0;j<p;j++)
                {
                    if(sum1[j] == 0 || sum2[j] == 0)
                    {
                        T[j] = 0.0;
                    }
                    else
                    {
                        T[j] = (1 / sum1[j]) * sum2[j];
                    }

                    if(r[j] == 0)
                    {
                        y1[j] = (MAX(0, (1 - 0)) * T[j]) + (MIN(1, 0) * y[l][j]);
                    }
                    else
                    {
                        y1[j] = (MAX(0, (1 - (1 / pow(r[j], 2.0)))) * T[j]) + (MIN(1, (1 / pow(r[j], 2.0))) * y[l][j]);
                    }
                    convDist += pow((y[l][j] - y1[j]), 2.0);

                    // Update median 
                    y[l][j] = y1[j];    
                    //SAY("y[%d][%ld] = %lf", l, j, y[l][j]); 
                }

                // Test convergence
                if(convDist < eps)
                {
                    conv = true;
                }

                nbIt++;
            }
        }

        // Compute objects weights
        for(l=0;l<k;l++)
        {
            sumWeights[l] = 0.0;
        }

        // Compute tmp weights
        for(i=0;i<n;i++)
        {
            // Initialize tmp weights
            w[i] = 0.0;

            for(j=0;j<p;j++)
            {
                w[i] += pow((dat[i].dim[j] - y[dat[i].clusterID][j]), 2.0); 
            }

            sumWeights[dat[i].clusterID] += w[i];
            totWeightSum += w[i];
        }

        // Compute objects weights
        for(i=0;i<n;i++)
        {
            if(c[dat[i].clusterID].nbData == 1)
            {
                ow[i] = 1.0;
            }
            else
            {
                //ow[i] = (w[i] / sumWeights[dat[i].clusterID]) * (double) c[dat[i].clusterID].nbData;
                ow[i] = (w[i] / totWeightSum) * (double) n;
                //ow[i] = w[i];
            }
            //SAY("ow[%ld] = %lf, c = %d", i, ow[i], dat[i].clusterID);
        }
    }
}

static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            SSE += CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN); 
        }

        return SSE;
    }
}

static void CLUSTER_computeWSS(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double wss[k])
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;
        uint32_t l;

        for(l=0;l<k;l++)
        {
            wss[l] = 0.0;
        }

        for(i=0;i<n;i++)
        {
            wss[dat[i].clusterID] += CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN); 
        }
    }
}

static double CLUSTER_computeWeightedSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            SSE += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)&(fw[dat[i].clusterID]), ow[i]);
        }

        return SSE;
    }
}

static double CLUSTER_computeNkWeightedSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            SSE += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)&(fw[dat[i].clusterID]), ow[i])/(double)c[dat[i].clusterID].nbData;
        }

        return SSE;
    }
}

static double CLUSTER_computeSSEInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 0)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                SSE += CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN);
            }
        }

        return SSE;
    }
}

static double CLUSTER_computeWeightedSSEInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 0)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                SSE += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)&(fw[dat[i].clusterID]), ow[i]);
            }
        }

        return SSE;
    }
}

static double CLUSTER_computeNkWeightedSSEInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 0)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                SSE += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)&(fw[dat[i].clusterID]), ow[i]) / (double) c[dat[i].clusterID].nbData;
            }
        }

        return SSE;
    }
}

static double CLUSTER_computeWeightedSSE2(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double fw[k][p], double *ow)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL)
    {
        ERR("Bad parameter");
        return -1.0;
    }
    else
    {
        double SSE = 0.0;
        uint64_t i;

        for(i=0;i<n;i++)
        {
            if(dat[i].clusterID == k) 
            {
                SSE += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)&(fw[k]), ow[i]); 
            }
        }

        return SSE;
    }
}

static void CLUSTER_ComputeMatDistPointToPoint(data *dat, uint64_t n, uint64_t p, double dist[n][n])
{
    uint64_t i, j;
    // Create the distance matrix of i vs j
    for(i=0;i<n;i++)
        for (j=0;j<n;j++)
            dist[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);
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

static void CLUSTER_ComputeStdDevForDimInClust(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double sd[p])
{
    if(dat == NULL || n < 2)
    {
        ERR("Bad parameter");
    }
    else
    {
        uint64_t i, j;

        // Init sd array
        for (j=0;j<p;j++)
        {
            sd[j] = 0.0;
        }

        // Compute the sd of each dimension in the cluster 
        for (i=0;i<n;i++)
        {
            if(dat[i].clusterID == k)
            {
                for (j=0;j<p;j++)
                {
                    //sd[j] += fabs((dat[i].dim[j] - c[k].centroid[j])) / (double) c[k].nbData;
                    sd[j] += pow((dat[i].dim[j] - c[k].centroid[j]), 2.0) / (double) c[k].nbData;
                }
            }
        }

        for (j=0;j<p;j++)
        {
            sd[j] = sqrt(sd[j]);
        }
    }
}
