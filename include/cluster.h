/** @file cluster.h
 *  @brief Function prototypes for the k-means clustering.
 *
 *  This contains the function prototypes for the k-means
 *  clustering and eventually any macros, constants, or 
 *  global variables.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

#ifndef _CLUSTER_H
#define _CLUSTER_H

/* -- Includes -- */

/* libc includes. */
#include <stdint.h>
#include <stdbool.h>

/** @brief Contains clusters informations.
 *
 */
typedef struct _cluster
{
    uint32_t ind; // Cluster index
    double *centroid; // Centroid dimensions 
    uint64_t nbData; // Number of data in cluster
    double *fw; // Features weights in cluster
    void *head; // Pointer to the head of cluster chained list
} cluster;

/** @brief Contains data informations.
 *
 */
typedef struct _data
{
    uint64_t ind; // Data index
    double *dim; // Data dimensions
    uint32_t clusterID; // Cluster ID of data
    double ow; // Data weight
    void *pred; // Pointer to datum predecessor in cluster chained list
    void *succ; // Pointer to datum successor in cluster chained list
} data;

/** @brief Contains the different weights calculation methods.
 *
 */
typedef enum _eMethodType 
{
    METHOD_SILHOUETTE = 0,
    METHOD_SILHOUETTE_NK,
    METHOD_MEDIAN,
    METHOD_MEDIAN_NK,
    METHOD_MIN_DIST_CENTROID,
    METHOD_MIN_DIST_CENTROID_NK,
    METHOD_SUM_DIST_CENTROID,
    METHOD_DISPERSION,
    METHOD_OTHER
} eMethodType; 

/** @brief Computes the classical version of k-means  
 *         algorithm.
 *  
 *  This will compute the classical version of k-means  
 *  algorithm from 2 clusters to kmax clusters and with 
 *  nbRep random starts for each numbers of clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param kmin The minimum number of clusters 
 *               that will be computed.
 *  @param kmax The maximum number of clusters 
 *               that will be computed.
 *  @param nbRep The number of random starts for 
 *               each number of clusters.
 *  @return Void.
 */
void CLUSTER_computeKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmin, uint32_t kmax,uint32_t nbRep);

/** @brief Computes the features weighted  
 *         version of k-means algorithm.
 *
 *  This will compute the weighted (features/objects) 
 *  version of k-means algorithm from 2 clusters to 
 *  kmax clusters and with nbRep random starts for 
 *  each numbers of clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param kmin The minimum number of clusters 
 *               that will be computed.
 *  @param kmax The maximum number of clusters 
 *               that will be computed.
 *  @param nbRep The number of random starts for 
 *               each number of clusters.
 *  @param internalFeatureWeights The boolean that 
 *               specified if the features weights come 
 *               from internal computation or from a file.
 *  @param featureWeightsFile The string to the features 
 *               weights file.
 *  @param featureWeightsMethod The features calculation 
 *               method.
 *  @return Void.
 */
void CLUSTER_computeFeaturesWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmin, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, eMethodType featureWeightsMethod);

/** @brief Computes the weighted (features/objects) 
 *         version of k-means algorithm.
 *
 *  This will compute the weighted (features/objects) 
 *  version of k-means algorithm from 2 clusters to 
 *  kmax clusters and with nbRep random starts for 
 *  each numbers of clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param kmin The minimum number of clusters 
 *               that will be computed.
 *  @param kmax The maximum number of clusters 
 *               that will be computed.
 *  @param nbRep The number of random starts for 
 *               each number of clusters.
 *  @param internalFeatureWeights The boolean that 
 *               specified if the features weights come 
 *               from internal computation or from a file.
 *  @param featureWeightsFile The string to the features 
 *               weights file.
 *  @param featureWeightsMethod The features calculation 
 *               method.
 *  @param internalObjectWeights The boolean that 
 *               specified if the objects weights come 
 *               from internal computation or from a file.
 *  @param objectWeightsFile The string to the objects 
 *               weights file.
 *  @param objectWeightsMethod The objects calculation 
 *               method.
 *  @return Void.
 */
void CLUSTER_computeWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t kmin, uint32_t kmax,uint32_t nbRep, bool internalFeatureWeights, const char *featureWeightsFile, eMethodType featureWeightsMethod, bool internalObjectWeights, const char *objectWeightsFile, eMethodType objectWeightsMethod);

#endif /* _CLUSTER_H */
